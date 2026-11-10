"""RetroactiveTGNMemory: a PyG TGNMemory that can change the past.

TGN keeps one memory vector s_n per node and updates it event by event (in batches, "steps"):

    s_n  <-  F(s_n, s_o, e) = GRU([ s_n || s_o || x_e || phi(t_e - tau_n) ], s_n)          (eval semantics of PyG)

where e is the latest event of node n in the step, o its partner and tau_n the time of n's previous update. Every
update in a step reads the memory from *before* the step. If an event of an old step k0 is inserted (it arrived late),
deleted (a privacy request) or amended (a corrected filing), every later update that read a changed memory is wrong.
That set is the dependency cone C (proposal Eq. 2):

    (n, k) in C,  k' > k,  update (m, k') reads s_n   =>   (m, k') in C .

This class keeps a journal (undo/redo log) of the last `horizon` steps: for every update it stores the node, the
chosen event and the memory before and after. To repair, it rolls the memory back to step k0 and walks forward:

    exact     recompute exactly the updates in the cone; copy all other results from the journal       (= S*)
    replay    recompute every update after k0 (rollback + full replay; same result, used as a reference)
    eps       epsilon-repair: like exact, but once a recomputed memory differs from the journal by at most
              eps * ||old|| (relative L2) keep the old value and stop the change there (edit endpoints are
              always repaired)
    khop      StreamTGN-style static affected set: only nodes in a given mask (e.g. K hops around the edit)
              are recomputed; everything else is copied from the journal

Usage (inference only; train with a plain TGNMemory and load its weights):

    mem = RetroactiveTGNMemory.from_trained(model.memory, events, horizon=None)
    mem.warmup(0, stream.test0)                     # process the history in order
    mem.advance(step_events)                        # process the next step
    mem.edit(insert=late_events, delete=removed)    # change the past (steps < mem.step)
    stats = mem.repair("exact")                     # fix the memory; returns cost / cone size
    s, last_update = mem(n_id)                      # same interface as TGNMemory in eval mode
"""
import copy
import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch_geometric.nn import TGNMemory


class EventTable:
    """All events an experiment can ever see (originals plus, e.g., amended versions), indexed by event id.

    step[e] is the processing step an event belongs to (decided by its occurrence time; -1 = history)."""

    def __init__(self, src, dst, t, msg, step):
        self.src, self.dst, self.t, self.msg, self.step = src.long(), dst.long(), t.long(), msg.float(), step.long()
        ids = torch.arange(src.numel(), dtype=torch.long)
        self.key = self.t * (1 << 22) + ids                   # (time, id): a strict total order on events

    @property
    def num_events(self):
        return int(self.src.numel())


@dataclass
class StepRecord:
    """Journal entry of one step: which nodes were updated with which event, and their memory before/after."""
    nodes: torch.Tensor          # [n] sorted node ids
    other: torch.Tensor          # [n] partner of the chosen event
    eid: torch.Tensor            # [n] chosen (latest) event of the node in this step
    pre_mem: torch.Tensor        # [n, d]
    pre_lu: torch.Tensor         # [n]
    post_mem: torch.Tensor       # [n, d]
    post_lu: torch.Tensor        # [n]


@dataclass
class RepairStats:
    k0: int = -1                 # earliest repaired step
    steps: int = 0               # number of steps walked through
    recomputed: int = 0          # GRU node-updates recomputed (the cost; = |cone| for mode='exact')
    window_updates: int = 0      # all node-updates in the repaired window (= cost of rollback + full replay)
    skipped: int = 0             # eps-repair: recomputed updates whose change was below eps (kept old value)
    max_skipped_change: float = 0.0   # eps-repair: largest skipped ||new - old|| (r in Prop. 2)
    changed_nodes: int = 0       # nodes whose final memory differs from before the repair
    unrepairable: int = 0        # edits older than the journal horizon
    extra: dict = field(default_factory=dict)


class RetroactiveTGNMemory(TGNMemory):

    def __init__(self, *args, horizon: Optional[int] = None, clamp_dt: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon            # journal length in steps (None = keep everything)
        self.clamp_dt = clamp_dt          # use max(t_e - tau_n, 0): out-of-order events never get negative ages
        self.events: Optional[EventTable] = None
        self.journal = {}                 # step -> StepRecord
        self.batch_events = {}            # step -> sorted event ids currently part of that step
        self.step = 0                     # next step to process
        self.pending = {}                 # step -> True for steps changed by edit() since the last repair
        self.normal_updates = 0           # node-updates done by advance() (cost of normal processing)
        self.pad_single_rows = True       # see compute_update (switch off only to reproduce the float32 issue)
        self.use_index = True             # repair visits only the steps that read a changed memory (False: walk every step)

    # ------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_trained(cls, memory: TGNMemory, events: EventTable, horizon: Optional[int] = None, clamp_dt=True):
        m = cls(memory.num_nodes, memory.raw_msg_dim, memory.memory_dim, memory.time_dim,
                message_module=copy.deepcopy(memory.msg_s_module), aggregator_module=copy.deepcopy(memory.aggr_module),
                horizon=horizon, clamp_dt=clamp_dt)
        m.gru = copy.deepcopy(memory.gru)         # the memory updater: GRUCell (TGN), RNNCell or ContractiveRNNCell
        m.to(memory.memory.dtype)                 # keep the trained module's precision (e.g. float64 for tests)
        m.load_state_dict(memory.state_dict())
        m.requires_grad_(False)
        super(TGNMemory, m).train(False)  # plain eval mode (TGNMemory.train(False) would flush messages into memory)
        m.reset_state()
        m.events = events
        return m

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("RetroactiveTGNMemory is for inference; train a TGNMemory and load its weights")
        return super(TGNMemory, self).train(False)

    def reset_state(self):
        super().reset_state()
        self.journal, self.batch_events, self.pending = {}, {}, {}
        self.step, self.normal_updates = 0, 0

    def copy_state_from(self, other: "RetroactiveTGNMemory"):
        """Make this memory an independent copy of `other` (memory, journal, step). Journal records are never
        modified in place (repair replaces them), so sharing them is safe."""
        self.memory.copy_(other.memory); self.last_update.copy_(other.last_update)
        self.journal, self.batch_events = dict(other.journal), dict(other.batch_events)
        self.pending = dict(other.pending)
        self.step, self.normal_updates, self.events = other.step, other.normal_updates, other.events
        self.pad_single_rows, self.use_index = other.pad_single_rows, other.use_index
        return self

    # ------------------------------------------------------------------------------------------- the update F
    def choose(self, ev: torch.Tensor):
        """LastAggregator: every node touched by events `ev` is updated once, with its latest event.
        Returns nodes (sorted), partner and chosen event id. Ties in time are broken by the larger event id."""
        E = self.events
        node = torch.cat([E.src[ev], E.dst[ev]]); other = torch.cat([E.dst[ev], E.src[ev]]); eid = torch.cat([ev, ev])
        order = torch.argsort(E.key[eid], descending=True)
        _, first = np.unique(node[order].numpy(), return_index=True)
        sel = order[torch.from_numpy(first)]
        return node[sel], other[sel], eid[sel]

    @torch.no_grad()
    def compute_update(self, nodes, other, eid, mem=None, lu=None):
        """F for a set of (node, partner, event) triples, reading memory `mem` (default: the current memory).

        Batch invariance: BLAS uses a different kernel for single-row products, and with float32 the time encoding
        cos(w * dt + b) of a large dt (w * dt ~ 1e6, where float32 steps are 0.06) then differs by up to 0.06. We pad
        single-row updates to two rows, so a recomputed update is bitwise identical to the same update computed inside
        a full step (checked in tests/test_core.py)."""
        mem = self.memory if mem is None else mem
        lu = self.last_update if lu is None else lu
        single = nodes.numel() == 1 and self.pad_single_rows
        if single:
            nodes, other, eid = nodes.repeat(2), other.repeat(2), eid.repeat(2)
        E = self.events
        tt = E.t[eid]
        dt = tt - lu[nodes]
        if self.clamp_dt:
            dt = dt.clamp(min=0)
        t_enc = self.time_enc(dt.to(mem.dtype))
        msg = self.msg_s_module(mem[nodes], mem[other], E.msg[eid].to(mem.dtype), t_enc)
        new_mem = self.gru(msg, mem[nodes])
        new_lu = torch.maximum(tt, lu[nodes]) if self.clamp_dt else tt
        if single:
            new_mem, new_lu = new_mem[:1], new_lu[:1]
        return new_mem, new_lu

    # ------------------------------------------------------------------------------------------- normal processing
    @torch.no_grad()
    def warmup(self, lo: int, hi: int, batch_size: int = 200):
        """Process events lo..hi-1 in order, in fixed batches, without journaling (the history before the test)."""
        for b in range(lo, hi, batch_size):
            ev = torch.arange(b, min(b + batch_size, hi))
            nodes, other, eid = self.choose(ev)
            new_mem, new_lu = self.compute_update(nodes, other, eid)
            self.memory[nodes] = new_mem; self.last_update[nodes] = new_lu

    @torch.no_grad()
    def apply(self, ev: torch.Tensor) -> int:
        """One update round with events `ev`, without journaling and without moving to the next step (used by
        policies that process several rounds per step, e.g. a buffer releasing held-back events in order)."""
        if ev.numel() == 0:
            return 0
        nodes, other, eid = self.choose(ev.long())
        new_mem, new_lu = self.compute_update(nodes, other, eid)
        self.memory[nodes] = new_mem; self.last_update[nodes] = new_lu
        self.normal_updates += int(nodes.numel())
        return int(nodes.numel())

    @torch.no_grad()
    def advance(self, ev: torch.Tensor, journal: bool = True) -> int:
        """Process step `self.step` with events `ev` (all events the policy applies at this step). Returns the number
        of node-updates."""
        k = self.step
        ev = torch.sort(ev.long()).values if ev.numel() else ev.long()
        self.batch_events[k] = ev
        if ev.numel():
            nodes, other, eid = self.choose(ev)
            pre_m, pre_l = self.memory[nodes].clone(), self.last_update[nodes].clone()
            new_mem, new_lu = self.compute_update(nodes, other, eid)
            self.memory[nodes] = new_mem; self.last_update[nodes] = new_lu
        else:
            nodes = other = eid = torch.empty(0, dtype=torch.long)
            pre_m = new_mem = self.memory.new_empty(0, self.memory_dim)
            pre_l = new_lu = torch.empty(0, dtype=torch.long)
        if journal:
            self.journal[k] = StepRecord(nodes, other, eid, pre_m, pre_l, new_mem, new_lu)
        self.step += 1
        self.normal_updates += int(nodes.numel())
        if self.horizon is not None:                          # forget steps that are older than the horizon
            for old in [s for s in self.journal if s < self.step - self.horizon]:
                del self.journal[old]; self.batch_events.pop(old, None)
        return int(nodes.numel())

    # ------------------------------------------------------------------------------------------- edits
    def oldest_repairable_step(self) -> int:
        return min(self.journal) if self.journal else self.step

    def edit(self, insert: torch.Tensor = None, delete: torch.Tensor = None):
        """Change the content of past steps. Returns (unrepairable_insert, unrepairable_delete): events whose step is
        older than the journal (the caller decides what to do with them, e.g. apply late events now)."""
        E, oldest = self.events, self.oldest_repairable_step()
        bad_ins = bad_del = torch.empty(0, dtype=torch.long)
        for kind, ev in (("insert", insert), ("delete", delete)):
            if ev is None or ev.numel() == 0:
                continue
            ev = ev.long()
            st = E.step[ev]
            assert bool((st < self.step).all()), "edit() changes the past; steps >= mem.step have not been processed"
            ok = st >= oldest
            if kind == "insert": bad_ins = ev[~ok]
            else: bad_del = ev[~ok]
            for k in torch.unique(st[ok]).tolist():
                cur = self.batch_events.get(k, torch.empty(0, dtype=torch.long))
                chg = ev[ok][st[ok] == k]
                if kind == "insert":
                    new = torch.unique(torch.cat([cur, chg]))
                else:
                    new = cur[~torch.isin(cur, chg)]
                self.batch_events[k] = new
                self.pending[k] = True
        return bad_ins, bad_del

    # ------------------------------------------------------------------------------------------- repair
    @torch.no_grad()
    def repair(self, mode: str = "exact", eps: float = 0.0, khop_mask: Optional[torch.Tensor] = None,
               hook=None, delta: float = None, rho: float = None) -> RepairStats:
        """Repair the memory after edit(). `hook(k, nodes, other, eid, skipped, skip_diff, mem, lu)` is called once per
        step before the new values are written (mem/lu then hold the pre-step memory); used for certificates.

        Modes exact / eps / khop use the step index (_repair_indexed): only steps that contain an edit or read a changed
        memory are visited. 'replay', a hook, or use_index=False walk every step of the window (_repair_walk). Both give
        bit-identical memories and journals (tests/test_index.py).

        mode='cert' (delta-certified eps-repair, for a memory update that contracts with a proven constant `rho` in the
        inf-norm, e.g. ContractiveRNNCell): a change may be skipped only if the propagated error bound
        b_n <- rho * max(b_n, b_partner) + skipped change stays <= `delta`. Then every memory is within `delta` (inf-norm)
        of exact repair at all times (tests/test_models.py)."""
        if mode not in ("replay", "cert") and hook is None and self.use_index:
            return self._repair_indexed(mode, eps, khop_mask)
        return self._repair_walk(mode, eps, khop_mask, hook, delta, rho)

    @torch.no_grad()
    def _repair_walk(self, mode, eps, khop_mask, hook, delta=None, rho=None) -> RepairStats:
        """Roll the whole memory back to the earliest edited step and walk forward through every later step."""
        st = RepairStats()
        if not self.pending:
            return st
        k0, K = min(self.pending), self.step
        st.k0, st.steps = k0, K - k0
        mem, lu = self.memory, self.last_update
        before = mem.clone()
        for k in range(K - 1, k0 - 1, -1):                    # 1) roll back to the state before step k0
            rec = self.journal[k]
            if rec.nodes.numel():
                mem[rec.nodes] = rec.pre_mem; lu[rec.nodes] = rec.pre_lu
        dirty = torch.zeros(self.num_nodes, dtype=torch.bool) # memory differs from the old (journal) trajectory
        if mode == "cert":
            assert delta is not None and rho is not None and rho < 1, "cert needs delta and a contraction constant rho < 1"
            bnd = torch.zeros(self.num_nodes)                 # proven inf-norm error bound of every memory
            st.extra["max_bound"] = 0.0
        for k in range(k0, K):                                # 2) walk forward, recomputing what changed
            old, ev = self.journal[k], self.batch_events.get(k, torch.empty(0, dtype=torch.long))
            st.window_updates += int(old.nodes.numel())
            if ev.numel():
                nodes, other, eid = self.choose(ev)
            else:
                nodes = other = eid = torch.empty(0, dtype=torch.long)
            n_old = old.nodes.numel()
            if n_old:
                pos = torch.searchsorted(old.nodes, nodes).clamp(max=n_old - 1)
                in_old = old.nodes[pos] == nodes
            else:
                pos = torch.zeros_like(nodes); in_old = torch.zeros(nodes.numel(), dtype=torch.bool)
            same = in_old & (old.eid[pos] == eid) & (old.other[pos] == other) if n_old else in_old
            reads_dirty = dirty[nodes] | dirty[other]
            if mode == "replay":
                aff = torch.ones(nodes.numel(), dtype=torch.bool)
            elif mode == "khop":
                aff = ~same | (reads_dirty & khop_mask[nodes])
            else:
                aff = ~same | reads_dirty
            pre_m, pre_l = mem[nodes].clone(), lu[nodes].clone()
            post_m, post_l = torch.empty_like(pre_m), torch.empty_like(pre_l)
            keep = ~aff
            post_m[keep] = old.post_mem[pos[keep]]; post_l[keep] = old.post_lu[pos[keep]]
            new_dirty = torch.zeros(nodes.numel(), dtype=torch.bool)
            if bool(aff.any()):
                na = aff.nonzero().view(-1)
                new_m, new_l = self.compute_update(nodes[na], other[na], eid[na], mem, lu)
                st.recomputed += int(na.numel())
                if mode in ("eps", "cert"):
                    # a recomputed update may be skipped only if it uses the same event as before (it is not an
                    # endpoint of the edit) and its timestamp did not change
                    if n_old:
                        can = same[na] & (new_l == old.post_lu[pos[na]])
                        ref = old.post_mem[pos[na]]
                    else:                                    # nothing happened in this step before the edit
                        can = torch.zeros(na.numel(), dtype=torch.bool); ref = new_m
                    diff = (new_m - ref).norm(dim=1)
                    skip = can & (diff <= eps * ref.norm(dim=1))
                    if mode == "cert":                       # only while the proven bound stays within delta
                        skip &= rho * torch.maximum(bnd[nodes[na]], bnd[other[na]]) + diff <= delta
                    if bool(skip.any()):
                        st.skipped += int(skip.sum())
                        st.max_skipped_change = max(st.max_skipped_change, float(diff[skip].max()))
                        new_m[skip] = ref[skip]
                    new_dirty[na] = ~skip
                elif mode == "khop":
                    new_dirty[na] = True
                else:
                    new_dirty[na] = True                     # structural cone: a recomputed memory stays "changed"
                post_m[na] = new_m; post_l[na] = new_l
            if mode == "cert":                               # every update of the step propagates the bound
                add = torch.zeros(nodes.numel())
                if bool(aff.any()):
                    add[na[skip]] = diff[skip]
                bnd[nodes] = rho * torch.maximum(bnd[nodes], bnd[other]) + add
                st.extra["max_bound"] = max(st.extra["max_bound"], float(bnd.max()))
            # nodes updated in the old trajectory but not any more (their only event here was deleted)
            if n_old:
                gone = old.nodes[~torch.isin(old.nodes, nodes)]
            else:
                gone = torch.empty(0, dtype=torch.long)
            if hook is not None:
                skipped = torch.zeros(nodes.numel(), dtype=torch.bool); skip_diff = torch.zeros(nodes.numel())
                if mode == "eps" and bool(aff.any()):
                    skipped[na[skip]] = True; skip_diff[na[skip]] = diff[skip]
                hook(k, nodes, other, eid, skipped, skip_diff, mem, lu)
            mem[nodes] = post_m; lu[nodes] = post_l
            dirty[nodes] = new_dirty
            if gone.numel():
                dirty[gone] = True
            self.journal[k] = StepRecord(nodes, other, eid, pre_m, pre_l, post_m, post_l)
        self.pending = {}
        st.changed_nodes = int(((mem - before).abs().amax(dim=1) > 0).sum())
        return st

    @torch.no_grad()
    def _repair_indexed(self, mode, eps, khop_mask) -> RepairStats:
        """Repair without rolling back: visit only the steps that are edited or update a node whose memory changed.

        Invariant: a node is *clean* while its memory in the repaired (new) trajectory equals the old one. Every read
        of a recomputed update is the pre-step memory of a node updated in that step, which the journal holds for clean
        nodes; changed (dirty) nodes keep their current new value in self.memory. So self.memory only ever changes for
        dirty nodes (their old final value is kept in `orig` and restored if they become clean again, e.g. after an
        eps skip), and a step none of whose nodes is dirty is identical in both trajectories and is never visited."""
        st = RepairStats()
        if not self.pending:
            return st
        k0, K = min(self.pending), self.step
        st.k0, st.steps = k0, K - k0
        old_recs = {k: self.journal[k] for k in range(k0, K)}            # the old trajectory (records get replaced)
        sizes = torch.tensor([old_recs[k].nodes.numel() for k in range(k0, K)], dtype=torch.long)
        st.window_updates = int(sizes.sum())
        # step index of the old trajectory: for every node, the window steps where it is updated (CSR)
        alln = torch.cat([old_recs[k].nodes for k in range(k0, K)]) if st.window_updates else torch.empty(0, dtype=torch.long)
        allk = torch.repeat_interleave(torch.arange(k0, K), sizes)
        order = torch.argsort(alln, stable=True)
        sk = allk[order].numpy()
        ptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(np.bincount(alln.numpy(), minlength=self.num_nodes))

        def node_steps(x):
            return sk[ptr[x]:ptr[x + 1]]

        mem, lu = self.memory, self.last_update
        dirty = torch.zeros(self.num_nodes, dtype=torch.bool)
        orig = {}                                                        # node -> (old final memory, old last update)

        def backup(xs):
            for x in xs.tolist():
                if x not in orig:
                    orig[x] = (mem[x].clone(), lu[x].clone())

        def old_before(x, k):
            """Memory and last update of node x just before step k in the old trajectory (x not updated at k)."""
            ks = node_steps(x)
            i = int(np.searchsorted(ks, k))
            if i > 0:
                r = old_recs[int(ks[i - 1])]; q = int(torch.searchsorted(r.nodes, torch.tensor(x)))
                return r.post_mem[q], r.post_lu[q]
            if i < len(ks):
                r = old_recs[int(ks[i])]; q = int(torch.searchsorted(r.nodes, torch.tensor(x)))
                return r.pre_mem[q], r.pre_lu[q]
            return (orig[x] if x in orig else (mem[x], lu[x]))           # not updated in the window at all

        heap = sorted(self.pending)
        queued = set(heap)
        visited = 0

        def push_next(xs, k):
            for x in xs.tolist():
                ks = node_steps(x)
                i = int(np.searchsorted(ks, k, side="right"))
                if i < len(ks):
                    k2 = int(ks[i])
                    if k2 not in queued:
                        queued.add(k2); heapq.heappush(heap, k2)

        while heap:
            k = heapq.heappop(heap)
            visited += 1
            old = old_recs[k]
            if k in self.pending:
                ev = self.batch_events.get(k, torch.empty(0, dtype=torch.long))
                nodes, other, eid = self.choose(ev) if ev.numel() else (torch.empty(0, dtype=torch.long),) * 3
            else:
                nodes, other, eid = old.nodes, old.other, old.eid
            n_old = old.nodes.numel()
            if n_old:
                pos = torch.searchsorted(old.nodes, nodes).clamp(max=n_old - 1)
                in_old = old.nodes[pos] == nodes
            else:
                pos = torch.zeros_like(nodes); in_old = torch.zeros(nodes.numel(), dtype=torch.bool)
            same = in_old & (old.eid[pos] == eid) & (old.other[pos] == other) if n_old else in_old
            reads_dirty = dirty[nodes] | dirty[other]
            if mode == "khop":
                aff = ~same | (reads_dirty & khop_mask[nodes])
            else:
                aff = ~same | reads_dirty
            # pre-step memory of every node of the step in the new trajectory
            dn = dirty[nodes]
            pre_m = mem.new_empty(nodes.numel(), self.memory_dim); pre_l = torch.empty(nodes.numel(), dtype=torch.long)
            pre_m[dn] = mem[nodes[dn]]; pre_l[dn] = lu[nodes[dn]]
            c1 = ~dn & in_old
            pre_m[c1] = old.pre_mem[pos[c1]]; pre_l[c1] = old.pre_lu[pos[c1]]
            for i in (~dn & ~in_old).nonzero().view(-1).tolist():
                m_, l_ = old_before(int(nodes[i]), k)
                pre_m[i] = m_; pre_l[i] = l_
            post_m, post_l = torch.empty_like(pre_m), torch.empty_like(pre_l)
            keep = ~aff
            post_m[keep] = old.post_mem[pos[keep]]; post_l[keep] = old.post_lu[pos[keep]]
            new_dirty = torch.zeros(nodes.numel(), dtype=torch.bool)
            if bool(aff.any()):
                na = aff.nonzero().view(-1)
                loc_other = torch.searchsorted(nodes, other[na])          # partners are nodes of the same step
                new_m, new_l = self.compute_update(na, loc_other, eid[na], pre_m, pre_l)
                st.recomputed += int(na.numel())
                if mode == "eps":
                    if n_old:
                        can = same[na] & (new_l == old.post_lu[pos[na]])
                        ref = old.post_mem[pos[na]]
                    else:
                        can = torch.zeros(na.numel(), dtype=torch.bool); ref = new_m
                    diff = (new_m - ref).norm(dim=1)
                    skip = can & (diff <= eps * ref.norm(dim=1))
                    if bool(skip.any()):
                        st.skipped += int(skip.sum())
                        st.max_skipped_change = max(st.max_skipped_change, float(diff[skip].max()))
                        new_m[skip] = ref[skip]
                    new_dirty[na] = ~skip
                else:
                    new_dirty[na] = True                                 # structural cone (exact) / recomputed (khop)
                post_m[na] = new_m; post_l[na] = new_l
            gone = old.nodes[~torch.isin(old.nodes, nodes)] if n_old else torch.empty(0, dtype=torch.long)
            # write the new values: dirty nodes hold their new value, nodes that became clean get their old one back
            becomes = nodes[new_dirty]
            backup(becomes)
            mem[becomes] = post_m[new_dirty]; lu[becomes] = post_l[new_dirty]
            for x in nodes[dn & ~new_dirty].tolist():
                mem[x], lu[x] = orig[x]
            dirty[nodes] = new_dirty
            if gone.numel():                                             # updated before, not any more
                gc = gone[~dirty[gone]]
                if gc.numel():
                    gpos = torch.searchsorted(old.nodes, gc)
                    backup(gc)
                    mem[gc] = old.pre_mem[gpos]; lu[gc] = old.pre_lu[gpos]
                dirty[gone] = True
            self.journal[k] = StepRecord(nodes, other, eid, pre_m, pre_l, post_m, post_l)
            push_next(torch.cat([becomes, gone]), k)
        self.pending = {}
        st.changed_nodes = sum(1 for x, (m_, _) in orig.items() if not torch.equal(mem[x], m_))
        st.extra["visited_steps"] = visited
        return st

    # ------------------------------------------------------------------------------------------- helpers
    def journal_bytes(self) -> int:
        tot = 0
        for r in self.journal.values():
            tot += sum(x.numel() * x.element_size() for x in (r.nodes, r.other, r.eid, r.pre_mem, r.pre_lu, r.post_mem, r.post_lu))
        return tot


def khop_mask_for(events: EventTable, seeds: torch.Tensor, ev_window: torch.Tensor, K: int, num_nodes: int):
    """Nodes within K hops of `seeds` in the static interaction graph of the events `ev_window` (StreamTGN-style
    affected set: it ignores time order, so it can miss parts of the cone and include nodes outside it)."""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[seeds] = True
    s, d = events.src[ev_window], events.dst[ev_window]
    for _ in range(K):
        hit = mask[s] | mask[d]
        mask[s[hit]] = True; mask[d[hit]] = True
    return mask
