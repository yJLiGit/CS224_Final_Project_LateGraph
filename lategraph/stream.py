"""Streaming evaluation of repair policies when test events arrive late.

Protocol (as in PyG's TGN evaluation, plus a second clock):
  * test events are processed in steps; step j covers occurrence times [T_j, T_{j+1}) (data.BatchGrid)
  * before step j is scored, the system has received exactly the events with receipt step r_e <= j
    (r_e = first step with T_j > a_e, never earlier than own step + 1)
  * at step j every policy turns what it has received into a memory, then scores the queries of step j:
    the true destination against 100 random and 100 "historical" negatives (half from the user's past items)
  * all policies share the trained weights, the negatives and an occurrence-ordered neighbour store, so the
    differences come only from how each policy handles the memory

Policies
  oracle          every event on time (an upper bound no deployed system has)
  exact           insert late events into their own step and repair the dependency cone (= S*)
  replay          same state as exact, by rolling back and replaying every update after the earliest late event
  eps:<x>         epsilon-repair with threshold x
  khop:<K>        StreamTGN-style: recompute only nodes within K static hops of the late events' endpoints
  arrival         apply late events when they arrive, as part of the current step (their own timestamps)
  arrival_proc    same, but late events are re-stamped with their arrival time
  drop            ignore events that arrive after their step
  buffer:<h>      hold events for h hours (watermark), then apply them in occurrence order
  Suffix @H limits the journal to H steps (older late events are applied on arrival), e.g. exact@20.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import BatchGrid
from .memory import EventTable, RetroactiveTGNMemory, khop_mask_for
from .neighbors import TimeOrderedNeighborStore


# ------------------------------------------------------------------------------------------------- setup
@dataclass
class StreamSetup:
    stream: object
    grid: BatchGrid
    table: EventTable
    arrival: torch.Tensor
    own: torch.Tensor                 # [E] own step of each event (-1 = history)
    recv: torch.Tensor                # [E] receipt step (-1 = history)
    batches: list                     # own step -> event ids
    arrivals: list                    # receipt step -> event ids (only steps < J)
    late: torch.Tensor                # [E] bool: received after its own step's successor
    info: dict = field(default_factory=dict)
    queries: list = None              # own step -> event ids used as queries (default: all events of the step)
    replaces: dict = None             # amended version -> the version it replaces (SEC Form 4/A)
    is_new: torch.Tensor = None       # [E] bool: event is an amended version
    is_old: torch.Tensor = None       # [E] bool: event is replaced by an amendment later


def make_setup(S, arrival: torch.Tensor, bq: int = 200, max_steps=None, start=None, grid=None, replaces=None,
               dedupe_queries=False) -> StreamSetup:
    start = S.test0 if start is None else start
    grid = grid if grid is not None else BatchGrid(S.t, start, bq, max_steps)
    E = S.num_events
    own = torch.full((E,), -1, dtype=torch.long)
    test = torch.arange(start, E)
    own[test] = grid.step_of_time(S.t[test])
    inwin = (own >= 0) & (own < grid.J)
    recv = torch.full((E,), -1, dtype=torch.long)
    recv[inwin] = grid.receipt_step(arrival[inwin], own[inwin])
    table = EventTable(S.src, S.dst, S.t, S.msg, own)
    ids = inwin.nonzero().view(-1)
    o, r = own[ids], recv[ids]
    batches = [ids[o == j] for j in range(grid.J)]
    arrivals = [ids[r == j] for j in range(grid.J)]
    late = torch.zeros(E, dtype=torch.bool); late[ids] = r > o + 1
    lag_steps = (r - o - 1)[r > o + 1]
    info = dict(J=grid.J, test_events=int(ids.numel()), late_frac=float(late[ids].float().mean()),
                never_received=int((r >= grid.J).sum()),
                late_steps_median=float(lag_steps.float().median()) if lag_steps.numel() else 0.0,
                late_steps_p90=float(lag_steps.float().quantile(0.9)) if lag_steps.numel() else 0.0,
                late_steps_max=int(lag_steps.max()) if lag_steps.numel() else 0)
    is_new = torch.zeros(E, dtype=torch.bool); is_old = torch.zeros(E, dtype=torch.bool)
    if replaces:
        is_new[torch.tensor(list(replaces.keys()), dtype=torch.long)] = True
        is_old[torch.tensor(list(replaces.values()), dtype=torch.long)] = True
    queries = []
    for b in batches:
        q = b[~is_old[b]]                                   # what really happened: final versions only
        if dedupe_queries and q.numel():
            key = S.src[q] * S.num_nodes + S.dst[q]
            _, first = np.unique(key.numpy(), return_index=True)
            q = q[torch.from_numpy(np.sort(first))]
        queries.append(q)
    info["queries"] = int(sum(q.numel() for q in queries))
    if replaces:
        info["amendments"] = len(replaces)
    return StreamSetup(S, grid, table, arrival, own, recv, batches, arrivals, late, info, queries, replaces, is_new, is_old)


def sample_negatives(S, setup: StreamSetup, n_neg=100, seed=1234):
    """Per step: [Q, n_neg] uniform negatives and [Q, n_neg] TGB-style negatives (half from the user's pre-test items)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    nd = S.dst_max - S.dst_min + 1
    sh, dh = S.src[:S.test0].numpy(), S.dst[:S.test0].numpy()
    order = np.lexsort((dh, sh)); sh, dh = sh[order], dh[order]
    keep = np.r_[True, (sh[1:] != sh[:-1]) | (dh[1:] != dh[:-1])]
    sh, dh = sh[keep], dh[keep]
    bounds = np.r_[0, np.flatnonzero(sh[1:] != sh[:-1]) + 1, len(sh)]
    hist = {int(sh[a]): dh[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    rand, histn = [], []
    for ev in setup.queries:
        pos = S.dst[ev]
        off = torch.randint(1, nd, (pos.numel(), n_neg), generator=g)
        r = (pos.view(-1, 1) - S.dst_min + off) % nd + S.dst_min        # never the true item
        h = ((pos.view(-1, 1) - S.dst_min + torch.randint(1, nd, (pos.numel(), n_neg), generator=g)) % nd + S.dst_min)
        for i, (u, v) in enumerate(zip(S.src[ev].tolist(), pos.tolist())):
            hd = hist.get(u)
            if hd is None: continue
            hd = hd[hd != v]
            k = min(len(hd), n_neg // 2)
            if k: h[i, :k] = torch.from_numpy(rng.choice(hd, k, replace=False))
        rand.append(r); histn.append(h)
    return rand, histn


@torch.no_grad()
def score_pairs(model, mem: RetroactiveTGNMemory, store: TimeOrderedNeighborStore, events: EventTable,
                q_src: torch.Tensor, q_dst: torch.Tensor, now: int) -> torch.Tensor:
    n_id = torch.cat([q_src, q_dst]).unique()
    all_id, edge_index, e_id = store(n_id)
    assoc = torch.empty(mem.num_nodes, dtype=torch.long); assoc[all_id] = torch.arange(all_id.numel())
    z = model.embed(mem.memory[all_id], mem.last_update[all_id], edge_index, events.t[e_id], events.msg[e_id], now=now)
    return model.score(z[assoc[q_src]], z[assoc[q_dst]])


def warm_start(model, S, table: EventTable, store_size=10, horizon=None):
    """Memory and neighbour store after processing the history (train + val) in order."""
    mem = RetroactiveTGNMemory.from_trained(model.memory, table, horizon=horizon)
    mem.warmup(0, S.test0)
    store = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, store_size)
    store.insert(torch.arange(S.test0))
    return mem, store


# ------------------------------------------------------------------------------------------------- policies
class Policy:
    def __init__(self, name: str, model, setup: StreamSetup, base_mem: RetroactiveTGNMemory, base_store):
        self.name, self.setup, self.model = name, setup, model
        spec, _, hor = name.partition("@")
        self.kind, _, arg = spec.partition(":")
        self.arg = float(arg) if arg else None
        horizon = int(hor) if hor else None
        self.events = setup.table
        self.mem = RetroactiveTGNMemory.from_trained(model.memory, self.events, horizon=horizon).copy_state_from(base_mem)
        self.mem.horizon = horizon
        self.mem.normal_updates = 0
        self.store = base_store.clone()
        self.journaled = self.kind in ("exact", "replay", "eps", "khop")
        self.repair_updates = 0; self.sec = 0.0; self.fallback = 0; self.repairs = 0; self.skipped = 0
        self.max_window = 0
        if self.kind == "buffer":
            self.W = self.arg * 3600.0
            self.pending = torch.empty(0, dtype=torch.long)
        if self.kind == "arrival_proc":           # its own copy of the clock: late events get re-stamped
            tb = setup.table
            self.events = EventTable(tb.src, tb.dst, tb.t.clone(), tb.msg, tb.step)
            self.mem.events = self.events
            self.store.key = self.store.key.clone()

    def step(self, j: int):
        """Bring the memory to 'steps 0..j-1 processed' using what this policy has received by step j."""
        if j == 0:
            return
        t0 = time.perf_counter()
        su = self.setup
        R = su.arrivals[j]
        own = su.own
        rep = su.replaces
        if self.kind == "oracle":                   # hindsight: every event on time, final (corrected) versions only
            ev = su.batches[j - 1]
            if rep:
                ev = ev[~su.is_old[ev]]
            self.store.insert(ev); self.mem.advance(ev, journal=False)
        elif self.kind in ("arrival", "arrival_proc", "arrival_logfix"):
            if self.kind == "arrival_proc":
                lt = R[own[R] < j - 1]
                if lt.numel():
                    new_t = int(su.grid.T[j]) - 1
                    self.events.t[lt] = new_t
                    self.events.key[lt] = new_t * (1 << 22) + lt
                    self.store.key[lt] = self.events.key[lt]
            if self.kind == "arrival_logfix" and rep:   # amendments: drop the old version from the log only
                nv = R[su.is_new[R]]
                if nv.numel():
                    self.store.remove(torch.tensor([rep[int(e)] for e in nv], dtype=torch.long))
            self.store.insert(R); self.mem.advance(R, journal=False)
        elif self.kind == "drop":
            ev = R[own[R] == j - 1]
            self.store.insert(ev); self.mem.advance(ev, journal=False)
        elif self.kind == "buffer":
            self._buffer_step(j, R)
        else:                                       # exact / replay / eps / khop
            delete = torch.empty(0, dtype=torch.long)
            if rep:                                 # an amendment replaces an earlier version
                if not hasattr(self, "cancelled"):
                    self.cancelled = torch.zeros(su.table.num_events, dtype=torch.bool)
                R = R[~self.cancelled[R]]
                nv = R[su.is_new[R]]
                if nv.numel():
                    old = torch.tensor([rep[int(e)] for e in nv], dtype=torch.long)
                    have = self.store.alive[old]
                    self.cancelled[old[~have]] = True       # not received yet: ignore it when it arrives
                    R = R[~self.cancelled[R]]
                    delete = old[have]
                    self.store.remove(delete)
            late, ontime = R[own[R] < j - 1], R[own[R] == j - 1]
            self.store.insert(R)
            extra = torch.empty(0, dtype=torch.long)
            past_del = delete[own[delete] < j - 1] if delete.numel() else delete
            if late.numel() or past_del.numel():
                bad, bad_del = self.mem.edit(insert=late, delete=past_del)
                self.fallback += int(bad.numel()) + int(bad_del.numel())
                extra = bad
                mask = None
                if self.kind == "khop":
                    chg = torch.cat([late, past_del])
                    k0 = int(own[chg].min())
                    wins = [self.mem.batch_events[k] for k in range(k0, j - 1) if k in self.mem.batch_events]
                    win = torch.cat(wins) if wins else chg
                    seeds = torch.cat([self.events.src[chg], self.events.dst[chg]]).unique()
                    mask = khop_mask_for(self.events, seeds, win, int(self.arg), self.mem.num_nodes)
                mode = {"exact": "exact", "replay": "replay", "eps": "eps", "khop": "khop"}[self.kind]
                st = self.mem.repair(mode, eps=self.arg or 0.0, khop_mask=mask)
                self.repair_updates += st.recomputed; self.skipped += st.skipped
                self.repairs += 1; self.max_window = max(self.max_window, st.steps)
            self.mem.advance(torch.cat([ontime, extra]), journal=True)
        self.sec += time.perf_counter() - t0

    def _buffer_step(self, j, R):
        su, own, t = self.setup, self.setup.own, self.events.t
        wm = float(su.grid.T[j]) - self.W
        prev_wm = float(su.grid.T[j - 1]) - self.W
        if R.numel():
            too_late = t[R].double() < prev_wm      # its release time has passed already: apply it now
            if bool(too_late.any()):
                self.mem.apply(R[too_late]); self.store.insert(R[too_late])
            self.pending = torch.cat([self.pending, R[~too_late]])
        if self.pending.numel():
            ready = t[self.pending].double() < wm
            ev = self.pending[ready]; self.pending = self.pending[~ready]
            if ev.numel():
                ev = ev[torch.argsort(ev)]
                for b in torch.unique(own[ev]).tolist():   # release in occurrence order, step by step
                    self.mem.apply(ev[own[ev] == b])
                self.store.insert(ev)
        self.mem.step = j

    @property
    def total_updates(self):
        return self.mem.normal_updates + self.repair_updates


# ------------------------------------------------------------------------------------------------- evaluation
def evaluate_stream(model, setup: StreamSetup, policies: list, negs_rand, negs_hist, ref: str = "exact",
                    log=print, log_every=25, strata: dict = None):
    """Run all policies in lockstep over the test steps. Returns (summary dict, per-query arrays dict)."""
    S, J = setup.stream, setup.grid.J
    names = [p.name for p in policies]
    ref_i = names.index(ref) if ref in names else None
    acc = {n: dict(rr_r=[], rr_h=[], pos=[], top=[], div=[], agree=[], dlogit=[]) for n in names}
    t0 = time.time()
    for j in range(J):
        for p in policies:
            p.step(j)
        ev = setup.queries[j]
        if ev.numel() == 0:
            continue
        q_src, q_pos = S.src[ev], S.dst[ev]
        cand = torch.cat([q_pos.view(-1, 1), negs_rand[j], negs_hist[j]], 1)
        Q, C = cand.shape
        now = int(setup.grid.T[j])
        logits = {}
        for p in policies:
            lg = score_pairs(model, p.mem, p.store, p.events, q_src.repeat_interleave(C), cand.reshape(-1), now).view(Q, C)
            logits[p.name] = lg
            pos = lg[:, :1]
            nr, nh = lg[:, 1:101], lg[:, 101:]
            a = acc[p.name]
            a["rr_r"].append((1.0 / (1 + (nr > pos).sum(1) + 0.5 * (nr == pos).sum(1))).numpy())
            a["rr_h"].append((1.0 / (1 + (nh > pos).sum(1) + 0.5 * (nh == pos).sum(1))).numpy())
            a["pos"].append(pos.squeeze(1).numpy())
            a["top"].append(cand.gather(1, lg[:, :101].argmax(1, keepdim=True)).squeeze(1).numpy())   # top item id
        if ref_i is not None:
            rp = policies[ref_i]
            qn = torch.cat([q_src, q_pos]).unique()
            rm = rp.mem.memory[qn]
            for p in policies:
                a = acc[p.name]
                a["div"].append(float((p.mem.memory[qn] - rm).norm() / (rm.norm() + 1e-12)))
                a["agree"].append(float((a["top"][-1] == acc[ref]["top"][-1]).mean()))
                a["dlogit"].append(float((logits[p.name] - logits[ref]).abs().mean()))
        if log and (j % log_every == 0 or j == J - 1):
            msg = " ".join(f"{n}={np.concatenate(acc[n]['rr_r']).mean():.4f}" for n in names)
            log(f"[{time.time() - t0:5.0f}s] step {j + 1}/{J} MRR {msg}")
    oracle_updates = None
    for p in policies:
        if p.kind == "oracle":
            oracle_updates = p.mem.normal_updates
    summary = dict(info=setup.info, methods={})
    arrays = {}
    for p in policies:
        a = acc[p.name]
        rr_r, rr_h = np.concatenate(a["rr_r"]), np.concatenate(a["rr_h"])
        s = dict(mrr=float(rr_r.mean()), mrr_hist=float(rr_h.mean()), total_updates=int(p.total_updates),
                 normal_updates=int(p.mem.normal_updates), repair_updates=int(p.repair_updates), sec=p.sec,
                 repairs=p.repairs, fallback=p.fallback, skipped=p.skipped, max_window=p.max_window)
        if oracle_updates:
            s["cost_x_normal"] = p.total_updates / oracle_updates
        if p.journaled:
            s["journal_MB"] = p.mem.journal_bytes() / 2**20
        if ref_i is not None:
            s.update(mem_div=float(np.mean(a["div"])), top1_agree=float(np.mean(a["agree"])), dlogit=float(np.mean(a["dlogit"])))
        summary["methods"][p.name] = s
        arrays[f"rr_r__{p.name}"] = rr_r; arrays[f"rr_h__{p.name}"] = rr_h
        arrays[f"pos__{p.name}"] = np.concatenate(a["pos"]).astype(np.float32)
        arrays[f"top__{p.name}"] = np.concatenate(a["top"]).astype(np.int32)
    arrays["query_event"] = torch.cat([b for b in setup.queries if b.numel()]).numpy()
    return summary, arrays


def in_transit_flags(setup: StreamSetup) -> np.ndarray:
    """For each query (in step order): does its user have an event that already happened but has not arrived yet?"""
    S, own, recv = setup.stream, setup.own, setup.recv
    late_ev = setup.late.nonzero().view(-1)
    lu = S.src[late_ev].numpy(); lo = own[late_ev].numpy(); lr = recv[late_ev].numpy()
    by_user = {}
    for u, o, r in zip(lu, lo, lr):
        by_user.setdefault(int(u), []).append((int(o), int(r)))
    flags = []
    for j, ev in enumerate(setup.queries):
        for u in S.src[ev].tolist():
            flags.append(any(o < j < r for o, r in by_user.get(u, ())))
    return np.array(flags, dtype=bool)
