"""The other memory models: the contractive RNN cell really is a contraction, and exact repair stays exact for the RNN
(JODIE / DyRep updater) and the contractive cell, because their updates read the same memories as TGN's GRU."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import synthetic_stream                                 # noqa: E402
from lategraph.data import BatchGrid                                   # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory          # noqa: E402
from lategraph.model import TGN, TGNConfig, ContractiveRNNCell         # noqa: E402


def test_contractive_cell_is_a_contraction_for_any_input():
    torch.manual_seed(0)
    d, extra = 16, 12
    cell = ContractiveRNNCell(2 * d + extra, d, a_min=0.3, c=0.6)
    for p in cell.parameters():
        torch.nn.init.normal_(p, std=2.0)                 # large weights: the constraint has to do the work
    rho = cell.rho()
    assert rho <= 1 - 0.3 * (1 - 0.6) + 1e-6
    for _ in range(200):
        s, p, x = torch.randn(64, d) * 3, torch.randn(64, d) * 3, torch.randn(64, extra)
        s2, p2 = s + torch.randn(64, d) * 0.5, p + torch.randn(64, d) * 0.5
        out1 = cell(torch.cat([s, p, x], 1), s); out2 = cell(torch.cat([s2, p2, x], 1), s2)
        lhs = (out1 - out2).abs().amax(1)
        rhs = rho * torch.maximum((s - s2).abs().amax(1), (p - p2).abs().amax(1))
        assert bool((lhs <= rhs + 1e-6).all())


@pytest.mark.parametrize("updater,emb", [("rnn", "proj"), ("crnn", "attn")])
def test_exact_repair_is_exact_for_other_updaters(updater, emb):
    S = synthetic_stream(seed=5)
    S.msg = S.msg.double()
    torch.manual_seed(0)
    model = TGN(S.num_nodes, S.msg_dim, TGNConfig(updater=updater, emb=emb)).double()
    model.memory.reset_state()                            # before eval(): PyG keeps float32 message placeholders otherwise
    model.eval()
    grid = BatchGrid(S.t, S.test0, 20, 20)
    step = torch.full((S.num_events,), -1, dtype=torch.long); test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    win = test[step[test] < grid.J]
    rng = np.random.default_rng(0)
    base = RetroactiveTGNMemory.from_trained(model.memory, table); base.warmup(0, S.test0)
    ck, content = [], []
    for k in range(grid.J):
        ev = win[step[win] == k]; ck.append((base.memory.clone(), base.last_update.clone())); content.append(ev); base.advance(ev)
    work = RetroactiveTGNMemory.from_trained(model.memory, table); ref = RetroactiveTGNMemory.from_trained(model.memory, table)
    for trial in range(20):
        dele = torch.from_numpy(rng.choice(win.numpy(), 2, replace=False)).long()
        work.copy_state_from(base); work.edit(delete=dele); work.repair("exact")
        k0 = int(step[dele].min()); ref.memory.copy_(ck[k0][0]); ref.last_update.copy_(ck[k0][1])
        for k in range(k0, grid.J):
            ref.apply(content[k][~torch.isin(content[k], dele)])
        assert (work.memory - ref.memory).abs().max() <= 1e-12


def test_rigorous_eps_bound_holds_for_contractive_memory():
    """eps-repair on a contractive memory: the inf-norm error of every node stays below the propagated bound
    b_n <- rho * max(b_n, b_partner) + skipped change, and below the global bound r / (1 - rho)."""
    S = synthetic_stream(seed=7)
    torch.manual_seed(1)
    model = TGN(S.num_nodes, S.msg_dim, TGNConfig(updater="crnn", crnn_amin=0.4, crnn_c=0.5)).eval()
    for p in model.memory.gru.parameters():
        torch.nn.init.normal_(p, std=0.5)
    rho = model.memory.gru.rho()
    grid = BatchGrid(S.t, S.test0, 20, 25)
    step = torch.full((S.num_events,), -1, dtype=torch.long); test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    win = test[step[test] < grid.J]
    base = RetroactiveTGNMemory.from_trained(model.memory, table); base.warmup(0, S.test0)
    for k in range(grid.J):
        base.advance(win[step[win] == k])
    rng = np.random.default_rng(3); a = RetroactiveTGNMemory.from_trained(model.memory, table)
    skipped_any = False
    for trial in range(30):
        dele = torch.from_numpy(rng.choice(win.numpy(), 3, replace=False)).long()
        a.copy_state_from(base); a.edit(delete=dele); a.repair("exact"); exact = a.memory.clone()
        b = torch.zeros(S.num_nodes)

        def hook(k, nodes, other, eid, skipped, skip_diff, mem_, lu_):
            b[nodes] = rho * torch.maximum(b[nodes], b[other]) + skip_diff
        a.copy_state_from(base); a.edit(delete=dele); st = a.repair("eps", eps=0.3, hook=hook)
        skipped_any |= st.skipped > 0
        err = (a.memory - exact).abs().amax(1)
        assert bool((err <= b + 1e-6).all())
        assert float(err.max()) <= st.max_skipped_change / (1 - rho) + 1e-6
    assert skipped_any


def test_delta_certified_repair_keeps_every_memory_within_delta():
    S = synthetic_stream(seed=8)
    torch.manual_seed(2)
    model = TGN(S.num_nodes, S.msg_dim, TGNConfig(updater="crnn", crnn_amin=0.4, crnn_c=0.5)).eval()
    for p in model.memory.gru.parameters():
        torch.nn.init.normal_(p, std=0.5)
    rho = model.memory.gru.rho()
    grid = BatchGrid(S.t, S.test0, 20, 25)
    step = torch.full((S.num_events,), -1, dtype=torch.long); test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    win = test[step[test] < grid.J]
    base = RetroactiveTGNMemory.from_trained(model.memory, table); base.warmup(0, S.test0)
    for k in range(grid.J):
        base.advance(win[step[win] == k])
    rng = np.random.default_rng(4); a = RetroactiveTGNMemory.from_trained(model.memory, table)
    for delta in (0.02, 0.2):
        skipped = 0
        for trial in range(15):
            dele = torch.from_numpy(rng.choice(win.numpy(), 3, replace=False)).long()
            a.copy_state_from(base); a.edit(delete=dele); se = a.repair("exact"); exact = a.memory.clone()
            a.copy_state_from(base); a.edit(delete=dele); sc = a.repair("cert", eps=0.5, delta=delta, rho=rho)
            assert float((a.memory - exact).abs().amax()) <= delta + 1e-6
            assert sc.extra["max_bound"] <= delta + 1e-6 and sc.recomputed <= se.recomputed
            skipped += sc.skipped
        if delta == 0.2:
            assert skipped > 0                                   # a loose tolerance does save work
