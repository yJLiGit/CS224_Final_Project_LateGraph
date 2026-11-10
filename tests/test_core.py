"""Unit tests for the memory kernel, exact repair and the neighbour store.

Run:  pytest -q tests/
They use a small synthetic stream (no downloads) so they finish in a few seconds.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lategraph.data import BatchGrid, EventStream                     # noqa: E402
from lategraph.memory import EventTable, RetroactiveTGNMemory         # noqa: E402
from lategraph.model import TGN                                       # noqa: E402
from lategraph.neighbors import TimeOrderedNeighborStore              # noqa: E402


def synthetic_stream(E=3000, users=60, items=30, seed=0, ties=False):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, users, (E,), generator=g)
    dst = torch.randint(users, users + items, (E,), generator=g)
    if ties:
        t = torch.sort(torch.randint(0, E // 3, (E,), generator=g)).values       # many equal timestamps
    else:
        t = torch.arange(E) * 7 + torch.randint(0, 5, (E,), generator=g)          # strictly increasing
    msg = torch.randn(E, 8, generator=g)
    return EventStream("synthetic", src, dst, t.long(), msg, users + items, users, users + items - 1, int(E * .7), int(E * .85))


def build(S, steps=20, bq=20, seed=0, std=0.3, double=False):
    torch.manual_seed(seed)
    model = TGN(S.num_nodes, S.msg_dim)
    for p in model.parameters():                       # larger weights, so updates are far from trivial
        torch.nn.init.normal_(p, std=std)
    if double:
        model = model.double(); S.msg = S.msg.double()
        model.memory.reset_state()          # PyG's message store keeps float32 placeholders otherwise
    model.eval()
    grid = BatchGrid(S.t, S.test0, bq, steps)
    step = torch.full((S.num_events,), -1, dtype=torch.long)
    test = torch.arange(S.test0, S.num_events)
    step[test] = grid.step_of_time(S.t[test])
    table = EventTable(S.src, S.dst, S.t, S.msg, step)
    return model, grid, table, step, test


def test_kernel_matches_pyg_tgnmemory():
    """On a stream without timestamp ties our update equals PyG's TGNMemory (eval mode) step by step. (float64, so
    that rounding differences between the two summation orders do not grow through the random, chaotic weights.)"""
    S = synthetic_stream()
    model, grid, table, step, test = build(S, double=True)
    ours = RetroactiveTGNMemory.from_trained(model.memory, table)
    ref = model.memory; ref.reset_state()
    for b in range(0, 2000, 40):
        ev = torch.arange(b, b + 40)
        ref.update_state(S.src[ev], S.dst[ev], S.t[ev], S.msg[ev])
        ours.apply(ev)
        assert torch.allclose(ref.memory, ours.memory, atol=1e-10)
        assert torch.equal(ref.last_update, ours.last_update)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_repair_equals_sequential_recomputation(ties, dtype):
    S = synthetic_stream(ties=ties)
    # float32 with realistic weight sizes (rounding differences of reordered sums stay at 1e-7);
    # float64 with large weights (any logic error would show up far above the 1e-12 tolerance)
    model, grid, table, step, test = build(S, std=0.05 if dtype == torch.float32 else 0.3, double=dtype == torch.float64)
    rng = np.random.default_rng(0)
    win = test[step[test] < grid.J]
    missing = torch.zeros(S.num_events, dtype=torch.bool)
    missing[win[torch.from_numpy(rng.random(win.numel()) < 0.1)]] = True
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    base.warmup(0, S.test0)
    ckpt, content = [], []
    for k in range(grid.J):
        ev = win[(step[win] == k) & ~missing[win]]
        ckpt.append((base.memory.clone(), base.last_update.clone())); content.append(ev)
        base.advance(ev)
    work = RetroactiveTGNMemory.from_trained(model.memory, table)
    ref = RetroactiveTGNMemory.from_trained(model.memory, table)
    present, absent = win[~missing[win]].numpy(), win[missing[win]].numpy()
    for trial in range(60):
        ins = torch.from_numpy(rng.choice(absent, int(rng.integers(0, 3)), replace=False)).long()
        dele = torch.from_numpy(rng.choice(present, int(rng.integers(1 if ins.numel() == 0 else 0, 3)), replace=False)).long()
        k0 = int(torch.cat([step[ins], step[dele]]).min())
        for mode in ("exact", "replay"):
            work.copy_state_from(base); work.edit(insert=ins, delete=dele); work.repair(mode)
            ref.memory.copy_(ckpt[k0][0]); ref.last_update.copy_(ckpt[k0][1])
            for k in range(k0, grid.J):
                c = content[k]
                c = torch.unique(torch.cat([c[~torch.isin(c, dele)], ins[step[ins] == k]]))
                ref.apply(c)
            tol = 1e-5 if dtype == torch.float32 else 1e-12
            assert (work.memory - ref.memory).abs().max() <= tol, (trial, mode)
            assert torch.equal(work.last_update, ref.last_update)


def test_eps_zero_equals_exact_and_eps_saves_work():
    S = synthetic_stream()
    model, grid, table, step, test = build(S)
    base = RetroactiveTGNMemory.from_trained(model.memory, table)
    base.warmup(0, S.test0)
    win = test[step[test] < grid.J]
    for k in range(grid.J):
        base.advance(win[step[win] == k])
    dele = win[:3]
    a = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(base)
    a.edit(delete=dele); sa = a.repair("exact")
    b = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(base)
    b.edit(delete=dele); b.repair("eps", eps=0.0)
    assert (a.memory - b.memory).abs().max() <= 1e-6
    c = RetroactiveTGNMemory.from_trained(model.memory, table).copy_state_from(base)
    c.edit(delete=dele); sc = c.repair("eps", eps=0.5)
    assert sc.recomputed <= sa.recomputed


def test_neighbor_store_is_order_invariant_and_supports_deletion():
    S = synthetic_stream(E=800)
    ev = torch.arange(S.num_events)
    a = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 5); a.insert(ev)
    b = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 5)
    perm = ev[torch.randperm(ev.numel(), generator=torch.Generator().manual_seed(1))]
    for chunk in perm.split(37):                      # same events, scrambled arrival order
        b.insert(chunk)
    assert torch.equal(a.e_id, b.e_id)
    gone = ev[::7]
    a.remove(gone)
    keep = torch.ones(S.num_events, dtype=torch.bool); keep[gone] = False
    c = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 5); c.rebuild_from(keep)
    assert torch.equal(a.e_id, c.e_id)
