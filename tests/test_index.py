"""The step-indexed repair (default) must give exactly what walking every step gives: the same memory, the same last
update times and the same journal, bit for bit, for exact / eps / khop repairs, chains of repairs and a short journal."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import build, synthetic_stream                          # noqa: E402
from lategraph.memory import RetroactiveTGNMemory, khop_mask_for       # noqa: E402


def same_state(a, b):
    assert torch.equal(a.memory, b.memory) and torch.equal(a.last_update, b.last_update)
    assert set(a.journal) == set(b.journal)
    for k in a.journal:
        ra, rb = a.journal[k], b.journal[k]
        for f in ("nodes", "other", "eid", "pre_mem", "pre_lu", "post_mem", "post_lu"):
            assert torch.equal(getattr(ra, f), getattr(rb, f)), (k, f)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("horizon", [None, 12])
def test_indexed_repair_is_bitwise_equal_to_the_full_walk(ties, horizon):
    S = synthetic_stream(ties=ties, seed=3)
    model, grid, table, step, test = build(S, steps=30, std=0.2)
    rng = np.random.default_rng(1)
    win = test[step[test] < grid.J]
    missing = torch.zeros(S.num_events, dtype=torch.bool)
    missing[win[torch.from_numpy(rng.random(win.numel()) < 0.15)]] = True
    base = RetroactiveTGNMemory.from_trained(model.memory, table, horizon=horizon)
    base.warmup(0, S.test0)
    for k in range(grid.J):
        base.advance(win[(step[win] == k) & ~missing[win]])
    oldest = base.oldest_repairable_step()
    ok = step[win] >= oldest
    present, absent = win[ok & ~missing[win]].numpy(), win[ok & missing[win]].numpy()
    a = RetroactiveTGNMemory.from_trained(model.memory, table, horizon=horizon)
    b = RetroactiveTGNMemory.from_trained(model.memory, table, horizon=horizon)
    for trial in range(40):
        ins = torch.from_numpy(rng.choice(absent, int(rng.integers(0, 4)), replace=False)).long()
        dele = torch.from_numpy(rng.choice(present, int(rng.integers(1 if ins.numel() == 0 else 0, 4)), replace=False)).long()
        mode = ["exact", "eps", "khop"][trial % 3]
        mask = None
        if mode == "khop":
            seeds = torch.cat([S.src[torch.cat([ins, dele])], S.dst[torch.cat([ins, dele])]]).unique()
            mask = khop_mask_for(table, seeds, win, 1, S.num_nodes)
        a.copy_state_from(base); b.copy_state_from(base); b.use_index = False
        for m in (a, b):
            m.edit(insert=ins, delete=dele)
        sa = a.repair(mode, eps=0.05, khop_mask=mask); sb = b.repair(mode, eps=0.05, khop_mask=mask)
        same_state(a, b)
        assert (sa.recomputed, sa.skipped, sa.window_updates, sa.changed_nodes) == (sb.recomputed, sb.skipped, sb.window_updates, sb.changed_nodes)
        assert sa.extra["visited_steps"] <= sa.steps
        # a second edit on top of the repaired state (checks the journal written by the indexed repair)
        if present.size > 8:
            d2 = torch.from_numpy(rng.choice(present, 2, replace=False)).long()
            d2 = d2[~torch.isin(d2, dele)]
            for m in (a, b):
                m.edit(delete=d2); m.repair(mode, eps=0.05, khop_mask=mask)
            same_state(a, b)
