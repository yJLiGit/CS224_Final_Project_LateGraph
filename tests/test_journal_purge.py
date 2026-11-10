"""After exact repair of a deletion, nothing of the deleted events is left in the journal either: memory, last-update times
and every journal record (nodes, partners, events, memories before and after each update) are those of a memory that
never saw them. Checked in float64 (bit for bit) on the synthetic stream."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import build, synthetic_stream                          # noqa: E402
from test_index import same_state                                      # noqa: E402
from lategraph.memory import RetroactiveTGNMemory                      # noqa: E402


def test_exact_repair_purges_the_journal():
    S = synthetic_stream(ties=False, seed=5)
    model, grid, table, step, test = build(S, steps=30, std=0.2, double=True)
    rng = np.random.default_rng(2)
    win = test[step[test] < grid.J]
    for trial in range(10):
        dele = torch.from_numpy(rng.choice(win.numpy(), int(rng.integers(1, 6)), replace=False)).long()
        keep = ~torch.isin(win, dele)
        repaired = RetroactiveTGNMemory.from_trained(model.memory, table)
        repaired.reset_state(); repaired.warmup(0, S.test0)
        for k in range(grid.J):
            repaired.advance(win[step[win] == k])
        repaired.edit(delete=dele); repaired.repair("exact")
        never = RetroactiveTGNMemory.from_trained(model.memory, table)
        never.reset_state(); never.warmup(0, S.test0)
        for k in range(grid.J):
            never.advance(win[(step[win] == k) & keep])
        same_state(repaired, never)
