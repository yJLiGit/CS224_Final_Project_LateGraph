"""DyRep-style memory: exact repair (cone through own, partner and the partner's neighbours, and through changed
neighbour lists) equals recomputing the corrected stream from scratch."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import synthetic_stream                                  # noqa: E402
from lategraph.data import BatchGrid                                    # noqa: E402
from lategraph.dyrep import DyRep, RetroactiveDyRepMemory               # noqa: E402
from lategraph.memory import EventTable                                 # noqa: E402


def test_dyrep_exact_repair_equals_recomputation():
    torch.set_default_dtype(torch.float64)
    try:
        S = synthetic_stream(seed=11); S.msg = S.msg.double()
        torch.manual_seed(0)
        model = DyRep(S.num_nodes, S.msg_dim, memory_dim=16, time_dim=8, emb_dim=16, n_neighbors=4).eval()
        grid = BatchGrid(S.t, S.test0, 20, 20)
        step = torch.full((S.num_events,), -1, dtype=torch.long); test = torch.arange(S.test0, S.num_events)
        step[test] = grid.step_of_time(S.t[test])
        table = EventTable(S.src, S.dst, S.t, S.msg, step)
        win = test[step[test] < grid.J]
        rng = np.random.default_rng(0)
        missing = torch.zeros(S.num_events, dtype=torch.bool); missing[win[torch.from_numpy(rng.random(win.numel()) < 0.1)]] = True
        base = RetroactiveDyRepMemory(model, table, S.test0)
        content = [win[(step[win] == k) & ~missing[win]] for k in range(grid.J)]
        for c in content:
            base.advance(c)
        present, absent = win[~missing[win]].numpy(), win[missing[win]].numpy()
        work = RetroactiveDyRepMemory.__new__(RetroactiveDyRepMemory); work.__dict__.update(base.__dict__)
        cones = []
        for trial in range(25):
            ins = torch.from_numpy(rng.choice(absent, int(rng.integers(0, 3)), replace=False)).long()
            dele = torch.from_numpy(rng.choice(present, int(rng.integers(1 if ins.numel() == 0 else 0, 3)), replace=False)).long()
            work.copy_state_from(base); work.edit(insert=ins, delete=dele); st = work.repair("exact")
            ref = RetroactiveDyRepMemory(model, table, S.test0)
            for k, c in enumerate(content):
                ref.advance(torch.unique(torch.cat([c[~torch.isin(c, dele)], ins[step[ins] == k]])))
            assert (work.memory - ref.memory).abs().max() <= 1e-10, trial
            assert torch.equal(work.last_update, ref.last_update)
            assert torch.equal(work.store.e_id, ref.store.e_id)
            work.copy_state_from(base); work.edit(insert=ins, delete=dele); sr = work.repair("replay")
            assert (work.memory - ref.memory).abs().max() <= 1e-10
            cones.append(st.recomputed / max(sr.recomputed, 1))
        assert np.mean(cones) < 1.0                     # the cone is smaller than full replay
    finally:
        torch.set_default_dtype(torch.float32)
