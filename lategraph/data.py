"""Event streams: one timestamped interaction log with a chronological train/val/test split.

An event e is (src_e, dst_e, t_e, x_e): who interacted with what, when it happened (occurrence time, seconds) and an
edge feature vector. Events are stored sorted by occurrence time; the event id is the position in that order.
A second clock, the arrival time a_e >= t_e, is attached later (simulated for JODIE, real for SEC Form 4).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import paths


@dataclass
class EventStream:
    name: str
    src: torch.Tensor            # [E] long
    dst: torch.Tensor            # [E] long
    t: torch.Tensor              # [E] long, occurrence time in seconds, non-decreasing
    msg: torch.Tensor            # [E, F] float edge features
    num_nodes: int
    dst_min: int                 # destination ("item") ids are the contiguous range [dst_min, dst_max]
    dst_max: int
    val0: int                    # first validation event
    test0: int                   # first test event
    arrival: Optional[torch.Tensor] = None   # [E] long arrival time (None = no second clock yet)
    extra: dict = field(default_factory=dict)

    @property
    def num_events(self) -> int:
        return int(self.src.numel())

    @property
    def msg_dim(self) -> int:
        return int(self.msg.size(-1))

    def summary(self) -> dict:
        t = self.t.double()
        return dict(name=self.name, events=self.num_events, nodes=self.num_nodes,
                    src_nodes=int(torch.unique(self.src).numel()), dst_nodes=self.dst_max - self.dst_min + 1,
                    msg_dim=self.msg_dim, train=self.val0, val=self.test0 - self.val0, test=self.num_events - self.test0,
                    span_days=float((t.max() - t.min()) / 86400))

    def subset(self, keep: torch.Tensor, name: Optional[str] = None) -> "EventStream":
        """The same stream without some events (boolean mask `keep`); node ids stay, the split points move with it."""
        idx = keep.nonzero().view(-1)
        return EventStream(name or self.name, self.src[idx], self.dst[idx], self.t[idx], self.msg[idx], self.num_nodes,
                           self.dst_min, self.dst_max, int(keep[:self.val0].sum()), int(keep[:self.test0].sum()),
                           None if self.arrival is None else self.arrival[idx], dict(self.extra))

    def to_temporal_data(self):
        """The same stream as a PyG TemporalData object (used for training with TemporalDataLoader)."""
        from torch_geometric.data import TemporalData
        return TemporalData(src=self.src, dst=self.dst, t=self.t, msg=self.msg)


def chronological_split(t: torch.Tensor, val_ratio=0.15, test_ratio=0.15):
    """Exactly PyG's TemporalData.train_val_test_split: split by the 70% and 85% quantiles of the timestamps."""
    val_time, test_time = np.quantile(t.cpu().numpy(), [1. - val_ratio - test_ratio, 1. - test_ratio])
    return int((t <= val_time).sum()), int((t <= test_time).sum())


def load_jodie(name: str, root=None) -> EventStream:
    """Wikipedia, Reddit or MOOC from PyG's JODIEDataset (Kumar et al., KDD 2019). Users are src, items are dst."""
    from torch_geometric.datasets import JODIEDataset
    data = JODIEDataset(str(root or paths.JODIE_ROOT), name=name)[0]
    t = data.t.long()
    assert bool((t[1:] >= t[:-1]).all()), "JODIE events are expected to be sorted by time"
    val0, test0 = chronological_split(t)
    return EventStream(name=name, src=data.src.long(), dst=data.dst.long(), t=t, msg=data.msg.float(),
                       num_nodes=int(data.num_nodes), dst_min=int(data.dst.min()), dst_max=int(data.dst.max()),
                       val0=val0, test0=test0)


TGB_SPLITS = {"tgbl-review": ("2016-06-03", "2017-03-07")}   # TGB's official validation / test start dates


def load_tgb(name: str, root=None) -> EventStream:
    """A TGB link-prediction dataset (Huang et al., NeurIPS 2023) from the Hugging Face mirror's parquet event table
    (`stanford-star/tgb`, <root>/<name>/db/events.parquet), with TGB's chronological split and the edge weight as the
    1-dimensional edge feature. Timestamps have a resolution of one day. In this mirror sources and destinations share
    one id space and the graph is not bipartite (2.3M of tgbl-review's 4.9M edges close an odd cycle), so every node is
    a possible destination: dst_min = 0, dst_max = num_nodes - 1."""
    import pyarrow.parquet as pq
    root = Path(root or paths.DATA / "tgb")
    cache = root / name / "processed_stream.pt"
    if cache.exists():
        d = torch.load(cache, weights_only=False)
    else:
        tab = pq.read_table(root / name / "db" / "events.parquet").to_pandas()
        src, dst = tab.src_id.to_numpy(), tab.dst_id.to_numpy()
        n = int(max(src.max(), dst.max())) + 1
        t = (tab.event_ts.astype("int64").to_numpy() // 10 ** 9).astype(np.int64)
        assert (np.diff(t) >= 0).all()
        val_ts, test_ts = (int(np.datetime64(x, "s").astype(np.int64)) for x in TGB_SPLITS[name])
        d = dict(src=torch.from_numpy(src).long(), dst=torch.from_numpy(dst).long(), t=torch.from_numpy(t),
                 msg=torch.from_numpy(tab.weight.to_numpy(np.float32)).view(-1, 1), num_nodes=n, dst_min=0, dst_max=n - 1,
                 val0=int(np.searchsorted(t, val_ts)), test0=int(np.searchsorted(t, test_ts)))
        torch.save(d, cache)
    return EventStream(name=name, **d)


def load(name: str) -> EventStream:
    if name in ("wikipedia", "reddit", "mooc", "lastfm"):
        return load_jodie(name)
    if name.startswith("tgbl-"):
        return load_tgb(name)
    if name.startswith("sec"):
        from .sec import load_sec_stream
        return load_sec_stream()
    raise ValueError(name)


class BatchGrid:
    """The query/processing steps of the streaming evaluation.

    Step j covers occurrence times [T_j, T_{j+1}). Following PyG's TGN evaluation we place a step boundary every `bq`
    test events (T_j = time of test event j*bq), but an event's step is decided by its *time*, not its position:
    step(e) = max{j : T_j <= t_e}. So a late, deleted or amended event always belongs to the same step, which is what
    makes repair well defined ("same batching" in the definition of the target memory S*).
    """

    def __init__(self, t: torch.Tensor, start: int, bq: int, max_steps: Optional[int] = None):
        n = t.numel() - start
        J = (n + bq - 1) // bq
        if max_steps is not None:
            J = min(J, max_steps)
        bounds = [int(t[start + j * bq]) for j in range(J)]
        last = min(start + J * bq, t.numel()) - 1          # last test event covered by the J steps
        end = int(t[last]) + 1
        self.T = torch.tensor(bounds + [max(end, bounds[-1] + 1)], dtype=torch.long)
        self.J, self.bq, self.start = J, bq, start

    @classmethod
    def from_boundaries(cls, T):
        """A grid with explicit step boundaries T_0 < T_1 < ... < T_J (e.g. one step per calendar day)."""
        g = cls.__new__(cls)
        g.T = torch.as_tensor(T, dtype=torch.long)
        g.J, g.bq, g.start = int(g.T.numel() - 1), None, None
        return g

    def step_of_time(self, tt: torch.Tensor) -> torch.Tensor:
        """Step index of occurrence times (-1 = before the first step, J = after the last step)."""
        return torch.searchsorted(self.T, tt.contiguous(), right=True) - 1

    def receipt_step(self, arrival: torch.Tensor, own_step: torch.Tensor) -> torch.Tensor:
        """First step whose predictions may use the event: T_j > a_e, and never before the step after its own."""
        r = torch.searchsorted(self.T, arrival.contiguous(), right=True)
        return torch.maximum(r, own_step + 1)
