"""Delay models: they give every test event an arrival time a_e = t_e + d_e (the second clock).

The JODIE datasets record only when an interaction happened, so we simulate how late it reaches the system. All models
are deterministic given the seed. Delays are in seconds; 0 means "on time".

  rand       a fraction p of events is late; delay ~ LogNormal(median `med` hours, sigma)
  burst      `outages` outages of `odur` hours; during each, the events of a random `ofrac` of users are held back
             until the outage ends (plus a short exponential tail)
  active     the most active `hubq` of users (by pre-test activity) send half of their events late (lognormal);
             delays concentrated on busy nodes
  offline    a fraction p of users is offline and uploads its events every `upload` hours (phones syncing)
  itemshard  a fraction p of items lives on a slow shard; half of their events are late (lognormal)
"""
import math

import numpy as np
import torch

H = 3600.0


def make_delays(stream, kind: str, seed: int = 0, p: float = 0.2, med: float = 2.0, sigma: float = 1.0,
                outages: int = 4, odur: float = 3.0, ofrac: float = 0.5, hubq: float = 0.05, upload: float = 6.0,
                lo: int = None, hi: int = None) -> np.ndarray:
    """Delay (seconds) for events lo..hi-1 (default: the test period)."""
    lo = stream.test0 if lo is None else lo
    hi = stream.num_events if hi is None else hi
    rng = np.random.default_rng(seed)
    n = hi - lo
    tt = stream.t[lo:hi].numpy().astype(np.float64)
    sv = stream.src[lo:hi].numpy(); dv = stream.dst[lo:hi].numpy()
    delay = np.zeros(n)
    users = np.unique(stream.src[:stream.test0].numpy())
    if kind == "none":
        return delay
    if kind == "rand":
        late = rng.random(n) < p
        delay[late] = rng.lognormal(math.log(med * H), sigma, late.sum())
    elif kind == "active":
        act = np.bincount(stream.src[:stream.test0].numpy(), minlength=stream.num_nodes)
        k = max(1, int(hubq * len(users)))
        hubs = np.zeros(stream.num_nodes, bool); hubs[users[np.argsort(-act[users])][:k]] = True
        late = hubs[sv] & (rng.random(n) < 0.5)
        delay[late] = rng.lognormal(math.log(med * H), sigma, late.sum())
    elif kind == "burst":
        starts = np.sort(rng.uniform(tt.min(), tt.max() - odur * H, outages))
        for s0 in starts:
            aff = np.zeros(stream.num_nodes, bool)
            aff[rng.choice(users, int(ofrac * len(users)), replace=False)] = True
            inwin = (tt >= s0) & (tt < s0 + odur * H) & aff[sv]
            delay[inwin] = np.maximum(delay[inwin], (s0 + odur * H - tt[inwin]) + rng.exponential(60.0, inwin.sum()))
    elif kind == "offline":
        off = np.zeros(stream.num_nodes, bool)
        off[rng.choice(users, int(p * len(users)), replace=False)] = True
        late = off[sv]
        phase = rng.uniform(0, upload * H, stream.num_nodes)
        up = np.ceil((tt - phase[sv]) / (upload * H)) * upload * H + phase[sv]
        delay[late] = (up - tt)[late] + 1.0
    elif kind == "itemshard":
        items = np.arange(stream.dst_min, stream.dst_max + 1)
        slow = np.zeros(stream.num_nodes, bool)
        slow[rng.choice(items, max(1, int(p * len(items))), replace=False)] = True
        late = slow[dv] & (rng.random(n) < 0.5)
        delay[late] = rng.lognormal(math.log(med * H), sigma, late.sum())
    else:
        raise ValueError(kind)
    return delay


def arrival_times(stream, delay: np.ndarray, lo: int = None) -> torch.Tensor:
    """Full-length arrival tensor: history events arrive on time, events lo.. get t + delay (rounded up)."""
    lo = stream.test0 if lo is None else lo
    a = stream.t.clone()
    a[lo:lo + len(delay)] = stream.t[lo:lo + len(delay)] + torch.from_numpy(np.ceil(delay)).long()
    return a
