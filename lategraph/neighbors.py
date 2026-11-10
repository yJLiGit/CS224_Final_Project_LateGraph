"""A last-k temporal neighbour store ordered by occurrence time.

PyG's LastNeighborLoader keeps, for every node, the k most recently *inserted* interactions. When events arrive late
that is the wrong notion of "most recent": a late event would push out neighbours that actually happened after it. This
store keeps the k neighbours with the largest occurrence key (t_e, e), so its content depends only on *which* events
it holds, not on the order they arrived in. It also supports deletion: removed events are dropped and the row is refilled
from the node's older events.

The interface mirrors LastNeighborLoader: store(n_id) -> (n_id, edge_index, e_id) with edge_index = [neighbour, centre]
in local indices.
"""
import numpy as np
import torch


def event_keys(t: torch.Tensor) -> torch.Tensor:
    """Strict total order on events: occurrence time first, then event id (ids are assigned in time order)."""
    ids = torch.arange(t.numel(), dtype=torch.long)
    return t.long() * (1 << 22) + ids


class TimeOrderedNeighborStore:
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, num_nodes: int, size: int = 10):
        self.src, self.dst, self.N, self.K = src, dst, num_nodes, size
        self.key = event_keys(t)
        self.E = src.numel()
        self.e_id = torch.full((num_nodes, size), -1, dtype=torch.long)
        self.nbr = torch.full((num_nodes, size), -1, dtype=torch.long)
        self.ekey = torch.full((num_nodes, size), -1, dtype=torch.long)
        self.alive = torch.zeros(self.E, dtype=torch.bool)
        self._csr = None
        self._assoc = torch.empty(num_nodes, dtype=torch.long)

    # ------------------------------------------------------------------------------------------------ bookkeeping
    def clone(self):
        c = TimeOrderedNeighborStore.__new__(TimeOrderedNeighborStore)
        c.__dict__.update(self.__dict__)
        c.e_id, c.nbr, c.ekey, c.alive = self.e_id.clone(), self.nbr.clone(), self.ekey.clone(), self.alive.clone()
        c._assoc = torch.empty(self.N, dtype=torch.long)
        return c

    def _node_csr(self):
        """For every node, all incident event ids sorted by key (built once, used to refill rows after deletions)."""
        if self._csr is None:
            node = torch.cat([self.src, self.dst]).numpy()
            eid = np.concatenate([np.arange(self.E), np.arange(self.E)])
            order = np.lexsort((self.key.numpy()[eid], node))
            node, eid = node[order], eid[order]
            ptr = np.zeros(self.N + 1, dtype=np.int64)
            np.add.at(ptr, node + 1, 1)
            self._csr = (np.cumsum(ptr), eid)
        return self._csr

    # ------------------------------------------------------------------------------------------------ updates
    def insert(self, ev: torch.Tensor):
        """Add events `ev`; each endpoint keeps its k neighbours with the largest occurrence key."""
        ev = ev[~self.alive[ev]] if ev.numel() else ev
        if ev.numel() == 0:
            return
        self.alive[ev] = True
        node = torch.cat([self.src[ev], self.dst[ev]]); nbr = torch.cat([self.dst[ev], self.src[ev]])
        eid = torch.cat([ev, ev]); key = self.key[eid]
        uniq, inv = torch.unique(node, return_inverse=True)
        U, K = uniq.numel(), self.K
        order = torch.from_numpy(np.lexsort((-key.numpy(), inv.numpy())))      # by node, newest first
        inv_o = inv[order]
        cnt = torch.bincount(inv_o, minlength=U)
        start = torch.zeros(U, dtype=torch.long); start[1:] = torch.cumsum(cnt, 0)[:-1]
        rank = torch.arange(order.numel()) - start[inv_o]
        keep = rank < K
        o, r, g = order[keep], rank[keep], inv_o[keep]
        nk = torch.full((U, K), -1, dtype=torch.long); ne = torch.full((U, K), -1, dtype=torch.long)
        nn_ = torch.full((U, K), -1, dtype=torch.long)
        nk[g, r] = key[o]; ne[g, r] = eid[o]; nn_[g, r] = nbr[o]
        ck = torch.cat([self.ekey[uniq], nk], 1); ce = torch.cat([self.e_id[uniq], ne], 1); cn = torch.cat([self.nbr[uniq], nn_], 1)
        top = ck.topk(K, dim=1).indices
        self.ekey[uniq] = ck.gather(1, top); self.e_id[uniq] = ce.gather(1, top); self.nbr[uniq] = cn.gather(1, top)

    def remove(self, ev: torch.Tensor):
        """Delete events `ev` and refill their endpoints' rows from older live events."""
        ev = ev[self.alive[ev]] if ev.numel() else ev
        if ev.numel() == 0:
            return
        self.alive[ev] = False
        ptr, eids = self._node_csr()
        for n in torch.unique(torch.cat([self.src[ev], self.dst[ev]])).tolist():
            cand = eids[ptr[n]:ptr[n + 1]]
            cand = cand[self.alive.numpy()[cand]][-self.K:][::-1]            # k largest keys, newest first
            row_e = torch.full((self.K,), -1, dtype=torch.long); row_k = row_e.clone(); row_n = row_e.clone()
            if len(cand):
                c = torch.from_numpy(cand.copy())
                row_e[:len(cand)] = c; row_k[:len(cand)] = self.key[c]
                row_n[:len(cand)] = torch.where(self.src[c] == n, self.dst[c], self.src[c])
            self.e_id[n], self.ekey[n], self.nbr[n] = row_e, row_k, row_n

    def reset(self):
        self.e_id.fill_(-1); self.nbr.fill_(-1); self.ekey.fill_(-1); self.alive.zero_()

    # ------------------------------------------------------------------------------------------------ queries
    def __call__(self, n_id: torch.Tensor):
        e_id = self.e_id[n_id]; nbr = self.nbr[n_id]
        mask = e_id >= 0
        centre = n_id.view(-1, 1).expand_as(e_id)[mask]
        nbr, e_id = nbr[mask], e_id[mask]
        all_id = torch.cat([n_id, nbr]).unique()
        self._assoc[all_id] = torch.arange(all_id.numel())
        return all_id, torch.stack([self._assoc[nbr], self._assoc[centre]]), e_id

    def rebuild_from(self, alive: torch.Tensor):
        """Reset and insert every event in the boolean mask `alive` (used to build stores without some events)."""
        self.reset()
        idx = alive.nonzero().view(-1)
        for b in range(0, idx.numel(), 50000):
            self.insert(idx[b:b + 50000])
