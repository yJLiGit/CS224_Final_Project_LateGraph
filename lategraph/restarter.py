"""A TIGER-style restarter (Zhang et al., WWW 2023): a small network that regenerates a node's memory from its last k
interactions, trained by distillation from the TGN memory.

LateGraph uses it as the "learned correction" baseline: instead of recomputing the dependency cone, an edited node gets
a fresh memory from its corrected neighbour list. It is cheap (one forward pass per node) but approximate, and it only
touches the nodes we restart (TIGER restarts nodes; it does not track what else changed).

    s_hat_n = MLP( sum_j alpha_j * W [ s_{partner_j} || x_{e_j} || phi(now - t_{e_j}) ] ),   alpha = softmax_j(w^T h_j)
"""
import torch
from torch.nn import Linear
from torch_geometric.nn.models.tgn import TimeEncoder

from .memory import RetroactiveTGNMemory
from .neighbors import TimeOrderedNeighborStore


class Restarter(torch.nn.Module):
    def __init__(self, memory_dim: int, msg_dim: int, time_dim: int = 100, hidden: int = 200):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        d_in = memory_dim + msg_dim + time_dim
        self.score = Linear(d_in, 1)
        self.proj = Linear(d_in, hidden)
        self.out = torch.nn.Sequential(torch.nn.ReLU(), Linear(hidden, hidden), torch.nn.ReLU(), Linear(hidden, memory_dim))

    def forward(self, nodes, mem: torch.Tensor, store: TimeOrderedNeighborStore, events, now: int):
        """Regenerated memory for `nodes` from their rows in the neighbour store (empty rows -> zero memory)."""
        e_id, nbr = store.e_id[nodes], store.nbr[nodes]                        # [B, K]
        mask = e_id >= 0
        e_safe, n_safe = e_id.clamp(min=0), nbr.clamp(min=0)
        dt = (now - events.t[e_safe]).clamp(min=0).to(mem.dtype)
        h = torch.cat([mem[n_safe], events.msg[e_safe].to(mem.dtype), self.time_enc(dt.view(-1)).view(*dt.shape, -1)], -1)
        a = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(a, dim=1).nan_to_num(0.0) * mask
        pooled = (alpha.unsqueeze(-1) * self.proj(h)).sum(1)
        out = self.out(pooled)
        return torch.where(mask.any(1, keepdim=True), out, torch.zeros_like(out))


def train_restarter(model, S, table, epochs=3, batch_size=200, lr=1e-3, seed=0, log=print):
    """Distil the restarter from the trained TGN on the history (train + val): after every batch, the restarter must
    reproduce the new memory of the updated nodes from their last-k neighbours (MSE)."""
    torch.manual_seed(seed)
    R = Restarter(model.memory.memory_dim, S.msg_dim)
    opt = torch.optim.Adam(R.parameters(), lr=lr)
    for ep in range(epochs):
        mem = RetroactiveTGNMemory.from_trained(model.memory, table)
        store = TimeOrderedNeighborStore(S.src, S.dst, S.t, S.num_nodes, 10)
        tot, n = 0.0, 0
        for b in range(0, S.test0, batch_size):
            ev = torch.arange(b, min(b + batch_size, S.test0))
            nodes, _, _ = mem.choose(ev)
            mem.apply(ev); store.insert(ev)
            if b < batch_size * 20:          # skip the very beginning (empty memories)
                continue
            pred = R(nodes, mem.memory, store, table, int(S.t[ev].max()))
            loss = torch.nn.functional.mse_loss(pred, mem.memory[nodes])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * nodes.numel(); n += nodes.numel()
        log(f"restarter epoch {ep + 1}: distillation MSE {tot / max(n, 1):.4f} (memory variance "
            f"{float(mem.memory[mem.memory.abs().sum(1) > 0].var()):.4f})")
    R.eval()
    return R
