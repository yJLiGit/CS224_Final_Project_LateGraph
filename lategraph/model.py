"""The temporal GNN we study: PyG's TGN (Rossi et al., 2020) exactly as in PyG's examples/tgn.py.

  memory      TGNMemory(IdentityMessage, LastAggregator): one GRU-updated vector per node
  embedding   one TransformerConv layer over each node's last-k temporal neighbours (edge attr = [phi(dt) || x_e])
  decoder     MLP link score s(u, v) = w^T relu(W_s z_u + W_d z_v)

Variants (cfg.emb):
  'attn'   standard TGN
  'id'     memory-only TGN: z_u = s_u (the memory is the whole representation)
  'nomem'  memory-free control (TGAT-like, one attention layer): no memory at all, z_u attends over u's neighbour list
           only, so deleting an event from the log already deletes it from the model.
  'proj'   JODIE's time projection: z_u(t) = s_u * (1 + w * dt_u), dt_u = normalised time since u's last update
Memory updater (cfg.updater):
  'gru'    TGN (PyG's TGNMemory)
  'rnn'    a tanh RNN cell, as in JODIE and DyRep (with emb='proj' this is the JODIE memory model of DyGLib)
  'crnn'   a contractive RNN cell (ContractiveRNNCell): every update provably shrinks memory differences
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as Fn
from torch.nn import Linear, Parameter
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


@dataclass
class TGNConfig:
    memory_dim: int = 100
    time_dim: int = 100
    emb_dim: int = 100
    n_neighbors: int = 10
    emb: str = "attn"            # 'attn' | 'id' | 'nomem' | 'proj'
    updater: str = "gru"         # 'gru' | 'rnn' | 'crnn'
    crnn_amin: float = 0.2       # 'crnn': smallest write rate a_min
    crnn_c: float = 0.5          # 'crnn': L1 bound of each row of the memory block of W
    proj_mean: float = 0.0       # 'proj': time gaps are standardised with the training gaps' mean and std (seconds)
    proj_std: float = 1.0


class ContractiveRNNCell(torch.nn.Module):
    """A memory update that is a contraction for every input (used to make the eps-repair certificate a theorem).

        s' = (1 - a) * s + a * tanh(W [s || p || x || phi] + b),   a = a_min + (1 - a_min) * sigmoid(theta)  (per dim)

    with every row of the memory block W[:, :2d] (own memory s and partner memory p) rescaled to L1 norm <= c. For any
    two inputs that differ only in the memories, row i changes by at most (1 - a_i)|ds_i| + a_i (|W_s,i| |ds| + |W_p,i| |dp|)
    (tanh is 1-Lipschitz), so
        ||s'(s, p) - s'(s~, p~)||_inf <= rho * max(||s - s~||_inf, ||p - p~||_inf),
        rho = max_i (1 - a_i + a_i * ||W_i,:2d||_1) <= 1 - a_min (1 - c) < 1.
    The call signature is GRUCell's (input, hidden), so it drops into PyG's TGNMemory as `memory.gru`."""

    def __init__(self, input_size: int, hidden_size: int, a_min: float = 0.2, c: float = 0.5):
        super().__init__()
        assert 0 < a_min <= 1 and 0 <= c < 1
        self.input_size, self.hidden_size, self.a_min, self.c = input_size, hidden_size, a_min, c
        self.lin = Linear(input_size, hidden_size)
        self.theta = Parameter(torch.zeros(hidden_size))

    def reset_parameters(self):
        self.lin.reset_parameters(); torch.nn.init.zeros_(self.theta)

    def weight(self) -> torch.Tensor:
        W = self.lin.weight
        m = W[:, :2 * self.hidden_size]
        scale = (self.c / m.abs().sum(1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)
        return torch.cat([m * scale, W[:, 2 * self.hidden_size:]], 1)

    def alpha(self) -> torch.Tensor:
        return self.a_min + (1 - self.a_min) * torch.sigmoid(self.theta)

    def forward(self, x, h):
        a = self.alpha().to(h.dtype)
        return (1 - a) * h + a * torch.tanh(Fn.linear(x, self.weight().to(x.dtype), self.lin.bias.to(x.dtype)))

    @torch.no_grad()
    def rho(self) -> float:
        """The contraction constant of this cell (computed from its weights; <= 1 - a_min (1 - c))."""
        a = self.alpha(); r = self.weight()[:, :2 * self.hidden_size].abs().sum(1)
        return float((1 - a + a * r).max())


class TimeProjection(torch.nn.Module):
    """JODIE's embedding projection: z_u(t) = s_u * (1 + w * dt_u), dt_u = (t - t_u - mean) / std."""

    def __init__(self, dim: int, mean: float, std: float):
        super().__init__()
        self.w = Linear(1, dim)
        torch.nn.init.normal_(self.w.weight, 0, 0.1); torch.nn.init.zeros_(self.w.bias)
        self.mean, self.std = mean, max(std, 1e-6)

    def forward(self, mem, last_update, now: int):
        dt = ((now - last_update).clamp(min=0).to(mem.dtype) - self.mean) / self.std
        return mem * (1 + self.w(dt.view(-1, 1)))


class GraphAttentionEmbedding(torch.nn.Module):
    """z_u = TransformerConv over u's last-k neighbours; edge attributes are [phi(t_last(u) - t_e) || x_e]."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg, return_attention=False):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        if return_attention:                      # (z, (edge_index, alpha [edges, heads]))
            return self.conv(x, edge_index, edge_attr, return_attention_weights=True)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        return self.lin_final((self.lin_src(z_src) + self.lin_dst(z_dst)).relu())


class TGN(torch.nn.Module):
    """Container for the three TGN modules (kept separate, as in PyG's example, so TGNMemory stays a plain module)."""

    def __init__(self, num_nodes: int, msg_dim: int, cfg: TGNConfig = TGNConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.memory_dim
        self.memory = TGNMemory(num_nodes, msg_dim, d, cfg.time_dim,
                                message_module=IdentityMessage(msg_dim, d, cfg.time_dim),
                                aggregator_module=LastAggregator())
        n_in = self.memory.msg_s_module.out_channels
        if cfg.updater == "rnn":
            self.memory.gru = torch.nn.RNNCell(n_in, d)
        elif cfg.updater == "crnn":
            self.memory.gru = ContractiveRNNCell(n_in, d, cfg.crnn_amin, cfg.crnn_c)
        self.gnn = GraphAttentionEmbedding(d, cfg.emb_dim, msg_dim, self.memory.time_enc)
        if cfg.emb == "proj":
            self.proj = TimeProjection(d, cfg.proj_mean, cfg.proj_std)
        self.link_pred = LinkPredictor(cfg.emb_dim if cfg.emb not in ("id", "proj") else d)

    def embed(self, mem, last_update, edge_index, t_e, msg_e, now=None):
        """Node embeddings from memory rows `mem` (aligned with the local subgraph) and the neighbour edges."""
        if self.cfg.emb == "id":
            return mem
        if self.cfg.emb == "proj":
            return self.proj(mem, last_update, int(now) if now is not None else int(last_update.max()))
        if self.cfg.emb == "nomem":
            mem = torch.zeros_like(mem)
            last_update = torch.full_like(last_update, int(now) if now is not None else int(t_e.max() if t_e.numel() else 0))
        return self.gnn(mem, last_update, edge_index, t_e, msg_e)

    def score(self, z_src, z_dst):
        return self.link_pred(z_src, z_dst).squeeze(-1)


def save_checkpoint(path, model: TGN, meta: dict):
    torch.save(dict(state=model.state_dict(), cfg=asdict(model.cfg), meta=meta,
                    num_nodes=model.memory.num_nodes, msg_dim=model.memory.raw_msg_dim), path)


def load_checkpoint(path, map_location="cpu") -> TGN:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = TGN(ck["num_nodes"], ck["msg_dim"], TGNConfig(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.meta = ck.get("meta", {})
    model.eval()
    return model
