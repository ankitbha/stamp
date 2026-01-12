import torch
import torch.nn as nn
from typing import Tuple

# =============================================================================
# MPRNN model
# =============================================================================

class MPRNN(nn.Module):
    """
    Per-node LSTMCell + message passing on hidden states after each time step.

    Input:
      x: [B, S, K] (scalar per node per time)
    Output:
      yhat: [B, S, K] (predict next step at each t)
    """

    def __init__(
        self,
        num_sensors: int,
        edge_attr_dim: int,
        hidden_dim: int,
        msg_dim: int,
        edge_mlp_dim: int,
        mp_rounds: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.S = int(num_sensors)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.msg_dim = int(msg_dim)
        self.mp_rounds = int(mp_rounds)

        self.in_proj = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # Message MLP: takes (h_src, h_dst, e_ij) -> msg
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_attr_dim, edge_mlp_dim),
            nn.ReLU(),
            nn.Linear(edge_mlp_dim, msg_dim),
            nn.ReLU(),
        )

        # Update MLP: (h_dst, aggregated_msg) -> delta_h
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _mp_step(
        self,
        h: torch.Tensor,                 # [B,S,H]
        edge_index: torch.Tensor,        # [2,E] (src,dst)
        edge_attr: torch.Tensor,         # [E,F]
    ) -> torch.Tensor:
        B, S, H = h.shape
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]
        E = src.shape[0]

        h_src = h[:, src, :]  # [B,E,H]
        h_dst = h[:, dst, :]  # [B,E,H]
        e = edge_attr[None, :, :].expand(B, E, edge_attr.shape[1])  # [B,E,F]

        m_in = torch.cat([h_src, h_dst, e], dim=-1)  # [B,E,2H+F]
        m = self.msg_mlp(m_in)                       # [B,E,M]

        # Optional geometry weight via 1/r included in edge_attr: assume it is at index 3 (dx,dy,r,1/r,...)
        # If not present, this is harmless if edge_attr has >=4 dims; else skip.
        if edge_attr.shape[1] >= 4:
            w = edge_attr[None, :, 3].expand(B, E)  # [B,E] 1/r
            m = m * w[..., None]

        # Aggregate messages to destination nodes via scatter-add
        agg = torch.zeros((B, S, self.msg_dim), device=h.device, dtype=h.dtype)
        agg.index_add_(1, dst, m)  # sum over edges

        upd_in = torch.cat([h, agg], dim=-1)  # [B,S,H+M]
        dh = self.upd_mlp(upd_in)             # [B,S,H]
        h_new = h + self.dropout(dh)
        return h_new

    def forward_one_step(
        self,
        x_t: torch.Tensor,               # [B,S] scalar observation at time t
        h: torch.Tensor,                 # [B,S,H]
        c: torch.Tensor,                 # [B,S,H]
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LSTMCell expects [B*S, H]
        B, S = x_t.shape
        x_emb = self.in_proj(x_t[..., None])  # [B,S,H]
        x_emb = x_emb.reshape(B * S, self.hidden_dim)
        h0 = h.reshape(B * S, self.hidden_dim)
        c0 = c.reshape(B * S, self.hidden_dim)

        h1, c1 = self.lstm(x_emb, (h0, c0))
        h1 = h1.reshape(B, S, self.hidden_dim)
        c1 = c1.reshape(B, S, self.hidden_dim)

        # Message passing rounds
        for _ in range(self.mp_rounds):
            h1 = self._mp_step(h1, edge_index=edge_index, edge_attr=edge_attr)

        # Decode prediction for next observation
        yhat = self.decoder(h1).squeeze(-1)  # [B,S]
        return yhat, h1, c1

    def forward(
        self,
        x: torch.Tensor,                 # [B,S,K]
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        scheduled_sampling_p: float = 0.0,
    ) -> torch.Tensor:
        B, S, K = x.shape
        device = x.device
        h = torch.zeros((B, S, self.hidden_dim), device=device, dtype=x.dtype)
        c = torch.zeros((B, S, self.hidden_dim), device=device, dtype=x.dtype)

        yhat_seq = torch.zeros((B, S, K), device=device, dtype=x.dtype)

        # Scheduled sampling: for t>=1, decide whether to feed prediction from previous step
        # Mask sampled per batch (broadcast to sensors).
        rng = torch.rand((B, 1), device=device, dtype=x.dtype)

        x_in = x[:, :, 0]
        prev_pred = None
        for t in range(K):
            if t == 0:
                x_in = x[:, :, 0]
            else:
                if prev_pred is None:
                    x_in = x[:, :, t]
                else:
                    use_pred = (rng < float(scheduled_sampling_p)).to(x.dtype)  # [B,1]
                    x_in = use_pred * prev_pred.detach() + (1.0 - use_pred) * x[:, :, t]

            pred, h, c = self.forward_one_step(x_in, h, c, edge_index, edge_attr)
            yhat_seq[:, :, t] = pred
            prev_pred = pred

        return yhat_seq

