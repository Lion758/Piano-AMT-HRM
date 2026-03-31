import torch
import torch.nn as nn
import torch.nn.functional as F


class HRMBlock(nn.Module):
    """
    A lightweight HRM-style reasoning block for continuous embeddings.
    This is NOT a token generator. It preserves sequence length and dimension.
    """

    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Feed-forward
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class HRMAdapter(nn.Module):
    """
    HRM-style hierarchical refiner for pooled encoder memory.

    Input:  [B, T_pool, d_model]
    Output: [B, T_pool, d_model]

    Uses recurrent H/L cycles with gated residual output.
    """

    def __init__(
        self,
        d_model,
        hidden_size=None,
        num_heads=8,
        mlp_dim=1024,
        H_cycles=2,
        L_cycles=2,
        gate_init=0.0,
    ):
        super().__init__()

        hidden_size = hidden_size or d_model

        # Projections into/out of HRM space
        self.in_proj = nn.Linear(d_model, hidden_size)
        self.out_proj = nn.Linear(hidden_size, d_model)

        # HRM-style blocks
        self.H_blocks = nn.ModuleList(
            [HRMBlock(hidden_size, num_heads, mlp_dim) for _ in range(H_cycles)]
        )
        self.L_blocks = nn.ModuleList(
            [HRMBlock(hidden_size, num_heads, mlp_dim) for _ in range(L_cycles)]
        )

        # Learnable gate (starts near zero → safe)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x):
        """
        x: [B, T_pool, d_model]
        """
        residual = x

        # Project to HRM hidden space
        h = self.in_proj(x)

        # Hierarchical reasoning cycles
        for blk in self.H_blocks:
            h = blk(h)

        for blk in self.L_blocks:
            h = blk(h)

        # Project back
        delta = self.out_proj(h)

        # Gated residual refinement
        return residual + torch.tanh(self.gate) * delta
