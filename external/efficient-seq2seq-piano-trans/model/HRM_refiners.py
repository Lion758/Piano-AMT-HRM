from dataclasses import dataclass
from typing import Optional, Tuple
from types import SimpleNamespace

import torch
from torch import nn

# Reuse your existing implementations:
from model.layers.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Block, HierarchicalReasoningModel_ACTV1ReasoningModule
# - RotaryEmbedding (optional)
from model.layers.hrm_common import trunc_normal_init_
from model.layers.hrm_layers import RotaryEmbedding

@dataclass
class HRMEncoderCarry:
    z_H: torch.Tensor  # [B, T, d]
    z_L: torch.Tensor  # [B, T, d]


class HRMEncoderAdapter(nn.Module):
    """
    Encoder-friendly HRM core:
      input:  x_pool  [B, T, d_model]
      output: refined [B, T, d_model]

    Faithful HRM mechanics:
      - persistent z_H, z_L
      - nested H/L recurrence
      - restart dynamic (L converges under fixed H; H updates; L continues under new H)
      - optional one-step gradient trick
    """

    def __init__(
        self,
        d_model: int,
        H_cycles: int,
        L_cycles: int,
        H_layers: int,
        L_layers: int,
        num_heads: int,
        expansion: float = 4.0,
        rms_norm_eps: float = 1e-5,
        use_rope: bool = True, #False
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
        forward_dtype: torch.dtype = torch.bfloat16,
        one_step_grad: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.one_step_grad = one_step_grad
        self.forward_dtype = forward_dtype

        # Build HRM-style blocks (post-norm)
        # You can adapt your HierarchicalReasoningModel_ACTV1Block to accept a tiny config,
        # or just instantiate directly if it only needs hidden_size/heads/expansion/eps.
        def make_block():
            cfg = SimpleNamespace(
                hidden_size=d_model,
                expansion=expansion, 
                num_heads=num_heads,
                pos_encodings="rope",      # or anything your Attention path expects
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                seq_len=max_seq_len,
            )
            return HierarchicalReasoningModel_ACTV1Block(cfg)
        
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[make_block() for _ in range(H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[make_block() for _ in range(L_layers)]
        )

        # Optional RoPE cache provider
        self.rotary_emb = None
        if use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=d_model // num_heads,
                max_position_embeddings=max_seq_len,
                base=rope_theta,
            )

        # Initial states (learnable buffers like HRM uses)
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(d_model, dtype=forward_dtype), std=1.0))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(d_model, dtype=forward_dtype), std=1.0))

        # Optional output gate (not in HRM core, but helps stability in encoder adapters)
        self.out_gate = nn.Parameter(torch.tensor(0.0))


    def empty_carry(self, batch_size: int, T: int, device=None) -> HRMEncoderCarry:
        device = device or self.H_init.device
        H0 = self.H_init.to(device).view(1,1,-1).expand(batch_size, T, -1).clone()
        L0 = self.L_init.to(device).view(1,1,-1).expand(batch_size, T, -1).clone()
        return HRMEncoderCarry(z_H=H0, z_L=L0)
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRMEncoderCarry) -> HRMEncoderCarry:
        """
        reset_flag: [B] bool
        """
        H0 = self.H_init.to(carry.z_H.device).view(1,1,-1)
        L0 = self.L_init.to(carry.z_L.device).view(1,1,-1)
        mask = reset_flag.view(-1, 1, 1)
        return HRMEncoderCarry(
            z_H=torch.where(mask, H0, carry.z_H),
            z_L=torch.where(mask, L0, carry.z_L),
        )

    def forward(
        self,
        x_pool: torch.Tensor,                # [B, T, d]
        carry: Optional[HRMEncoderCarry] = None,
        reset_flag: Optional[torch.Tensor] = None,  # [B] bool
    ) -> Tuple[torch.Tensor, HRMEncoderCarry]:
        B, T, D = x_pool.shape
        assert D == self.d_model

        orig_dtype = x_pool.dtype
        x_pool = x_pool.to(self.forward_dtype)

        if carry is None:
            carry = self.empty_carry(B, T, device=x_pool.device)
            # Treat as reset-all when no carry is provided
            reset_flag = torch.ones((B,), dtype=torch.bool, device=x_pool.device)

        if reset_flag is None:
            reset_flag = torch.zeros((B,), dtype=torch.bool, device=x_pool.device)

        carry = self.reset_carry(reset_flag, carry)
        z_H, z_L = carry.z_H, carry.z_L

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb()
            seq_info = dict(cos_sin=(cos[:T], sin[:T]))
        else:
            seq_info = dict(cos_sin=None)

        # ---- HRM recurrence with restart dynamics ----
        if self.one_step_grad:
            # Unroll most steps without grad, then do last steps with grad (like the HRM code).
            with torch.no_grad():
                for h_step in range(self.H_cycles):
                    for l_step in range(self.L_cycles):
                        is_last = (h_step == self.H_cycles - 1) and (l_step == self.L_cycles - 1)
                        if not is_last:
                            # L update injects (z_H + x_pool)
                            z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)

                    # H updates between cycles (not on last cycle)
                    if h_step != self.H_cycles - 1:
                        z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)

            # Final “1-step grad” updates (faithful to HRM)
            z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)
            z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)

        else:
            # Full BPTT through recurrence (less HRM-faithful but sometimes useful for ablations)
            for h_step in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)
                if h_step != self.H_cycles - 1:
                    z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)

        new_carry = HRMEncoderCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # Return refined memory. HRM “uses z_H as the output state”; we keep that.
        refined = x_pool + torch.tanh(self.out_gate) * z_H.to(x_pool.dtype)
        return refined.to(orig_dtype), new_carry
