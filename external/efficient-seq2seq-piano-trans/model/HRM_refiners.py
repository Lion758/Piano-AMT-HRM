from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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
    steps: torch.Tensor  # [B]
    halted: torch.Tensor  # [B]


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
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.0,
        use_act_halt: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.one_step_grad = one_step_grad
        self.forward_dtype = forward_dtype
        self.halt_max_steps = max(1, halt_max_steps)
        self.halt_exploration_prob = halt_exploration_prob
        self.use_act_halt = use_act_halt

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
        self.halt_head = nn.Linear(d_model, 2)
        with torch.no_grad():
            self.halt_head.weight.zero_()
            self.halt_head.bias.fill_(-5.0)

        # Optional output gate (not in HRM core, but helps stability in encoder adapters)
        self.out_gate = nn.Parameter(torch.tensor(0.0))


    def empty_carry(self, batch_size: int, T: int, device=None) -> HRMEncoderCarry:
        device = device or self.H_init.device
        H0 = self.H_init.to(device).view(1,1,-1).expand(batch_size, T, -1).clone()
        L0 = self.L_init.to(device).view(1,1,-1).expand(batch_size, T, -1).clone()
        return HRMEncoderCarry(
            z_H=H0,
            z_L=L0,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
        )
    
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
            steps=torch.where(reset_flag, torch.zeros_like(carry.steps), carry.steps),
            halted=torch.where(reset_flag, torch.zeros_like(carry.halted), carry.halted),
        )

    def _forward_iteration(self, z_H: torch.Tensor, z_L: torch.Tensor, x_pool: torch.Tensor, seq_info: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.one_step_grad:
            with torch.no_grad():
                for h_step in range(self.H_cycles):
                    for l_step in range(self.L_cycles):
                        is_last = (h_step == self.H_cycles - 1) and (l_step == self.L_cycles - 1)
                        if not is_last:
                            z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)
                    if h_step != self.H_cycles - 1:
                        z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)

            z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)
            z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)
            return z_H, z_L

        for h_step in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x_pool, **seq_info)
            if h_step != self.H_cycles - 1:
                z_H = self.H_level(hidden_states=z_H, input_injection=z_L, **seq_info)
        return z_H, z_L

    def forward(
        self,
        x_pool: torch.Tensor,                # [B, T, d]
        carry: Optional[HRMEncoderCarry] = None,
        reset_flag: Optional[torch.Tensor] = None,  # [B] bool
    ) -> Tuple[torch.Tensor, HRMEncoderCarry, Dict[str, torch.Tensor]]:
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

        reset_mask = reset_flag | carry.halted if self.use_act_halt else reset_flag
        carry = self.reset_carry(reset_mask, carry)
        z_H, z_L = carry.z_H, carry.z_L
        steps = carry.steps
        halted = carry.halted

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb()
            seq_info = dict(cos_sin=(cos[:T], sin[:T]))
        else:
            seq_info = dict(cos_sin=None)

        max_steps = self.halt_max_steps if self.use_act_halt else 1
        halt_logits = self.halt_head(z_H[:, 0]).to(torch.float32)
        target_q_continue = None

        for _ in range(max_steps):
            active = ~halted
            if not torch.any(active):
                break

            next_z_H, next_z_L = self._forward_iteration(z_H, z_L, x_pool, seq_info)
            active_mask = active.view(-1, 1, 1)
            z_H = torch.where(active_mask, next_z_H, z_H)
            z_L = torch.where(active_mask, next_z_L, z_L)

            halt_logits = self.halt_head(z_H[:, 0]).to(torch.float32)
            q_halt_logits, q_continue_logits = halt_logits[..., 0], halt_logits[..., 1]

            steps = steps + active.to(steps.dtype)
            is_last_step = steps >= max_steps

            step_halted = is_last_step if self.use_act_halt else torch.zeros_like(is_last_step)
            if self.training and self.use_act_halt and (max_steps > 1):
                step_halted = step_halted | (q_halt_logits > q_continue_logits)

                if self.halt_exploration_prob > 0:
                    explore = (torch.rand_like(q_halt_logits) < self.halt_exploration_prob).to(steps.dtype)
                    min_halt_steps = explore * torch.randint_like(steps, low=2, high=max_steps + 1)
                    step_halted = step_halted & (steps >= min_halt_steps)

                with torch.no_grad():
                    next_bootstrap_H, _next_bootstrap_L = self._forward_iteration(
                        z_H.detach(),
                        z_L.detach(),
                        x_pool.detach(),
                        seq_info,
                    )
                    next_halt_logits = self.halt_head(next_bootstrap_H[:, 0]).to(torch.float32)
                    next_q_halt_logits, next_q_continue_logits = next_halt_logits[..., 0], next_halt_logits[..., 1]
                    target_q_continue = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    )

            halted = halted | (step_halted & active)

        new_carry = HRMEncoderCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=steps.detach(),
            halted=halted.detach(),
        )

        # Return refined memory. HRM “uses z_H as the output state”; we keep that.
        refined = x_pool + torch.tanh(self.out_gate) * z_H.to(x_pool.dtype)
        diagnostics = {
            "steps": steps,
            "halted": halted,
            "halt_logits": halt_logits,
            "q_halt_logits": halt_logits[..., 0],
            "q_continue_logits": halt_logits[..., 1],
        }
        if target_q_continue is not None:
            diagnostics["target_q_continue"] = target_q_continue

        return refined.to(orig_dtype), new_carry, diagnostics
