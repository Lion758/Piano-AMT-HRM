"""
Exponential Moving Average (EMA) helper for model weights.

Ported from reference/TinyRecursiveModels-main/models/ema.py with
additional checkpoint save/load support.
"""

import copy
import torch
import torch.nn as nn


class EMAHelper:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMAHelper(mu=0.999)
        ema.register(model)          # call once after model init

        # After each optimizer.step():
        ema.update(model)

        # For evaluation:
        ema_model = ema.ema_copy(model)   # returns a deep copy with EMA weights

        # Checkpoint:
        state = ema.state_dict()
        ema.load_state_dict(state)
    """

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module) -> None:
        """Initialize shadow weights from current model parameters."""
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        """Update shadow weights with EMA of current parameters."""
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].data = (
                    (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data
                )

    def ema(self, module: nn.Module) -> None:
        """Copy EMA weights into module (in-place)."""
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Return a deep copy of module with EMA weights applied."""
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
