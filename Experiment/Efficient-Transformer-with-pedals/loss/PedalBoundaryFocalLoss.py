import torch
import torch.nn as nn
import torch.nn.functional as Functional


def _config_get(config, key, default):
    if config is None:
        return default
    try:
        if key in config:
            return config[key]
    except TypeError:
        pass
    return getattr(config, key, default)


class PedalBoundaryFocalLoss(nn.Module):
    """Binary focal loss for sparse pedal onset/offset boundary heads.

    Inputs are [B, T] logits and soft targets in [0, 1]. The state head should
    keep using PedalFrameBCELoss because its class balance is very different.
    """

    def __init__(self, config=None):
        super().__init__()
        training_config = _config_get(config, "training", config)
        self.alpha = float(_config_get(training_config, "pedal_boundary_focal_alpha", 0.25))
        self.gamma = float(_config_get(training_config, "pedal_boundary_focal_gamma", 2.0))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"pedal_boundary_focal_alpha must be in [0, 1], got {self.alpha}.")
        if self.gamma < 0.0:
            raise ValueError(f"pedal_boundary_focal_gamma must be >= 0, got {self.gamma}.")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor):
        targets = targets.float()
        bce = Functional.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
        probabilities = torch.sigmoid(outputs)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        per_elem = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce

        if targets_mask is not None:
            mask = targets_mask.float()
            denom = mask.sum().clamp_min(1.0)
            return (per_elem * mask).sum() / denom

        return per_elem.mean()
