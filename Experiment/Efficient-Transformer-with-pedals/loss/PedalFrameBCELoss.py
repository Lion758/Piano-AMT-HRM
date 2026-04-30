import torch
import torch.nn as nn
import torch.nn.functional as Functional


class PedalFrameBCELoss(nn.Module):
    """BCE on per-frame sustain-pedal targets. Inputs are [B, T] logits / targets.

    Used for the auxiliary encoder pedal heads (state, onset, offset). Targets may be
    binary (state) or soft in [0, 1] (triangular onset/offset kernels) — both are
    valid inputs for binary_cross_entropy_with_logits.
    """

    def __init__(self, config=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, outputs, targets, targets_mask):
        # outputs: [B, T] logits, targets: [B, T] in [0, 1], mask: [B, T] {0, 1}
        if targets_mask is not None:
            mask = targets_mask.float()
            per_elem = Functional.binary_cross_entropy_with_logits(
                outputs, targets.float(), reduction="none"
            )
            denom = mask.sum().clamp_min(1.0)
            return (per_elem * mask).sum() / denom
        return self.criterion(outputs, targets.float())
