import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Frequency- and time-masking SpecAugment (Park et al. 2019).

    Operates on a (B, T, F) log-magnitude tensor (post AmplitudeToDB), which
    matches the layout produced by `features_extracter(...).transpose(-1, -2)`
    in train.py. No time warping is applied — it is the most expensive piece
    of SpecAugment and contributes the least.
    """

    def __init__(
        self,
        freq_mask_param: int = 13,
        n_freq_masks: int = 2,
        time_mask_param: int = 40,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = int(freq_mask_param)
        self.n_freq_masks = int(n_freq_masks)
        self.time_mask_param = int(time_mask_param)
        self.n_time_masks = int(n_time_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if x.dim() != 3:
            return x

        B, T, F = x.shape
        out = x

        mask_value = out.mean()

        for _ in range(self.n_freq_masks):
            if self.freq_mask_param <= 0 or F <= 1:
                break
            f = torch.randint(0, self.freq_mask_param + 1, (B,), device=out.device)
            f0 = (torch.rand(B, device=out.device) * (F - f).clamp(min=1).float()).long()
            idx = torch.arange(F, device=out.device).unsqueeze(0).expand(B, F)
            mask = (idx >= f0.unsqueeze(1)) & (idx < (f0 + f).unsqueeze(1))
            out = torch.where(mask.unsqueeze(1), torch.full_like(out, mask_value.item()), out)

        for _ in range(self.n_time_masks):
            if self.time_mask_param <= 0 or T <= 1:
                break
            t = torch.randint(0, self.time_mask_param + 1, (B,), device=out.device)
            t0 = (torch.rand(B, device=out.device) * (T - t).clamp(min=1).float()).long()
            idx = torch.arange(T, device=out.device).unsqueeze(0).expand(B, T)
            mask = (idx >= t0.unsqueeze(1)) & (idx < (t0 + t).unsqueeze(1))
            out = torch.where(mask.unsqueeze(2), torch.full_like(out, mask_value.item()), out)

        return out
