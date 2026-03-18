"""
Compatibility shim for the legacy HRM encoder module.

The canonical implementation now lives in model/trm_encoder.py.
"""

from __future__ import annotations

import warnings

from model.trm_encoder import TrmEncoder, compute_trm_halt_loss


class HRMEncoder(TrmEncoder):
    def __init__(self, config) -> None:
        warnings.warn(
            "HRMEncoder is deprecated; use TrmEncoder and encoder_name='TrmEncoder'.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)


def compute_hrm_q_loss(*args, **kwargs):
    warnings.warn(
        "compute_hrm_q_loss is deprecated; the Q-learning halt/continue loss was "
        "removed and now forwards to compute_trm_halt_loss.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_trm_halt_loss(*args, **kwargs)


__all__ = ["HRMEncoder", "compute_hrm_q_loss"]
