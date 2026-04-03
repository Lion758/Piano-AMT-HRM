"""
Compatibility shim for the legacy extras TRM encoder module.

The canonical implementation now lives in model/trm_encoder.py.
"""

from model.trm_encoder import TrmEncoder, compute_trm_halt_loss

__all__ = ["TrmEncoder", "compute_trm_halt_loss"]
