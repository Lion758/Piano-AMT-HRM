"""
TurboQuant KV Cache Compression (Google, ICLR 2026)

Compresses key/value cache to 3-4 bits per element without retraining using:
  Stage 1 - PolarQuant: Random orthogonal rotation + uniform quantization
  Stage 2 - QJL: 1-bit quantized Johnson-Lindenstrauss residual correction
"""

import torch
import torch.nn as nn
import math


class PolarQuant(nn.Module):
    """Stage 1: Random orthogonal rotation followed by uniform quantization."""

    def __init__(self, dim, n_bits=4, seed=0):
        super().__init__()
        self.dim = dim
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        # Generate a fixed random orthogonal matrix via QR decomposition
        gen = torch.Generator().manual_seed(seed)
        gaussian = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(gaussian)
        self.register_buffer('rotation_matrix', Q)

    def compress(self, x):
        """
        Compress float vectors via rotation + uniform quantization.

        Args:
            x: float tensor [..., dim]
        Returns:
            quantized: int8 tensor [..., dim]
            scale: float tensor [..., 1]
            zero_point: float tensor [..., 1]  (per-token min)
        """
        # Rotate to spread energy uniformly
        x_rot = x @ self.rotation_matrix

        # Per-token adaptive range
        x_min = x_rot.amin(dim=-1, keepdim=True)
        x_max = x_rot.amax(dim=-1, keepdim=True)

        # Avoid division by zero for constant vectors
        scale = (x_max - x_min) / (self.n_levels - 1)
        scale = scale.clamp(min=1e-8)

        quantized = torch.round((x_rot - x_min) / scale).to(torch.int8)
        return quantized, scale, x_min

    def decompress(self, quantized, scale, zero_point):
        """
        Decompress quantized vectors back to float.

        Args:
            quantized: int8 tensor [..., dim]
            scale: float tensor [..., 1]
            zero_point: float tensor [..., 1]
        Returns:
            x_approx: float tensor [..., dim]
        """
        x_rot_approx = quantized.float() * scale + zero_point
        x_approx = x_rot_approx @ self.rotation_matrix.T
        return x_approx


class QJLCorrection(nn.Module):
    """Stage 2: 1-bit quantized Johnson-Lindenstrauss residual correction."""

    def __init__(self, dim, projection_dim=128, seed=0):
        super().__init__()
        self.dim = dim
        self.projection_dim = projection_dim

        # Generate normalized random projection matrix
        gen = torch.Generator().manual_seed(seed)
        P = torch.randn(dim, projection_dim, generator=gen)
        P = P / math.sqrt(projection_dim)
        self.register_buffer('projection_matrix', P)

    def compress(self, residual):
        """
        Compress residual error to 1-bit signs + norms.

        Args:
            residual: float tensor [..., dim]
        Returns:
            signs: int8 tensor [..., projection_dim]  (0 or 1)
            norms: float tensor [..., 1]
        """
        projected = residual @ self.projection_matrix
        signs = (projected >= 0).to(torch.int8)
        norms = residual.norm(dim=-1, keepdim=True)
        return signs, norms

    def decompress(self, signs, norms):
        """
        Reconstruct approximate residual from 1-bit projections.

        Args:
            signs: int8 tensor [..., projection_dim]
            norms: float tensor [..., 1]
        Returns:
            residual_approx: float tensor [..., dim]
        """
        sign_values = 2.0 * signs.float() - 1.0
        residual_approx = sign_values @ self.projection_matrix.T
        residual_approx = residual_approx * norms / math.sqrt(self.projection_dim)
        return residual_approx


class TurboQuantCache(nn.Module):
    """
    Orchestrates PolarQuant + QJL for compressing the KV cache
    during autoregressive decoding.

    Stores compressed representations and decompresses on-the-fly
    for attention computation.
    """

    def __init__(self, head_dim, num_heads, n_bits=4,
                 qjl_projection_dim=128, layer_idx=0, enable_qjl=True,
                 min_cache_len=32):
        super().__init__()
        if n_bits not in (3, 4):
            raise ValueError(f"TurboQuant only supports 3-bit or 4-bit quantization, got {n_bits}.")
        if min_cache_len < 0:
            raise ValueError(f"TurboQuant min_cache_len must be >= 0, got {min_cache_len}.")
        if enable_qjl:
            if qjl_projection_dim <= 0:
                raise ValueError(f"TurboQuant qjl_projection_dim must be > 0, got {qjl_projection_dim}.")
            if qjl_projection_dim > head_dim:
                raise ValueError(
                    "TurboQuant qjl_projection_dim must be <= head_dim for the current speed profile: "
                    f"got qjl_projection_dim={qjl_projection_dim}, head_dim={head_dim}."
                )
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.enable_qjl = enable_qjl
        self.n_bits = n_bits
        self.qjl_projection_dim = qjl_projection_dim
        self.min_cache_len = min_cache_len

        # PolarQuant instances for key and value (different seeds)
        self.key_pq = PolarQuant(head_dim, n_bits, seed=layer_idx * 2)
        self.value_pq = PolarQuant(head_dim, n_bits, seed=layer_idx * 2 + 1)

        # QJL correction instances (optional)
        self.key_qjl = None
        self.value_qjl = None
        if enable_qjl:
            self.key_qjl = QJLCorrection(head_dim, qjl_projection_dim, seed=layer_idx * 2 + 1000)
            self.value_qjl = QJLCorrection(head_dim, qjl_projection_dim, seed=layer_idx * 2 + 1001)

        # Cache storage (set dynamically, not registered as buffers)
        self.cached_key_quantized = None
        self.cached_key_scale = None
        self.cached_key_zp = None
        self.cached_value_quantized = None
        self.cached_value_scale = None
        self.cached_value_zp = None

        # QJL cache storage
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_qjl_signs = None
        self.cached_value_qjl_norms = None
        self.reset_stats()

    def _compress_role(self, x, pq, qjl):
        """Compress a single role (key or value) through PolarQuant + optional QJL."""
        quantized, scale, zp = pq.compress(x)

        qjl_signs = None
        qjl_norms = None
        if qjl is not None:
            approx = pq.decompress(quantized, scale, zp)
            residual = x - approx
            qjl_signs, qjl_norms = qjl.compress(residual)

        return quantized, scale, zp, qjl_signs, qjl_norms

    def _decompress_role(self, quantized, scale, zp, qjl_signs, qjl_norms, pq, qjl):
        """Decompress a single role back to float tensors."""
        approx = pq.decompress(quantized, scale, zp)

        if qjl is not None and qjl_signs is not None:
            residual_approx = qjl.decompress(qjl_signs, qjl_norms)
            approx = approx + residual_approx

        return approx

    def _cat_or_init(self, cached, new):
        """Concatenate new tensor to cache along dim=1, or initialize if cache is None."""
        if cached is None:
            return new
        return torch.cat([cached, new], dim=1)

    def should_quantize(self, cache_len):
        return cache_len >= self.min_cache_len

    def has_cached_values(self):
        return self.cached_key_quantized is not None

    def get_cache_len(self):
        if self.cached_key_quantized is None:
            return 0
        return int(self.cached_key_quantized.size(1))

    def reset_stats(self):
        self.short_prefix_steps = 0
        self.quantized_steps = 0
        self.max_cache_len = 0
        self.last_cache_len = 0
        self.last_window_size = None

    def _record_step(self, cache_len, window_size, used_quantized):
        cache_len = int(cache_len)
        if used_quantized:
            self.quantized_steps += 1
        else:
            self.short_prefix_steps += 1
        self.max_cache_len = max(self.max_cache_len, cache_len)
        self.last_cache_len = cache_len
        self.last_window_size = int(window_size) if window_size is not None else None

    def note_short_prefix_step(self, cache_len, window_size=None):
        self._record_step(cache_len, window_size, used_quantized=False)

    def note_quantized_step(self, cache_len, window_size=None):
        self._record_step(cache_len, window_size, used_quantized=True)

    def get_stats(self):
        return {
            "short_prefix_steps": int(self.short_prefix_steps),
            "quantized_steps": int(self.quantized_steps),
            "max_cache_len": int(self.max_cache_len),
            "last_cache_len": int(self.last_cache_len),
            "last_window_size": self.last_window_size,
            "min_cache_len": int(self.min_cache_len),
            "n_bits": int(self.n_bits),
            "enable_qjl": bool(self.enable_qjl),
            "qjl_projection_dim": int(self.qjl_projection_dim),
        }

    def compress_and_cache(self, key, value):
        """
        Compress new key/value vectors and append to the cache.

        Args:
            key: float tensor [batch, new_len, num_heads, head_dim]
            value: float tensor [batch, new_len, num_heads, head_dim]
        """
        # Compress key
        k_q, k_s, k_zp, k_signs, k_norms = self._compress_role(key, self.key_pq, self.key_qjl)
        self.cached_key_quantized = self._cat_or_init(self.cached_key_quantized, k_q)
        self.cached_key_scale = self._cat_or_init(self.cached_key_scale, k_s)
        self.cached_key_zp = self._cat_or_init(self.cached_key_zp, k_zp)

        # Compress value
        v_q, v_s, v_zp, v_signs, v_norms = self._compress_role(value, self.value_pq, self.value_qjl)
        self.cached_value_quantized = self._cat_or_init(self.cached_value_quantized, v_q)
        self.cached_value_scale = self._cat_or_init(self.cached_value_scale, v_s)
        self.cached_value_zp = self._cat_or_init(self.cached_value_zp, v_zp)

        # QJL caches
        if self.enable_qjl:
            self.cached_key_qjl_signs = self._cat_or_init(self.cached_key_qjl_signs, k_signs)
            self.cached_key_qjl_norms = self._cat_or_init(self.cached_key_qjl_norms, k_norms)
            self.cached_value_qjl_signs = self._cat_or_init(self.cached_value_qjl_signs, v_signs)
            self.cached_value_qjl_norms = self._cat_or_init(self.cached_value_qjl_norms, v_norms)

    def get_decompressed(self):
        """
        Decompress the entire cached KV for attention computation.

        Returns:
            key: float tensor [batch, cache_len, num_heads, head_dim]
            value: float tensor [batch, cache_len, num_heads, head_dim]
        """
        key = self._decompress_role(
            self.cached_key_quantized, self.cached_key_scale, self.cached_key_zp,
            self.cached_key_qjl_signs, self.cached_key_qjl_norms,
            self.key_pq, self.key_qjl
        )
        value = self._decompress_role(
            self.cached_value_quantized, self.cached_value_scale, self.cached_value_zp,
            self.cached_value_qjl_signs, self.cached_value_qjl_norms,
            self.value_pq, self.value_qjl
        )
        return key, value

    def clear(self):
        """Reset cache for new sequence generation."""
        self.cached_key_quantized = None
        self.cached_key_scale = None
        self.cached_key_zp = None
        self.cached_value_quantized = None
        self.cached_value_scale = None
        self.cached_value_zp = None
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_qjl_signs = None
        self.cached_value_qjl_norms = None

    def apply_sliding_window(self, window_size):
        """Truncate cache to last window_size entries along the sequence dim."""
        if self.cached_key_quantized is None:
            return
        self.cached_key_quantized = self.cached_key_quantized[:, -window_size:]
        self.cached_key_scale = self.cached_key_scale[:, -window_size:]
        self.cached_key_zp = self.cached_key_zp[:, -window_size:]
        self.cached_value_quantized = self.cached_value_quantized[:, -window_size:]
        self.cached_value_scale = self.cached_value_scale[:, -window_size:]
        self.cached_value_zp = self.cached_value_zp[:, -window_size:]

        if self.enable_qjl:
            self.cached_key_qjl_signs = self.cached_key_qjl_signs[:, -window_size:]
            self.cached_key_qjl_norms = self.cached_key_qjl_norms[:, -window_size:]
            self.cached_value_qjl_signs = self.cached_value_qjl_signs[:, -window_size:]
            self.cached_value_qjl_norms = self.cached_value_qjl_norms[:, -window_size:]
