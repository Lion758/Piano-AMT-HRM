"""
TurboQuant KV Cache Compression (Zandieh et al., 2025)

Compresses key/value cache using:
  Stage 1 - TurboQuant_mse: Random orthogonal rotation + Lloyd-Max optimal quantization
  Stage 2 - QJL: 1-bit quantized Johnson-Lindenstrauss residual correction

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
           arXiv:2504.19874v1
"""

import torch
import torch.nn as nn
import math

# Module-level cache so the Lloyd-Max solver runs once per (dim, n_bits) pair.
_CODEBOOK_CACHE = {}


def _compute_lloyd_max_codebook(dim, n_bits):
    """
    Compute optimal Lloyd-Max codebook centroids for the coordinate distribution
    of a uniform random point on the unit hypersphere S^{d-1} (Lemma 1, Eq. 4).

    PDF: f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
         for x in [-1, 1]

    Returns:
        numpy array of shape [2^n_bits] with sorted centroids.
    """
    key = (dim, n_bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    from scipy import integrate
    from scipy.stats import norm as scipy_norm
    import numpy as np

    n_levels = 2 ** n_bits
    d = dim

    # Coefficient for the Beta distribution PDF (Lemma 1)
    log_coeff = math.lgamma(d / 2.0) - math.lgamma((d - 1) / 2.0) - 0.5 * math.log(math.pi)
    coeff = math.exp(log_coeff)
    exponent = (d - 3) / 2.0

    def pdf(x):
        t = 1.0 - x * x
        if t <= 0.0:
            return 0.0
        return coeff * (t ** exponent)

    # Initialize centroids at quantiles of the Gaussian approximation N(0, 1/d)
    sigma = 1.0 / math.sqrt(d)
    quantile_points = np.array([(2 * i + 1) / (2 * n_levels) for i in range(n_levels)])
    centroids = scipy_norm.ppf(quantile_points, loc=0, scale=sigma)
    centroids = np.clip(centroids, -1 + 1e-6, 1 - 1e-6)

    # Lloyd-Max iterations (solve the continuous 1-D k-means problem, Eq. 4)
    for _ in range(500):
        # Decision boundaries = midpoints between consecutive centroids
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        # Update each centroid to the conditional expectation within its Voronoi region
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-14:
                new_centroids[i] = (lo + hi) / 2.0
                continue
            numerator, _ = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=100)
            denominator, _ = integrate.quad(pdf, lo, hi, limit=100)
            if abs(denominator) > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2.0

        if np.max(np.abs(new_centroids - centroids)) < 1e-12:
            break
        centroids = new_centroids

    _CODEBOOK_CACHE[key] = centroids
    return centroids


class PolarQuant(nn.Module):
    """
    Stage 1: Random orthogonal rotation + Lloyd-Max optimal scalar quantization.

    Implements Algorithm 1 (TurboQuant_mse) from the paper:
      1. Rotate input by random orthogonal matrix Pi
      2. Quantize each coordinate to nearest precomputed Lloyd-Max centroid
      3. Dequantize by looking up centroids and rotating back
    """

    def __init__(self, dim, n_bits=4, seed=0):
        super().__init__()
        self.dim = dim
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        # Random rotation matrix Pi via QR decomposition (Algorithm 1, line 2)
        gen = torch.Generator().manual_seed(seed)
        gaussian = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(gaussian)
        self.register_buffer('rotation_matrix', Q)

        # Precompute Lloyd-Max codebook (Algorithm 1, line 3 / Eq. 4)
        import numpy as np
        codebook_np = _compute_lloyd_max_codebook(dim, n_bits)
        codebook = torch.from_numpy(np.array(codebook_np)).float()
        self.register_buffer('codebook', codebook)  # [n_levels], sorted ascending

        # Decision boundaries (midpoints between consecutive centroids)
        boundaries = (codebook[:-1] + codebook[1:]) / 2.0
        self.register_buffer('boundaries', boundaries)  # [n_levels - 1]

    def compress(self, x):
        """
        Quantize via rotation + nearest-centroid lookup (Algorithm 1, Quant_mse).

        Args:
            x: float tensor [..., dim]
        Returns:
            indices: int8 tensor [..., dim] -- codebook index per coordinate
            norms: float tensor [..., 1]   -- L2 norms for rescaling
        """
        # Store norms; normalize to unit sphere (paper assumes ||x|| = 1)
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # y <- Pi * x  (Algorithm 1, line 5)
        x_rot = x_unit @ self.rotation_matrix

        # idx_j <- argmin_k |y_j - c_k|  (Algorithm 1, line 6)
        indices = torch.searchsorted(self.boundaries, x_rot.contiguous())
        indices = indices.clamp(0, self.n_levels - 1).to(torch.int8)

        return indices, norms

    def decompress(self, indices, norms):
        """
        Reconstruct vectors from codebook indices (Algorithm 1, DeQuant_mse).

        Args:
            indices: int8 tensor [..., dim]
            norms: float tensor [..., 1]
        Returns:
            x_approx: float tensor [..., dim]
        """
        # y_tilde_j <- c_{idx_j}  (Algorithm 1, line 9)
        x_rot_approx = self.codebook[indices.long()]

        # x_tilde <- Pi^T * y_tilde  (Algorithm 1, line 10)
        x_approx = x_rot_approx @ self.rotation_matrix.T

        # Rescale by stored norms
        x_approx = x_approx * norms

        return x_approx


class QJLCorrection(nn.Module):
    """
    Stage 2: 1-bit Quantized Johnson-Lindenstrauss residual correction (Definition 1).

    Uses a d x d random Gaussian matrix S to produce d sign bits per vector,
    providing an unbiased inner product estimator on the residual.
    """

    def __init__(self, dim, seed=0):
        super().__init__()
        self.dim = dim

        # S in R^{d x d} with i.i.d. N(0,1) entries (Definition 1)
        gen = torch.Generator().manual_seed(seed)
        S = torch.randn(dim, dim, generator=gen)
        self.register_buffer('projection_matrix', S)

    def compress(self, residual):
        """
        Compress residual to 1-bit signs + norms (Algorithm 2, lines 7-8).

        Args:
            residual: float tensor [..., dim]
        Returns:
            signs: int8 tensor [..., dim]  (0 or 1)
            norms: float tensor [..., 1]   (||r||_2)
        """
        # qjl <- sign(S * r)  (Algorithm 2, line 7)
        projected = residual @ self.projection_matrix.T
        signs = (projected >= 0).to(torch.int8)
        norms = residual.norm(dim=-1, keepdim=True)
        return signs, norms

    def decompress(self, signs, norms):
        """
        Reconstruct approximate residual (Algorithm 2, line 11).

        x_tilde_qjl = sqrt(pi/2) / d * gamma * S^T * qjl

        Args:
            signs: int8 tensor [..., dim]
            norms: float tensor [..., 1]
        Returns:
            residual_approx: float tensor [..., dim]
        """
        # Convert 0/1 to -1/+1
        sign_values = 2.0 * signs.float() - 1.0

        # sqrt(pi/2) / d * S^T * z  (Definition 1 dequantization)
        # S^T * z in batch form: z @ S
        residual_approx = sign_values @ self.projection_matrix
        residual_approx = residual_approx * (math.sqrt(math.pi / 2.0) / self.dim)

        # Scale by gamma = ||r||_2  (Algorithm 2, line 11)
        residual_approx = residual_approx * norms

        return residual_approx


class TurboQuantCache(nn.Module):
    """
    Orchestrates PolarQuant + QJL for compressing the KV cache
    during autoregressive decoding.

    Stores compressed representations and decompresses on-the-fly
    for attention computation.
    """

    def __init__(self, head_dim, num_heads, n_bits=4,
                 qjl_projection_dim=None, layer_idx=0, enable_qjl=True,
                 min_cache_len=32):
        super().__init__()
        if n_bits not in (3, 4):
            raise ValueError(f"TurboQuant only supports 3-bit or 4-bit quantization, got {n_bits}.")
        if min_cache_len < 0:
            raise ValueError(f"TurboQuant min_cache_len must be >= 0, got {min_cache_len}.")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.enable_qjl = enable_qjl
        self.n_bits = n_bits
        self.min_cache_len = min_cache_len

        # PolarQuant instances for key and value (different seeds)
        self.key_pq = PolarQuant(head_dim, n_bits, seed=layer_idx * 2)
        self.value_pq = PolarQuant(head_dim, n_bits, seed=layer_idx * 2 + 1)

        # QJL correction instances (d x d projection per Definition 1)
        self.key_qjl = None
        self.value_qjl = None
        if enable_qjl:
            self.key_qjl = QJLCorrection(head_dim, seed=layer_idx * 2 + 1000)
            self.value_qjl = QJLCorrection(head_dim, seed=layer_idx * 2 + 1001)

        # Cache storage (set dynamically, not registered as buffers)
        self.cached_key_quantized = None
        self.cached_key_norms = None
        self.cached_value_quantized = None
        self.cached_value_norms = None

        # QJL cache storage
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_qjl_signs = None
        self.cached_value_qjl_norms = None
        self.reset_stats()

    def _compress_role(self, x, pq, qjl):
        """Compress a single role (key or value) through PolarQuant + optional QJL."""
        indices, norms = pq.compress(x)

        qjl_signs = None
        qjl_norms = None
        if qjl is not None:
            approx = pq.decompress(indices, norms)
            residual = x - approx
            qjl_signs, qjl_norms = qjl.compress(residual)

        return indices, norms, qjl_signs, qjl_norms

    def _decompress_role(self, indices, norms, qjl_signs, qjl_norms, pq, qjl):
        """Decompress a single role back to float tensors."""
        approx = pq.decompress(indices, norms)

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
            "qjl_projection_dim": int(self.head_dim),
        }

    def compress_and_cache(self, key, value):
        """
        Compress new key/value vectors and append to the cache.

        Args:
            key: float tensor [batch, new_len, num_heads, head_dim]
            value: float tensor [batch, new_len, num_heads, head_dim]
        """
        # Compress key
        k_idx, k_norms, k_signs, k_qjl_norms = self._compress_role(key, self.key_pq, self.key_qjl)
        self.cached_key_quantized = self._cat_or_init(self.cached_key_quantized, k_idx)
        self.cached_key_norms = self._cat_or_init(self.cached_key_norms, k_norms)

        # Compress value
        v_idx, v_norms, v_signs, v_qjl_norms = self._compress_role(value, self.value_pq, self.value_qjl)
        self.cached_value_quantized = self._cat_or_init(self.cached_value_quantized, v_idx)
        self.cached_value_norms = self._cat_or_init(self.cached_value_norms, v_norms)

        # QJL caches
        if self.enable_qjl:
            self.cached_key_qjl_signs = self._cat_or_init(self.cached_key_qjl_signs, k_signs)
            self.cached_key_qjl_norms = self._cat_or_init(self.cached_key_qjl_norms, k_qjl_norms)
            self.cached_value_qjl_signs = self._cat_or_init(self.cached_value_qjl_signs, v_signs)
            self.cached_value_qjl_norms = self._cat_or_init(self.cached_value_qjl_norms, v_qjl_norms)

    def get_decompressed(self):
        """
        Decompress the entire cached KV for attention computation.

        Returns:
            key: float tensor [batch, cache_len, num_heads, head_dim]
            value: float tensor [batch, cache_len, num_heads, head_dim]
        """
        key = self._decompress_role(
            self.cached_key_quantized, self.cached_key_norms,
            self.cached_key_qjl_signs, self.cached_key_qjl_norms,
            self.key_pq, self.key_qjl
        )
        value = self._decompress_role(
            self.cached_value_quantized, self.cached_value_norms,
            self.cached_value_qjl_signs, self.cached_value_qjl_norms,
            self.value_pq, self.value_qjl
        )
        return key, value

    def get_decompressed_slice(self, start: int, end: int):
        """
        Decompress a slice of the cached KV for blockwise attention.

        Args:
            start: inclusive sequence start index
            end: exclusive sequence end index
        Returns:
            key: float tensor [batch, end - start, num_heads, head_dim]
            value: float tensor [batch, end - start, num_heads, head_dim]
        """
        key = self._decompress_role(
            self.cached_key_quantized[:, start:end],
            self.cached_key_norms[:, start:end],
            None if self.cached_key_qjl_signs is None else self.cached_key_qjl_signs[:, start:end],
            None if self.cached_key_qjl_norms is None else self.cached_key_qjl_norms[:, start:end],
            self.key_pq,
            self.key_qjl,
        )
        value = self._decompress_role(
            self.cached_value_quantized[:, start:end],
            self.cached_value_norms[:, start:end],
            None if self.cached_value_qjl_signs is None else self.cached_value_qjl_signs[:, start:end],
            None if self.cached_value_qjl_norms is None else self.cached_value_qjl_norms[:, start:end],
            self.value_pq,
            self.value_qjl,
        )
        return key, value

    def clear(self):
        """Reset cache for new sequence generation."""
        self.cached_key_quantized = None
        self.cached_key_norms = None
        self.cached_value_quantized = None
        self.cached_value_norms = None
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_qjl_signs = None
        self.cached_value_qjl_norms = None

    def apply_sliding_window(self, window_size):
        """Truncate cache to last window_size entries along the sequence dim."""
        if self.cached_key_quantized is None:
            return
        self.cached_key_quantized = self.cached_key_quantized[:, -window_size:]
        self.cached_key_norms = self.cached_key_norms[:, -window_size:]
        self.cached_value_quantized = self.cached_value_quantized[:, -window_size:]
        self.cached_value_norms = self.cached_value_norms[:, -window_size:]

        if self.enable_qjl:
            self.cached_key_qjl_signs = self.cached_key_qjl_signs[:, -window_size:]
            self.cached_key_qjl_norms = self.cached_key_qjl_norms[:, -window_size:]
            self.cached_value_qjl_signs = self.cached_value_qjl_signs[:, -window_size:]
            self.cached_value_qjl_norms = self.cached_value_qjl_norms[:, -window_size:]
