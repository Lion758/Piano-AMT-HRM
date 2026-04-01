"""
TurboQuant KV Cache Compression (Zandieh et al., 2025)

Compresses the KV cache using role-specific quantization:
  Keys:
    Stage 1 - TurboQuant_mse: random rotation + Lloyd-Max optimal quantization
    Stage 2 - QJL: 1-bit Quantized Johnson-Lindenstrauss residual correction
  Values:
    Stage 1 only - MSE-optimized PolarQuant at the configured scalar bit-width

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
           arXiv:2504.19874v1
"""

import torch
import torch.nn as nn
import math

# Module-level cache so the Lloyd-Max solver runs once per (dim, n_bits) pair.
_CODEBOOK_CACHE = {}


def _packed_byte_width(dim, bit_width):
    """Number of bytes needed to pack dim values at bit_width bits each."""
    return (dim * bit_width + 7) // 8


def _pack_lowbit_tensor(values, bit_width):
    """
    Pack the last dimension of an integer tensor into uint8 storage.

    Args:
        values: integer tensor [..., dim]
        bit_width: number of bits per element, supported range [1, 4]
    Returns:
        packed: uint8 tensor [..., ceil(dim * bit_width / 8)]
    """
    if bit_width < 1 or bit_width > 4:
        raise ValueError(f"bit_width must be between 1 and 4, got {bit_width}.")

    original_shape = values.shape
    if len(original_shape) == 0:
        raise ValueError("values must have at least one dimension to pack.")

    flat = values.reshape(-1, original_shape[-1]).to(torch.int64)
    max_value = (1 << bit_width) - 1
    if flat.numel() > 0:
        min_value = int(flat.amin().item())
        max_observed = int(flat.amax().item())
        if min_value < 0 or max_observed > max_value:
            raise ValueError(
                f"values must be in [0, {max_value}] for {bit_width}-bit packing, "
                f"got range [{min_value}, {max_observed}]."
            )

    shift_offsets = torch.arange(bit_width - 1, -1, -1, device=flat.device, dtype=torch.int64)
    bits = ((flat.unsqueeze(-1) >> shift_offsets) & 1).to(torch.uint8).reshape(flat.size(0), -1)

    pad_bits = (-bits.size(-1)) % 8
    if pad_bits:
        padding = torch.zeros(bits.size(0), pad_bits, dtype=torch.uint8, device=bits.device)
        bits = torch.cat([bits, padding], dim=-1)

    bit_chunks = bits.reshape(bits.size(0), -1, 8).to(torch.int64)
    bit_weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int64, device=flat.device)
    packed = (bit_chunks * bit_weights).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*original_shape[:-1], packed.size(-1))


def _unpack_lowbit_tensor(packed, bit_width, original_dim):
    """
    Unpack a uint8 tensor produced by _pack_lowbit_tensor.

    Args:
        packed: uint8 tensor [..., packed_bytes]
        bit_width: number of bits per packed element, supported range [1, 4]
        original_dim: logical size of the unpacked last dimension
    Returns:
        unpacked: uint8 tensor [..., original_dim]
    """
    if bit_width < 1 or bit_width > 4:
        raise ValueError(f"bit_width must be between 1 and 4, got {bit_width}.")
    if original_dim < 0:
        raise ValueError(f"original_dim must be non-negative, got {original_dim}.")

    original_shape = packed.shape
    if len(original_shape) == 0:
        raise ValueError("packed must have at least one dimension to unpack.")

    flat = packed.reshape(-1, original_shape[-1]).to(torch.int64)
    bit_offsets = torch.arange(7, -1, -1, device=flat.device, dtype=torch.int64)
    bits = ((flat.unsqueeze(-1) >> bit_offsets) & 1).to(torch.uint8).reshape(flat.size(0), -1)

    total_bits = original_dim * bit_width
    bits = bits[:, :total_bits]
    if total_bits == 0:
        unpacked = torch.zeros(flat.size(0), 0, dtype=torch.uint8, device=flat.device)
        return unpacked.reshape(*original_shape[:-1], 0)

    value_bits = bits.reshape(flat.size(0), original_dim, bit_width).to(torch.int64)
    value_weights = (1 << torch.arange(bit_width - 1, -1, -1, device=flat.device, dtype=torch.int64))
    unpacked = (value_bits * value_weights).sum(dim=-1).to(torch.uint8)
    return unpacked.reshape(*original_shape[:-1], original_dim)


def _pack_indices(indices, bit_width):
    """Pack scalar-quantizer indices into uint8 bytes."""
    return _pack_lowbit_tensor(indices, bit_width)


def _unpack_indices(packed, bit_width, original_dim):
    """Unpack scalar-quantizer indices from uint8 bytes."""
    return _unpack_lowbit_tensor(packed, bit_width, original_dim)


def _pack_signs(signs):
    """Pack 0/1 QJL signs into uint8 bytes."""
    return _pack_lowbit_tensor(signs, 1)


def _unpack_signs(packed, original_dim):
    """Unpack 0/1 QJL signs from uint8 bytes."""
    return _unpack_lowbit_tensor(packed, 1, original_dim)


def _tensor_storage_bytes(tensor):
    """Return the backing storage size of a tensor in bytes."""
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _haar_random_rotation(dim, seed):
    """
    Generate a Haar-distributed random rotation matrix using QR decomposition.

    The sign correction on the diagonal of R makes Q Haar-distributed, and the
    determinant fix keeps the matrix in SO(d) instead of O(d).
    """
    gen = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(gaussian)

    diag = torch.diagonal(R)
    signs = torch.sign(diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Q = Q * signs.unsqueeze(0)

    det_sign, _ = torch.linalg.slogdet(Q)
    if det_sign.item() < 0:
        Q = Q.clone()
        Q[:, 0] = -Q[:, 0]

    return Q


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

        # Random rotation matrix Pi via Haar-corrected QR decomposition
        self.register_buffer('rotation_matrix', _haar_random_rotation(dim, seed))

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
            indices: uint8 tensor [..., dim] -- codebook index per coordinate
            norms: float tensor [..., 1]   -- L2 norms for rescaling
        """
        # Store norms; normalize to unit sphere (paper assumes ||x|| = 1)
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # y <- Pi * x  (Algorithm 1, line 5)
        x_rot = x_unit @ self.rotation_matrix

        # idx_j <- argmin_k |y_j - c_k|  (Algorithm 1, line 6)
        indices = torch.searchsorted(self.boundaries, x_rot.contiguous())
        indices = indices.clamp(0, self.n_levels - 1).to(torch.uint8)

        return indices, norms

    def decompress(self, indices, norms):
        """
        Reconstruct vectors from codebook indices (Algorithm 1, DeQuant_mse).

        Args:
            indices: integer tensor [..., dim]
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

    Uses an m x d random Gaussian matrix S to produce m sign bits per vector,
    providing an unbiased inner product estimator on the residual while allowing
    the key-side sketch width to differ from head_dim.
    """

    def __init__(self, input_dim, projection_dim=None, seed=0):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = input_dim if projection_dim is None else int(projection_dim)
        if self.projection_dim <= 0:
            raise ValueError(f"QJL projection_dim must be a positive integer, got {projection_dim}.")

        # S in R^{m x d} with i.i.d. N(0,1) entries.
        gen = torch.Generator().manual_seed(seed)
        S = torch.randn(self.projection_dim, input_dim, generator=gen)
        self.register_buffer('projection_matrix', S)

    def compress(self, residual):
        """
        Compress residual to 1-bit signs + norms (Algorithm 2, lines 7-8).

        Args:
            residual: float tensor [..., input_dim]
        Returns:
            signs: uint8 tensor [..., projection_dim]  (0 or 1)
            norms: float tensor [..., 1]   (||r||_2)
        """
        # qjl <- sign(S * r)
        projected = residual @ self.projection_matrix.T
        signs = (projected >= 0).to(torch.uint8)
        norms = residual.norm(dim=-1, keepdim=True)
        return signs, norms

    def decompress(self, signs, norms):
        """
        Reconstruct approximate residual (Algorithm 2, line 11).

        x_tilde_qjl = sqrt(pi/2) / m * gamma * S^T * qjl

        Args:
            signs: integer tensor [..., projection_dim] in {0, 1}
            norms: float tensor [..., 1]
        Returns:
            residual_approx: float tensor [..., input_dim]
        """
        # Convert 0/1 to -1/+1
        sign_values = 2.0 * signs.float() - 1.0

        # sqrt(pi/2) / m * S^T * z
        # S^T * z in batch form: z @ S
        residual_approx = sign_values @ self.projection_matrix
        residual_approx = residual_approx * (math.sqrt(math.pi / 2.0) / self.projection_dim)

        # Scale by gamma = ||r||_2  (Algorithm 2, line 11)
        residual_approx = residual_approx * norms

        return residual_approx


class TurboQuantCache(nn.Module):
    """
    Orchestrates role-specific TurboQuant for compressing the KV cache
    during autoregressive decoding.

    Keys use TurboQuant's two-stage inner-product path when QJL is enabled.
    Values always use MSE-only PolarQuant at the full configured scalar width.
    """

    def __init__(self, head_dim, num_heads, n_bits=4,
                 qjl_projection_dim=None, layer_idx=0, enable_qjl=True,
                 min_cache_len=32):
        super().__init__()
        if n_bits not in (3, 4):
            raise ValueError(f"TurboQuant only supports 3-bit or 4-bit total quantization, got {n_bits}.")
        if min_cache_len < 0:
            raise ValueError(f"TurboQuant min_cache_len must be >= 0, got {min_cache_len}.")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.enable_qjl = enable_qjl
        self.n_bits = n_bits
        self.key_mse_n_bits = n_bits - 1 if enable_qjl else n_bits
        self.value_mse_n_bits = n_bits
        if self.key_mse_n_bits < 1:
            raise ValueError(
                f"TurboQuant requires at least 1 PolarQuant bit, got total n_bits={n_bits} "
                f"with enable_qjl={enable_qjl}."
            )
        self.min_cache_len = min_cache_len
        resolved_qjl_projection_dim = head_dim if qjl_projection_dim is None else int(qjl_projection_dim)
        if resolved_qjl_projection_dim <= 0:
            raise ValueError(
                f"TurboQuant qjl_projection_dim must be a positive integer or None, got {qjl_projection_dim}."
            )
        self.key_qjl_projection_dim = resolved_qjl_projection_dim if enable_qjl else 0
        # When key_qjl_projection_dim == head_dim, the key path matches the nominal
        # total n_bits budget. Smaller/larger projections intentionally change the
        # effective key bitrate away from the paper's exact 1-bit QJL stage.
        self.key_effective_bits_per_coord = (
            float(self.key_mse_n_bits) + (float(self.key_qjl_projection_dim) / float(head_dim))
            if enable_qjl else float(self.key_mse_n_bits)
        )
        self.mse_n_bits = self.key_mse_n_bits  # Backward-compatible alias for the key path.
        self.qjl_projection_dim = self.key_qjl_projection_dim
        self.key_index_packed_bytes = _packed_byte_width(head_dim, self.key_mse_n_bits)
        self.value_index_packed_bytes = _packed_byte_width(head_dim, self.value_mse_n_bits)
        self.key_qjl_packed_bytes = _packed_byte_width(self.key_qjl_projection_dim, 1) if enable_qjl else 0

        # PolarQuant instances for key and value (different seeds).
        self.key_pq = PolarQuant(head_dim, self.key_mse_n_bits, seed=layer_idx * 2)
        self.value_pq = PolarQuant(head_dim, self.value_mse_n_bits, seed=layer_idx * 2 + 1)

        # Key-only QJL correction instance.
        self.key_qjl = None
        if enable_qjl:
            self.key_qjl = QJLCorrection(
                input_dim=head_dim,
                projection_dim=self.key_qjl_projection_dim,
                seed=layer_idx * 2 + 1000,
            )

        # Packed cache storage (set dynamically, not registered as buffers)
        self.cached_key_quantized = None
        self.cached_key_norms = None
        self.cached_value_quantized = None
        self.cached_value_norms = None

        # Packed key-side QJL cache storage
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.reset_stats()

    def _compress_key(self, key):
        """Compress key vectors through PolarQuant + optional key-side QJL."""
        indices, norms = self.key_pq.compress(key)

        qjl_signs = None
        qjl_norms = None
        if self.key_qjl is not None:
            approx = self.key_pq.decompress(indices, norms)
            residual = key - approx
            qjl_signs, qjl_norms = self.key_qjl.compress(residual)

        return indices, norms, qjl_signs, qjl_norms

    def _compress_value(self, value):
        """Compress value vectors with MSE-only PolarQuant."""
        return self.value_pq.compress(value)

    def _decompress_key(self, packed_indices, norms, packed_qjl_signs=None, qjl_norms=None):
        """Decompress packed key tensors back to dense floating point."""
        indices = _unpack_indices(packed_indices, self.key_pq.n_bits, self.head_dim)
        approx = self.key_pq.decompress(indices, norms)

        if self.key_qjl is not None and packed_qjl_signs is not None:
            qjl_signs = _unpack_signs(packed_qjl_signs, self.key_qjl.projection_dim)
            residual_approx = self.key_qjl.decompress(qjl_signs, qjl_norms)
            approx = approx + residual_approx

        return approx

    def _decompress_value(self, packed_indices, norms):
        """Decompress packed value tensors back to dense floating point."""
        indices = _unpack_indices(packed_indices, self.value_pq.n_bits, self.head_dim)
        return self.value_pq.decompress(indices, norms)

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
        key_index_bytes = _tensor_storage_bytes(self.cached_key_quantized)
        value_index_bytes = _tensor_storage_bytes(self.cached_value_quantized)
        key_qjl_bytes = _tensor_storage_bytes(self.cached_key_qjl_signs)
        value_qjl_bytes = 0
        key_norm_bytes = _tensor_storage_bytes(self.cached_key_norms) + _tensor_storage_bytes(self.cached_key_qjl_norms)
        value_norm_bytes = _tensor_storage_bytes(self.cached_value_norms)

        key_vector_count = 0 if self.cached_key_norms is None else int(self.cached_key_norms.numel())
        value_vector_count = 0 if self.cached_value_norms is None else int(self.cached_value_norms.numel())
        unpacked_key_discrete_bytes = key_vector_count * self.head_dim
        unpacked_value_discrete_bytes = value_vector_count * self.head_dim
        if self.enable_qjl:
            unpacked_key_discrete_bytes += key_vector_count * self.key_qjl_projection_dim

        return {
            "short_prefix_steps": int(self.short_prefix_steps),
            "quantized_steps": int(self.quantized_steps),
            "max_cache_len": int(self.max_cache_len),
            "last_cache_len": int(self.last_cache_len),
            "last_window_size": self.last_window_size,
            "min_cache_len": int(self.min_cache_len),
            "n_bits": int(self.n_bits),
            "mse_n_bits": int(self.key_mse_n_bits),
            "key_mse_n_bits": int(self.key_mse_n_bits),
            "value_mse_n_bits": int(self.value_mse_n_bits),
            "enable_qjl": bool(self.enable_qjl),
            "qjl_projection_dim": int(self.key_qjl_projection_dim),
            "key_qjl_projection_dim": int(self.key_qjl_projection_dim),
            "key_effective_bits_per_coord": float(self.key_effective_bits_per_coord),
            "packed_key_index_bytes": key_index_bytes,
            "packed_value_index_bytes": value_index_bytes,
            "packed_key_qjl_bytes": key_qjl_bytes,
            "packed_value_qjl_bytes": value_qjl_bytes,
            "packed_key_discrete_bytes": key_index_bytes + key_qjl_bytes,
            "packed_value_discrete_bytes": value_index_bytes + value_qjl_bytes,
            "norm_bytes": key_norm_bytes + value_norm_bytes,
            "total_bytes": key_index_bytes + value_index_bytes + key_qjl_bytes + value_qjl_bytes + key_norm_bytes + value_norm_bytes,
            "unpacked_key_discrete_bytes": unpacked_key_discrete_bytes,
            "unpacked_value_discrete_bytes": unpacked_value_discrete_bytes,
            "unpacked_discrete_bytes": unpacked_key_discrete_bytes + unpacked_value_discrete_bytes,
        }

    def compress_and_cache(self, key, value):
        """
        Compress new key/value vectors and append to the cache.

        Args:
            key: float tensor [batch, new_len, num_heads, head_dim]
            value: float tensor [batch, new_len, num_heads, head_dim]
        """
        # Compress key
        k_idx, k_norms, k_signs, k_qjl_norms = self._compress_key(key)
        self.cached_key_quantized = self._cat_or_init(
            self.cached_key_quantized,
            _pack_indices(k_idx, self.key_mse_n_bits),
        )
        self.cached_key_norms = self._cat_or_init(self.cached_key_norms, k_norms)

        # Compress value
        v_idx, v_norms = self._compress_value(value)
        self.cached_value_quantized = self._cat_or_init(
            self.cached_value_quantized,
            _pack_indices(v_idx, self.value_mse_n_bits),
        )
        self.cached_value_norms = self._cat_or_init(self.cached_value_norms, v_norms)

        # Key-side QJL cache.
        if self.key_qjl is not None:
            self.cached_key_qjl_signs = self._cat_or_init(self.cached_key_qjl_signs, _pack_signs(k_signs))
            self.cached_key_qjl_norms = self._cat_or_init(self.cached_key_qjl_norms, k_qjl_norms)

    def get_decompressed(self):
        """
        Decompress the entire cached KV for attention computation.

        Returns:
            key: float tensor [batch, cache_len, num_heads, head_dim]
            value: float tensor [batch, cache_len, num_heads, head_dim]
        """
        key = self._decompress_key(
            self.cached_key_quantized, self.cached_key_norms,
            self.cached_key_qjl_signs, self.cached_key_qjl_norms,
        )
        value = self._decompress_value(
            self.cached_value_quantized, self.cached_value_norms,
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
        key = self._decompress_key(
            self.cached_key_quantized[:, start:end],
            self.cached_key_norms[:, start:end],
            None if self.cached_key_qjl_signs is None else self.cached_key_qjl_signs[:, start:end],
            None if self.cached_key_qjl_norms is None else self.cached_key_qjl_norms[:, start:end],
        )
        value = self._decompress_value(
            self.cached_value_quantized[:, start:end],
            self.cached_value_norms[:, start:end],
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

    def apply_sliding_window(self, window_size):
        """Truncate cache to last window_size entries along the sequence dim."""
        if self.cached_key_quantized is None:
            return
        self.cached_key_quantized = self.cached_key_quantized[:, -window_size:]
        self.cached_key_norms = self.cached_key_norms[:, -window_size:]
        self.cached_value_quantized = self.cached_value_quantized[:, -window_size:]
        self.cached_value_norms = self.cached_value_norms[:, -window_size:]

        if self.key_qjl is not None:
            self.cached_key_qjl_signs = self.cached_key_qjl_signs[:, -window_size:]
            self.cached_key_qjl_norms = self.cached_key_qjl_norms[:, -window_size:]
