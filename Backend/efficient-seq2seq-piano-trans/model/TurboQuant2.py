"""
TurboQuant2: Enhanced KV Cache Compression (based on TurboQuant+ techniques)

Extends the base TurboQuant with:
  - Asymmetric K/V compression: Keys get higher precision than values
  - Layer-aware boundary V compression: First/last decoder layers get higher V precision
  - Sparse V dequantization: Skip V dequant for negligible attention positions
  - Outlier channel strategy: Fractional bit rates via channel splitting

Only active when turbo_quant_v2 config is enabled.
"""

import torch
import torch.nn as nn
import math

from model.TurboQuant import (
    PolarQuant, QJLCorrection,
    _pack_indices, _unpack_indices,
    _pack_signs, _unpack_signs,
    _packed_byte_width, _tensor_storage_bytes,
)


class OutlierPolarQuant(nn.Module):
    """
    Splits head_dim channels into outlier (higher bits) and normal (lower bits)
    groups for fractional average bit rates.

    E.g., with head_dim=64, outlier_fraction=0.25, outlier_n_bits=4, normal_n_bits=2:
      16 outlier channels at 4 bits + 48 normal channels at 2 bits = 2.5 avg bits
    """

    def __init__(self, head_dim, outlier_fraction=0.25, outlier_n_bits=4,
                 normal_n_bits=2, seed=0):
        super().__init__()
        self.head_dim = head_dim
        n_outlier = max(1, int(round(head_dim * outlier_fraction)))
        n_normal = head_dim - n_outlier
        self.n_outlier = n_outlier
        self.n_normal = n_normal
        self.outlier_n_bits = outlier_n_bits
        self.normal_n_bits = normal_n_bits
        self.effective_bits = (n_outlier * outlier_n_bits + n_normal * normal_n_bits) / head_dim

        self.register_buffer('outlier_idx', torch.arange(n_outlier))
        self.register_buffer('normal_idx', torch.arange(n_outlier, head_dim))

        self.pq_outlier = PolarQuant(n_outlier, outlier_n_bits, seed=seed)
        self.pq_normal = PolarQuant(n_normal, normal_n_bits, seed=seed + 500) if n_normal > 0 else None

    def compress(self, x):
        """
        Args:
            x: float tensor [..., head_dim]
        Returns:
            dict with packed indices and norms for both groups
        """
        x_out = x[..., :self.n_outlier]
        x_norm = x[..., self.n_outlier:]

        out_indices, out_norms = self.pq_outlier.compress(x_out)
        packed_out = _pack_indices(out_indices, self.outlier_n_bits)

        if self.pq_normal is not None:
            norm_indices, norm_norms = self.pq_normal.compress(x_norm)
            packed_norm = _pack_indices(norm_indices, self.normal_n_bits)
        else:
            packed_norm = None
            norm_norms = None

        return {
            'packed_outlier': packed_out,
            'outlier_norms': out_norms,
            'packed_normal': packed_norm,
            'normal_norms': norm_norms,
        }

    def decompress(self, compressed):
        """
        Args:
            compressed: dict from compress()
        Returns:
            x_approx: float tensor [..., head_dim]
        """
        out_indices = _unpack_indices(compressed['packed_outlier'],
                                      self.outlier_n_bits, self.n_outlier)
        x_out = self.pq_outlier.decompress(out_indices, compressed['outlier_norms'])

        if self.pq_normal is not None:
            norm_indices = _unpack_indices(compressed['packed_normal'],
                                          self.normal_n_bits, self.n_normal)
            x_norm = self.pq_normal.decompress(norm_indices, compressed['normal_norms'])
        else:
            x_norm = torch.zeros(*x_out.shape[:-1], 0, device=x_out.device, dtype=x_out.dtype)

        return torch.cat([x_out, x_norm], dim=-1)


class TurboQuant2Cache(nn.Module):
    """
    Enhanced KV cache compression with asymmetric K/V, layer-aware boundary
    compression, sparse V dequantization, and optional outlier channels.
    """

    def __init__(self, head_dim, num_heads, layer_idx, num_decoder_layers,
                 key_n_bits=4, value_n_bits=2, enable_qjl=True,
                 qjl_projection_dim=None, min_cache_len=32,
                 boundary_layers=2, boundary_value_n_bits=4,
                 sparsity_threshold=1e-6, enable_sparse_v=True,
                 outlier_config=None):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.enable_qjl = enable_qjl
        self.min_cache_len = min_cache_len
        self.enable_sparse_v = enable_sparse_v
        self.sparsity_threshold = sparsity_threshold

        # Asymmetric: keys always at key_n_bits
        self.key_mse_n_bits = (key_n_bits - 1) if enable_qjl else key_n_bits
        self.key_n_bits = key_n_bits

        # Layer-aware boundary V compression
        is_boundary = (layer_idx < boundary_layers) or (layer_idx >= num_decoder_layers - boundary_layers)
        effective_v_bits = boundary_value_n_bits if is_boundary else value_n_bits
        self.value_n_bits = effective_v_bits
        self.is_boundary = is_boundary

        # Key PolarQuant + optional QJL
        self.key_pq = PolarQuant(head_dim, self.key_mse_n_bits, seed=layer_idx * 2)

        # Value compressor: outlier or standard PolarQuant
        self.use_outlier_v = False
        if outlier_config is not None and outlier_config.get('enabled', False):
            self.use_outlier_v = True
            self.value_compressor = OutlierPolarQuant(
                head_dim,
                outlier_fraction=outlier_config.get('outlier_fraction', 0.25),
                outlier_n_bits=min(effective_v_bits + 1, 4),
                normal_n_bits=max(effective_v_bits - 1, 1),
                seed=layer_idx * 2 + 1,
            )
        else:
            self.value_pq = PolarQuant(head_dim, effective_v_bits, seed=layer_idx * 2 + 1)

        # Key QJL
        self.key_qjl = None
        resolved_qjl_dim = head_dim if qjl_projection_dim is None else int(qjl_projection_dim)
        self.key_qjl_projection_dim = resolved_qjl_dim if enable_qjl else 0
        if enable_qjl:
            self.key_qjl = QJLCorrection(
                input_dim=head_dim,
                projection_dim=resolved_qjl_dim,
                seed=layer_idx * 2 + 1000,
            )

        # Packed byte widths
        self.key_index_packed_bytes = _packed_byte_width(head_dim, self.key_mse_n_bits)
        self.key_qjl_packed_bytes = _packed_byte_width(resolved_qjl_dim, 1) if enable_qjl else 0

        # Cache storage
        self.cached_key_quantized = None
        self.cached_key_norms = None
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_data = None  # packed indices or outlier dict list
        self.cached_value_norms = None
        # For outlier mode, store separate packed tensors
        self.cached_v_outlier_packed = None
        self.cached_v_outlier_norms = None
        self.cached_v_normal_packed = None
        self.cached_v_normal_norms = None

    def _compress_key(self, key):
        indices, norms = self.key_pq.compress(key)
        qjl_signs, qjl_norms = None, None
        if self.key_qjl is not None:
            approx = self.key_pq.decompress(indices, norms)
            residual = key - approx
            qjl_signs, qjl_norms = self.key_qjl.compress(residual)
        return indices, norms, qjl_signs, qjl_norms

    def _compress_value(self, value):
        if self.use_outlier_v:
            return self.value_compressor.compress(value)
        else:
            indices, norms = self.value_pq.compress(value)
            return indices, norms

    def _decompress_key(self, packed_indices, norms, packed_qjl_signs=None, qjl_norms=None):
        indices = _unpack_indices(packed_indices, self.key_pq.n_bits, self.head_dim)
        approx = self.key_pq.decompress(indices, norms)
        if self.key_qjl is not None and packed_qjl_signs is not None:
            qjl_signs = _unpack_signs(packed_qjl_signs, self.key_qjl.projection_dim)
            residual_approx = self.key_qjl.decompress(qjl_signs, qjl_norms)
            approx = approx + residual_approx
        return approx

    def _decompress_value_standard(self, packed_indices, norms):
        indices = _unpack_indices(packed_indices, self.value_pq.n_bits, self.head_dim)
        return self.value_pq.decompress(indices, norms)

    def _decompress_value_outlier(self, packed_outlier, outlier_norms, packed_normal, normal_norms):
        compressed = {
            'packed_outlier': packed_outlier,
            'outlier_norms': outlier_norms,
            'packed_normal': packed_normal,
            'normal_norms': normal_norms,
        }
        return self.value_compressor.decompress(compressed)

    def _cat_or_init(self, cached, new):
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

    def compress_and_cache(self, key, value):
        """
        Compress new key/value vectors and append to cache.

        Args:
            key: float tensor [batch, new_len, num_heads, head_dim]
            value: float tensor [batch, new_len, num_heads, head_dim]
        """
        # Key compression
        k_idx, k_norms, k_signs, k_qjl_norms = self._compress_key(key)
        self.cached_key_quantized = self._cat_or_init(
            self.cached_key_quantized, _pack_indices(k_idx, self.key_mse_n_bits))
        self.cached_key_norms = self._cat_or_init(self.cached_key_norms, k_norms)
        if self.key_qjl is not None:
            self.cached_key_qjl_signs = self._cat_or_init(
                self.cached_key_qjl_signs, _pack_signs(k_signs))
            self.cached_key_qjl_norms = self._cat_or_init(
                self.cached_key_qjl_norms, k_qjl_norms)

        # Value compression
        if self.use_outlier_v:
            v_compressed = self._compress_value(value)
            self.cached_v_outlier_packed = self._cat_or_init(
                self.cached_v_outlier_packed, v_compressed['packed_outlier'])
            self.cached_v_outlier_norms = self._cat_or_init(
                self.cached_v_outlier_norms, v_compressed['outlier_norms'])
            if v_compressed['packed_normal'] is not None:
                self.cached_v_normal_packed = self._cat_or_init(
                    self.cached_v_normal_packed, v_compressed['packed_normal'])
                self.cached_v_normal_norms = self._cat_or_init(
                    self.cached_v_normal_norms, v_compressed['normal_norms'])
        else:
            v_idx, v_norms = self._compress_value(value)
            self.cached_value_data = self._cat_or_init(
                self.cached_value_data, _pack_indices(v_idx, self.value_pq.n_bits))
            self.cached_value_norms = self._cat_or_init(
                self.cached_value_norms, v_norms)

    def get_decompressed(self):
        """Decompress entire cached KV."""
        key = self._decompress_key(
            self.cached_key_quantized, self.cached_key_norms,
            self.cached_key_qjl_signs, self.cached_key_qjl_norms)

        if self.use_outlier_v:
            value = self._decompress_value_outlier(
                self.cached_v_outlier_packed, self.cached_v_outlier_norms,
                self.cached_v_normal_packed, self.cached_v_normal_norms)
        else:
            value = self._decompress_value_standard(
                self.cached_value_data, self.cached_value_norms)
        return key, value

    def get_decompressed_slice(self, start, end):
        """Decompress a slice of the cached KV."""
        key = self._decompress_key(
            self.cached_key_quantized[:, start:end],
            self.cached_key_norms[:, start:end],
            None if self.cached_key_qjl_signs is None else self.cached_key_qjl_signs[:, start:end],
            None if self.cached_key_qjl_norms is None else self.cached_key_qjl_norms[:, start:end])

        if self.use_outlier_v:
            value = self._decompress_value_outlier(
                self.cached_v_outlier_packed[:, start:end],
                self.cached_v_outlier_norms[:, start:end],
                None if self.cached_v_normal_packed is None else self.cached_v_normal_packed[:, start:end],
                None if self.cached_v_normal_norms is None else self.cached_v_normal_norms[:, start:end])
        else:
            value = self._decompress_value_standard(
                self.cached_value_data[:, start:end],
                self.cached_value_norms[:, start:end])
        return key, value

    def _decompress_value_slice(self, start, end):
        """Decompress only the value slice (used in sparse attention)."""
        if self.use_outlier_v:
            return self._decompress_value_outlier(
                self.cached_v_outlier_packed[:, start:end],
                self.cached_v_outlier_norms[:, start:end],
                None if self.cached_v_normal_packed is None else self.cached_v_normal_packed[:, start:end],
                None if self.cached_v_normal_norms is None else self.cached_v_normal_norms[:, start:end])
        else:
            return self._decompress_value_standard(
                self.cached_value_data[:, start:end],
                self.cached_value_norms[:, start:end])

    def sparse_blockwise_attention(self, query, attention_bias=None, block_size=32,
                                   float32_logits=False, dtype=torch.float32):
        """
        Decode-time blockwise attention with sparse V dequantization.

        Decompresses K blocks fully, computes attention logits, then only
        decompresses V for positions where attention weight > sparsity_threshold.

        Args:
            query: [batch, q_len, num_heads, head_dim]
            attention_bias: optional bias tensor
            block_size: KV block size for chunked processing
            float32_logits: whether to cast to float32 for stability
            dtype: output dtype
        Returns:
            output: [batch, q_len, num_heads, head_dim]
        """
        cache_len = self.get_cache_len()
        if cache_len == 0:
            raise ValueError("TurboQuant2 sparse attention requires a non-empty cache.")

        if attention_bias is not None:
            attention_bias = attention_bias.to(query.device)

        query_for_logits = query.float() if float32_logits else query
        # [batch, heads, q_len, head_dim]
        query_for_logits = query_for_logits.permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)

        running_max = None
        running_norm = None
        running_value = None

        for start in range(0, cache_len, block_size):
            end = min(start + block_size, cache_len)
            block_len = end - start

            # Decompress K block (always full)
            key_block = self._decompress_key(
                self.cached_key_quantized[:, start:end],
                self.cached_key_norms[:, start:end],
                None if self.cached_key_qjl_signs is None else self.cached_key_qjl_signs[:, start:end],
                None if self.cached_key_qjl_norms is None else self.cached_key_qjl_norms[:, start:end])

            key_for_logits = key_block.float() if float32_logits else key_block
            # logits: [batch, heads, q_len, block_len]
            logits = torch.einsum('bhqd,bkhd->bhqk', query_for_logits, key_for_logits) * scale

            if attention_bias is not None:
                logits = logits + attention_bias[..., start:end].to(logits.dtype)

            block_max = logits.amax(dim=-1)
            safe_block_max = torch.where(torch.isfinite(block_max), block_max, torch.zeros_like(block_max))
            block_exp = torch.exp(logits - safe_block_max.unsqueeze(-1))
            block_exp = torch.where(torch.isfinite(logits), block_exp, torch.zeros_like(block_exp))

            # Sparse V dequantization: only decompress positions with non-negligible weight
            if self.enable_sparse_v:
                # Compute per-position weight sum across query positions
                # block_exp: [batch, heads, q_len, block_len]
                block_norm_local = block_exp.sum(dim=-1)  # [batch, heads, q_len]
                # Normalize within block to estimate which positions matter
                safe_norm = block_norm_local.clamp_min(1e-12)
                attn_weights_approx = block_exp / safe_norm.unsqueeze(-1)
                # Max weight across query positions for each KV position
                max_weight_per_pos = attn_weights_approx.amax(dim=2)  # [batch, heads, block_len]
                sparse_mask = max_weight_per_pos > self.sparsity_threshold  # [batch, heads, block_len]

                # Check if any positions are active
                if sparse_mask.any():
                    # Decompress full V block then zero out masked positions
                    value_block = self._decompress_value_slice(start, end)
                    # value_block: [batch, block_len, heads, head_dim]
                    # Expand mask: [batch, heads, block_len] -> [batch, block_len, heads, 1]
                    v_mask = sparse_mask.permute(0, 2, 1).unsqueeze(-1)
                    value_block = value_block * v_mask.to(value_block.dtype)
                    # Zero out exp for masked positions to maintain correct softmax
                    exp_mask = sparse_mask.unsqueeze(2)  # [batch, heads, 1, block_len]
                    block_exp = block_exp * exp_mask.to(block_exp.dtype)
                else:
                    # All positions negligible, skip V dequant entirely
                    block_norm = block_exp.sum(dim=-1)
                    if running_max is None:
                        batch_sz, n_heads, q_len, _ = block_exp.shape
                        running_max = safe_block_max
                        running_norm = block_norm
                        running_value = torch.zeros(batch_sz, n_heads, q_len, self.head_dim,
                                                    device=query.device, dtype=block_exp.dtype)
                    else:
                        merged_max = torch.maximum(running_max, safe_block_max)
                        running_scale = torch.exp(running_max - merged_max)
                        block_scale = torch.exp(safe_block_max - merged_max)
                        running_value = running_value * running_scale.unsqueeze(-1)
                        running_norm = running_norm * running_scale + block_norm * block_scale
                        running_max = merged_max
                    continue
            else:
                value_block = self._decompress_value_slice(start, end)

            block_norm = block_exp.sum(dim=-1)
            # block_value: [batch, heads, q_len, head_dim]
            block_value = torch.einsum('bhqk,bkhd->bhqd', block_exp.to(value_block.dtype), value_block)

            if running_max is None:
                running_max = safe_block_max
                running_norm = block_norm
                running_value = block_value
                continue

            merged_max = torch.maximum(running_max, safe_block_max)
            running_scale = torch.exp(running_max - merged_max)
            block_scale = torch.exp(safe_block_max - merged_max)

            running_value = running_value * running_scale.unsqueeze(-1) + block_value * block_scale.unsqueeze(-1)
            running_norm = running_norm * running_scale + block_norm * block_scale
            running_max = merged_max

        if running_norm is None:
            # Edge case: empty cache
            return torch.zeros_like(query)

        running_norm = running_norm.clamp_min(torch.finfo(running_value.dtype).tiny)
        output = (running_value / running_norm.unsqueeze(-1)).permute(0, 2, 1, 3)
        return output.to(dtype)

    def clear(self):
        """Reset cache for new sequence generation."""
        self.cached_key_quantized = None
        self.cached_key_norms = None
        self.cached_key_qjl_signs = None
        self.cached_key_qjl_norms = None
        self.cached_value_data = None
        self.cached_value_norms = None
        self.cached_v_outlier_packed = None
        self.cached_v_outlier_norms = None
        self.cached_v_normal_packed = None
        self.cached_v_normal_norms = None

    def apply_sliding_window(self, window_size):
        """Truncate cache to last window_size entries."""
        if self.cached_key_quantized is None:
            return
        self.cached_key_quantized = self.cached_key_quantized[:, -window_size:]
        self.cached_key_norms = self.cached_key_norms[:, -window_size:]
        if self.key_qjl is not None:
            self.cached_key_qjl_signs = self.cached_key_qjl_signs[:, -window_size:]
            self.cached_key_qjl_norms = self.cached_key_qjl_norms[:, -window_size:]

        if self.use_outlier_v:
            self.cached_v_outlier_packed = self.cached_v_outlier_packed[:, -window_size:]
            self.cached_v_outlier_norms = self.cached_v_outlier_norms[:, -window_size:]
            if self.cached_v_normal_packed is not None:
                self.cached_v_normal_packed = self.cached_v_normal_packed[:, -window_size:]
                self.cached_v_normal_norms = self.cached_v_normal_norms[:, -window_size:]
        else:
            self.cached_value_data = self.cached_value_data[:, -window_size:]
            self.cached_value_norms = self.cached_value_norms[:, -window_size:]
