"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
import torch.nn as nn

from model.Layers import *
from model.Mask import *
import math
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except ImportError:  # pragma: no cover - optional dependency
    flash_attn_qkvpacked_func = None
    flash_attn_func = None

try:
    from xformers.ops import memory_efficient_attention
    import xformers.ops.fmha.attn_bias as xformer_attn_bias
except ImportError:  # pragma: no cover - optional dependency
    memory_efficient_attention = None
    xformer_attn_bias = None


# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, head_dim, dtype=torch.float32, dropout_rate=0.0, kernel_init=None, float32_logits=False, window_size=None, is_causal=False, turbo_quant_config=None, turbo_quant_v2_config=None, layer_idx=0):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.projection = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        self.float32_logits = float32_logits
        self.output = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim)

        self.is_causal = is_causal
        self.window_size = window_size

        # TurboQuant KV cache compression
        self.turbo_quant_cache = None
        if turbo_quant_config is not None and turbo_quant_config.get('enabled', False):
            from model.TurboQuant import TurboQuantCache
            self.turbo_quant_cache = TurboQuantCache(
                head_dim=head_dim,
                num_heads=num_heads,
                n_bits=turbo_quant_config.get('n_bits', 4),
                qjl_projection_dim=turbo_quant_config.get('qjl_projection_dim'),
                layer_idx=layer_idx,
                enable_qjl=turbo_quant_config.get('enable_qjl', True),
                min_cache_len=turbo_quant_config.get('min_cache_len', 32),
            )

        # TurboQuant V2 KV cache compression (asymmetric K/V, sparse V, boundary layers)
        self.turbo_quant_v2_cache = None
        if turbo_quant_v2_config is not None and turbo_quant_v2_config.get('enabled', False):
            from model.TurboQuant2 import TurboQuant2Cache
            outlier_cfg = turbo_quant_v2_config.get('outlier_channels', None)
            if isinstance(outlier_cfg, dict) and not outlier_cfg.get('enabled', False):
                outlier_cfg = None
            self.turbo_quant_v2_cache = TurboQuant2Cache(
                head_dim=head_dim,
                num_heads=num_heads,
                layer_idx=layer_idx,
                num_decoder_layers=turbo_quant_v2_config['num_decoder_layers'],
                key_n_bits=turbo_quant_v2_config.get('key_n_bits', 4),
                value_n_bits=turbo_quant_v2_config.get('value_n_bits', 2),
                enable_qjl=turbo_quant_v2_config.get('enable_qjl', True),
                qjl_projection_dim=turbo_quant_v2_config.get('qjl_projection_dim'),
                min_cache_len=turbo_quant_v2_config.get('min_cache_len', 32),
                boundary_layers=turbo_quant_v2_config.get('boundary_layers', 2),
                boundary_value_n_bits=turbo_quant_v2_config.get('boundary_value_n_bits', 4),
                sparsity_threshold=turbo_quant_v2_config.get('sparsity_threshold', 1e-6),
                enable_sparse_v=turbo_quant_v2_config.get('enable_sparse_v', True),
                outlier_config=outlier_cfg,
            )

    def _turbo_quant_blockwise_attention(self, query, attention_bias=None, block_size=32):
        """
        Run decode-time attention directly against the compressed TurboQuant cache by
        decompressing only a small KV block at a time.
        """
        if self.turbo_quant_cache is None or not self.turbo_quant_cache.has_cached_values():
            raise ValueError("TurboQuant blockwise attention requires an initialized compressed cache.")

        cache_len = self.turbo_quant_cache.get_cache_len()
        if cache_len == 0:
            raise ValueError("TurboQuant blockwise attention requires a non-empty cache.")

        if attention_bias is not None:
            attention_bias = attention_bias.to(query.device)

        query_for_logits = query.float() if self.float32_logits else query
        query_for_logits = query_for_logits.permute(0, 2, 1, 3)  # [batch, heads, q_len, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)

        running_max = None
        running_norm = None
        running_value = None

        for start in range(0, cache_len, block_size):
            end = min(start + block_size, cache_len)
            key_block, value_block = self.turbo_quant_cache.get_decompressed_slice(start, end)
            key_for_logits = key_block.float() if self.float32_logits else key_block

            logits = torch.einsum('bhqd,bkhd->bhqk', query_for_logits, key_for_logits) * scale
            if attention_bias is not None:
                logits = logits + attention_bias[..., start:end].to(logits.dtype)

            block_max = logits.amax(dim=-1)
            safe_block_max = torch.where(torch.isfinite(block_max), block_max, torch.zeros_like(block_max))
            block_exp = torch.exp(logits - safe_block_max.unsqueeze(-1))
            block_exp = torch.where(torch.isfinite(logits), block_exp, torch.zeros_like(block_exp))
            block_norm = block_exp.sum(dim=-1)
            block_value = torch.einsum('bhqk,bkhd->bhqd', block_exp.to(value_block.dtype), value_block)

            if running_max is None:
                running_max = block_max
                running_norm = block_norm
                running_value = block_value
                continue

            merged_max = torch.maximum(running_max, block_max)
            running_scale = torch.exp(running_max - merged_max)
            block_scale = torch.exp(block_max - merged_max)

            running_value = running_value * running_scale.unsqueeze(-1) + block_value * block_scale.unsqueeze(-1)
            running_norm = running_norm * running_scale + block_norm * block_scale
            running_max = merged_max

        running_norm = running_norm.clamp_min(torch.finfo(running_value.dtype).tiny)
        output = (running_value / running_norm.unsqueeze(-1)).permute(0, 2, 1, 3)
        return output.to(self.dtype)

    def dot_product_attention(self, query, key, value, bias=None, deterministic=False):
        assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], ('q, k, v batch dims must match.')
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
        assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # Casting logits and softmax computation for float32 for model stability.
        if self.float32_logits:
            query = query.float()
            key = key.float()

        # `attn_weights`: [batch, num_heads, q_length, kv_length]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key)

        # Apply attention bias: masking, dropout, proximity bias, etc.
        if bias is not None:
            attn_weights = attn_weights + bias.to(attn_weights.dtype).to(query.device)

        # Normalize the attention weights across `kv_length` dimension.
        attn_weights = F.softmax(attn_weights, dim=-1).to(self.dtype)

        attn_weights = self.dropout(attn_weights) #edited from original code

        # Take the linear combination of `value`.
        y = torch.einsum('bhqv,bvhd->bqhd', attn_weights, value)
        return y, attn_weights
    
    def flash_attn_sliding_window_attention(self, query, key, value,decode=False):
        assert self.window_size is not None, 'Sliding window attention requires a window size.'
        if flash_attn_func is None:
            raise ImportError("flash_attn is required for flash-attention sliding window attention.")
        if self.is_causal:
            window_size = (self.window_size - (self.window_size%2), -1)
        else:
            w = int(self.window_size // 2)
            window_size = (w, w)
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)
        x = flash_attn_func(query, key, value, dropout_p=self.dropout_rate, window_size=window_size, causal=self.is_causal, deterministic=decode)
        x = x.float()  # Convert back to float32
        return x
        
    def xformers_sliding_window_attention(self, query, key, value, dropout_p=0.0):
        assert self.window_size is not None, 'Sliding window attention requires a window size.'
        if memory_efficient_attention is None or xformer_attn_bias is None:
            raise ImportError("xformers is required for xFormers sliding window attention.")
        
        window_left = window_right = self.window_size // 2
        if self.is_causal:
            window_left = self.window_size
            window_right = 0
        bias = xformer_attn_bias.LocalAttentionFromBottomRightMask(window_left=window_left, window_right=window_right, dtype=query.dtype)

        x = memory_efficient_attention(
            query, # (batch_size, seq_len, num_heads, head_dim)
            key,
            value,
            attn_bias=bias,
            p=dropout_p,
        )
        
        return x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        
        
    def initialize_decoder_cache(self):
        """Initializes the cache for autoregressive decoding."""
        self._clear_raw_cache()
        if self.turbo_quant_cache is not None:
            self.turbo_quant_cache.clear()
        if self.turbo_quant_v2_cache is not None:
            self.turbo_quant_v2_cache.clear()

    def _clear_raw_cache(self):
        if hasattr(self, 'cached_key'):
            delattr(self, 'cached_key')
        if hasattr(self, 'cached_value'):
            delattr(self, 'cached_value')
        if hasattr(self, 'cache_index'):
            delattr(self, 'cache_index')

    def _append_to_raw_cache(self, key, value, sliding_window_size=None):
        if hasattr(self, 'cached_key'):
            cached_key = getattr(self, 'cached_key')
            cached_value = getattr(self, 'cached_value')
            cached_index = getattr(self, 'cache_index')
            key = torch.concat([cached_key, key], dim=1)
            value = torch.concat([cached_value, value], dim=1)
        else:
            cached_index = 0

        if sliding_window_size is not None:
            key = key[:, -sliding_window_size:, :, :]
            value = value[:, -sliding_window_size:, :, :]

        setattr(self, 'cached_key', key)
        setattr(self, 'cached_value', value)
        setattr(self, 'cache_index', cached_index + 1)
        return key, value

    def _get_decode_cache_window(self, sliding_window_size):
        if sliding_window_size is not None:
            return sliding_window_size
        if self.is_causal and self.window_size is not None:
            return self.window_size
        return None

    def forward(self, inputs_q, inputs_kv, mask=None, bias=None, decode=False, deterministic=False, return_attn_weights=False, sliding_window_size=None):
        #In Original MT3, they initialize the parameter with query_init, using customized Dense Layer
        query = self.projection(inputs_q).view(inputs_q.size(0), inputs_q.size(1), self.num_heads, self.head_dim)
        key = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)
        value = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)
        use_turbo_quant_blockwise = False
        use_turbo_quant_v2_blockwise = False

        if decode:
            batch, length, num_heads, head_dim,  = key.size()
            expected_shape = (batch, 1, num_heads, head_dim)
            if expected_shape != query.size():
                raise ValueError('Autoregressive cache shape error, '
                                'expected query shape %s instead got %s.' %
                                (expected_shape, query.size()))

            cache_window_size = self._get_decode_cache_window(sliding_window_size)
            if self.turbo_quant_v2_cache is not None:
                if self.turbo_quant_v2_cache.has_cached_values():
                    self.turbo_quant_v2_cache.compress_and_cache(key, value)
                    if cache_window_size is not None:
                        self.turbo_quant_v2_cache.apply_sliding_window(cache_window_size)
                    use_turbo_quant_v2_blockwise = not return_attn_weights
                    if not use_turbo_quant_v2_blockwise:
                        key, value = self.turbo_quant_v2_cache.get_decompressed()
                    self.turbo_quant_v2_cache.note_quantized_step(self.turbo_quant_v2_cache.get_cache_len(), cache_window_size)
                else:
                    key, value = self._append_to_raw_cache(key, value, sliding_window_size=cache_window_size)
                    raw_cache_len = key.size(1)
                    if self.turbo_quant_v2_cache.should_quantize(raw_cache_len):
                        self.turbo_quant_v2_cache.clear()
                        self.turbo_quant_v2_cache.compress_and_cache(key, value)
                        if cache_window_size is not None:
                            self.turbo_quant_v2_cache.apply_sliding_window(cache_window_size)
                        self._clear_raw_cache()
                        use_turbo_quant_v2_blockwise = not return_attn_weights
                        if not use_turbo_quant_v2_blockwise:
                            key, value = self.turbo_quant_v2_cache.get_decompressed()
                        self.turbo_quant_v2_cache.note_quantized_step(self.turbo_quant_v2_cache.get_cache_len(), cache_window_size)
                    else:
                        self.turbo_quant_v2_cache.note_short_prefix_step(raw_cache_len, cache_window_size)
            elif self.turbo_quant_cache is not None:
                if self.turbo_quant_cache.has_cached_values():
                    self.turbo_quant_cache.compress_and_cache(key, value)
                    if cache_window_size is not None:
                        self.turbo_quant_cache.apply_sliding_window(cache_window_size)
                    use_turbo_quant_blockwise = not return_attn_weights
                    if not use_turbo_quant_blockwise:
                        key, value = self.turbo_quant_cache.get_decompressed()
                    self.turbo_quant_cache.note_quantized_step(self.turbo_quant_cache.get_cache_len(), cache_window_size)
                else:
                    key, value = self._append_to_raw_cache(key, value, sliding_window_size=cache_window_size)
                    raw_cache_len = key.size(1)
                    if self.turbo_quant_cache.should_quantize(raw_cache_len):
                        self.turbo_quant_cache.clear()
                        self.turbo_quant_cache.compress_and_cache(key, value)
                        if cache_window_size is not None:
                            self.turbo_quant_cache.apply_sliding_window(cache_window_size)
                        self._clear_raw_cache()
                        use_turbo_quant_blockwise = not return_attn_weights
                        if not use_turbo_quant_blockwise:
                            key, value = self.turbo_quant_cache.get_decompressed()
                        self.turbo_quant_cache.note_quantized_step(self.turbo_quant_cache.get_cache_len(), cache_window_size)
                    else:
                        self.turbo_quant_cache.note_short_prefix_step(raw_cache_len, cache_window_size)
            else:
                # Original uncompressed cache path
                key, value = self._append_to_raw_cache(key, value, sliding_window_size=cache_window_size)

        if mask is not None:
            attention_bias = torch.where(mask > 0,
                                        torch.zeros_like(mask).to(self.dtype),
                                         float("-inf") * torch.ones_like(mask).to(self.dtype))
        else:
            attention_bias = None

        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)
        
        if return_attn_weights:
            x, attn_weights = self.dot_product_attention(
                    query,
                    key,
                    value,
                    bias=attention_bias,
                    deterministic=deterministic)
        else:
            dropout_p = self.dropout_rate if not decode else 0.0
            if use_turbo_quant_v2_blockwise:
                x = self.turbo_quant_v2_cache.sparse_blockwise_attention(
                    query, attention_bias=attention_bias,
                    float32_logits=self.float32_logits, dtype=self.dtype)
            elif use_turbo_quant_blockwise:
                x = self._turbo_quant_blockwise_attention(query, attention_bias=attention_bias)
            elif self.window_size is None:
                # Faster implementation using PyTorch's built-in attention
                query = query.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)

                x = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=dropout_p)
                x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
            else:
                x = self.flash_attn_sliding_window_attention(query, key, value, decode=decode)
                # x = self.xformers_sliding_window_attention(query, key, value, dropout_p=dropout_p)
        
        out = self.output(x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3)))
        
        if return_attn_weights:
            return out, None
        
        return out

class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, q, kv, mask = None, deterministic = None):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = kv.shape
        assert q.size()[1] == seq_len
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(kv).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(kv).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(q).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
