import math

import torch
import torch.nn.functional as F

from model.Attention import Multi_Head_Attention
from model.TurboQuant import (
    PolarQuant,
    QJLCorrection,
    TurboQuantCache,
    _pack_lowbit_tensor,
    _pack_signs,
    _unpack_lowbit_tensor,
    _unpack_signs,
)


def _dense_scaled_attention(query, key, value, bias=None):
    """Reference scaled attention over already-decompressed KV tensors."""
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.0)
    return output.permute(0, 2, 1, 3)


def test_lowbit_pack_unpack_roundtrip_for_indices():
    torch.manual_seed(0)

    for bit_width in (1, 2, 3, 4):
        values = torch.randint(0, 1 << bit_width, (2, 3, 4, 11), dtype=torch.uint8)
        packed = _pack_lowbit_tensor(values, bit_width)
        unpacked = _unpack_lowbit_tensor(packed, bit_width, values.size(-1))

        assert packed.dtype == torch.uint8
        assert packed.shape[-1] == math.ceil(values.size(-1) * bit_width / 8)
        assert torch.equal(unpacked, values)


def test_qjl_sign_pack_unpack_roundtrip():
    torch.manual_seed(1)
    signs = torch.randint(0, 2, (2, 5, 3, 13), dtype=torch.uint8)

    packed = _pack_signs(signs)
    unpacked = _unpack_signs(packed, signs.size(-1))

    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == math.ceil(signs.size(-1) / 8)
    assert torch.equal(unpacked, signs)


def test_turboquant_uses_role_specific_bit_budgets():
    cache_qjl_4 = TurboQuantCache(head_dim=8, num_heads=2, n_bits=4, enable_qjl=True, min_cache_len=0)
    cache_qjl_3 = TurboQuantCache(head_dim=8, num_heads=2, n_bits=3, enable_qjl=True, min_cache_len=0)
    cache_mse_4 = TurboQuantCache(head_dim=8, num_heads=2, n_bits=4, enable_qjl=False, min_cache_len=0)

    assert cache_qjl_4.key_mse_n_bits == 3
    assert cache_qjl_4.value_mse_n_bits == 4
    assert cache_qjl_4.key_pq.n_bits == 3
    assert cache_qjl_4.value_pq.n_bits == 4
    assert cache_qjl_4.key_effective_bits_per_coord == 4.0
    assert cache_qjl_4.get_stats()["key_mse_n_bits"] == 3
    assert cache_qjl_4.get_stats()["value_mse_n_bits"] == 4

    assert cache_qjl_3.key_mse_n_bits == 2
    assert cache_qjl_3.value_mse_n_bits == 3
    assert cache_qjl_3.key_pq.n_bits == 2
    assert cache_qjl_3.value_pq.n_bits == 3

    assert cache_mse_4.key_mse_n_bits == 4
    assert cache_mse_4.value_mse_n_bits == 4
    assert cache_mse_4.key_pq.n_bits == 4
    assert cache_mse_4.value_pq.n_bits == 4
    assert cache_mse_4.key_qjl is None
    assert cache_mse_4.get_stats()["key_effective_bits_per_coord"] == 4.0


def test_packed_cache_matches_expected_key_and_value_reconstruction():
    torch.manual_seed(2)
    cache = TurboQuantCache(head_dim=8, num_heads=2, n_bits=4, enable_qjl=True, min_cache_len=0)
    key = torch.randn(2, 3, 2, 8)
    value = torch.randn(2, 3, 2, 8)

    k_idx, k_norms, k_signs, k_qjl_norms = cache._compress_key(key)
    v_idx, v_norms = cache._compress_value(value)

    key_expected = cache.key_pq.decompress(k_idx, k_norms)
    key_expected = key_expected + cache.key_qjl.decompress(k_signs, k_qjl_norms)
    value_expected = cache.value_pq.decompress(v_idx, v_norms)

    cache.compress_and_cache(key, value)
    key_actual, value_actual = cache.get_decompressed()

    assert cache.cached_key_quantized.dtype == torch.uint8
    assert cache.cached_value_quantized.dtype == torch.uint8
    assert cache.cached_key_qjl_signs.dtype == torch.uint8
    assert not hasattr(cache, "cached_value_qjl_signs")
    assert torch.allclose(key_actual, key_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(value_actual, value_expected, atol=1e-6, rtol=1e-6)


def test_packed_cache_reports_role_split_storage_and_effective_bits():
    torch.manual_seed(3)
    key = torch.randn(1, 2, 2, 8)
    value = torch.randn(1, 2, 2, 8)

    cache = TurboQuantCache(
        head_dim=8,
        num_heads=2,
        n_bits=4,
        enable_qjl=True,
        qjl_projection_dim=4,
        min_cache_len=0,
    )
    cache.compress_and_cache(key, value)
    stats = cache.get_stats()

    assert cache.cached_key_quantized.shape[-1] == math.ceil(cache.head_dim * cache.key_mse_n_bits / 8)
    assert cache.cached_value_quantized.shape[-1] == math.ceil(cache.head_dim * cache.value_mse_n_bits / 8)
    assert cache.cached_key_qjl_signs.shape[-1] == math.ceil(cache.key_qjl_projection_dim / 8)
    assert stats["packed_value_qjl_bytes"] == 0
    assert stats["key_mse_n_bits"] == 3
    assert stats["value_mse_n_bits"] == 4
    assert stats["key_qjl_projection_dim"] == 4
    assert stats["key_effective_bits_per_coord"] == 3.5
    assert stats["packed_value_index_bytes"] > stats["packed_key_index_bytes"]
    assert stats["unpacked_discrete_bytes"] > (
        stats["packed_key_discrete_bytes"] + stats["packed_value_discrete_bytes"]
    )


def test_qjl_projection_dim_controls_output_width_and_scaling():
    torch.manual_seed(4)
    qjl_full = QJLCorrection(input_dim=8, projection_dim=8, seed=11)
    qjl_half = QJLCorrection(input_dim=8, projection_dim=4, seed=11)

    signs_full = torch.ones(2, 8, dtype=torch.uint8)
    signs_half = torch.ones(2, 4, dtype=torch.uint8)
    norms = torch.ones(2, 1)

    expected_full = (
        (2.0 * signs_full.float() - 1.0) @ qjl_full.projection_matrix
    ) * (math.sqrt(math.pi / 2.0) / 8.0)
    expected_half = (
        (2.0 * signs_half.float() - 1.0) @ qjl_half.projection_matrix
    ) * (math.sqrt(math.pi / 2.0) / 4.0)

    actual_full = qjl_full.decompress(signs_full, norms)
    actual_half = qjl_half.decompress(signs_half, norms)

    assert actual_full.shape[-1] == 8
    assert actual_half.shape[-1] == 8
    assert torch.allclose(actual_full, expected_full, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_half, expected_half, atol=1e-6, rtol=1e-6)


def test_blockwise_attention_matches_dense_scaled_attention_with_reduced_qjl_projection():
    torch.manual_seed(5)
    attention = Multi_Head_Attention(num_heads=2, head_dim=8, dropout_rate=0.0)
    cache = TurboQuantCache(
        head_dim=8,
        num_heads=2,
        n_bits=4,
        enable_qjl=True,
        qjl_projection_dim=4,
        min_cache_len=0,
    )
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)
    query = torch.randn(2, 1, 2, 8)

    cache.compress_and_cache(key, value)
    attention.turbo_quant_cache = cache
    decompressed_key, decompressed_value = cache.get_decompressed()

    expected_no_bias = _dense_scaled_attention(query, decompressed_key, decompressed_value)
    actual_no_bias = attention._turbo_quant_blockwise_attention(query, attention_bias=None, block_size=2)
    assert torch.allclose(actual_no_bias, expected_no_bias, atol=1e-5, rtol=1e-5)

    bias = torch.randn(2, 2, 1, decompressed_key.size(1)) * 0.1
    bias[..., 0] = float("-inf")
    expected_with_bias = _dense_scaled_attention(query, decompressed_key, decompressed_value, bias=bias)
    actual_with_bias = attention._turbo_quant_blockwise_attention(query, attention_bias=bias, block_size=2)
    assert torch.allclose(actual_with_bias, expected_with_bias, atol=1e-5, rtol=1e-5)


def test_decode_transition_quantizes_and_slides_window_cleanly():
    torch.manual_seed(6)
    attention = Multi_Head_Attention(
        num_heads=2,
        head_dim=4,
        dropout_rate=0.0,
        turbo_quant_config={
            "enabled": True,
            "n_bits": 4,
            "qjl_projection_dim": 2,
            "enable_qjl": True,
            "min_cache_len": 2,
        },
    )
    attention.initialize_decoder_cache()

    step1 = torch.randn(1, 1, 8)
    out1 = attention(step1, step1, decode=True, sliding_window_size=2)
    assert out1.shape == (1, 1, 8)
    assert torch.isfinite(out1).all()
    assert hasattr(attention, "cached_key")
    assert not attention.turbo_quant_cache.has_cached_values()
    assert attention.turbo_quant_cache.get_stats()["short_prefix_steps"] == 1

    step2 = torch.randn(1, 1, 8)
    out2 = attention(step2, step2, decode=True, sliding_window_size=2)
    stats_after_transition = attention.turbo_quant_cache.get_stats()
    assert out2.shape == (1, 1, 8)
    assert torch.isfinite(out2).all()
    assert not hasattr(attention, "cached_key")
    assert attention.turbo_quant_cache.has_cached_values()
    assert attention.turbo_quant_cache.get_cache_len() == 2
    assert stats_after_transition["quantized_steps"] == 1
    assert stats_after_transition["key_qjl_projection_dim"] == 2
    assert stats_after_transition["packed_value_qjl_bytes"] == 0

    step3 = torch.randn(1, 1, 8)
    out3 = attention(step3, step3, decode=True, sliding_window_size=2)
    stats = attention.turbo_quant_cache.get_stats()

    assert out3.shape == (1, 1, 8)
    assert torch.isfinite(out3).all()
    assert attention.turbo_quant_cache.get_cache_len() == 2
    assert stats["quantized_steps"] == 2
    assert stats["last_window_size"] == 2


def test_haar_corrected_rotation_is_orthogonal_positive_and_deterministic():
    pq_a = PolarQuant(dim=8, n_bits=4, seed=17)
    pq_b = PolarQuant(dim=8, n_bits=4, seed=17)

    rotation = pq_a.rotation_matrix
    identity = torch.eye(rotation.size(0), dtype=rotation.dtype)
    sign, _ = torch.linalg.slogdet(rotation)

    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-5, rtol=1e-5)
    assert sign.item() > 0
    assert torch.allclose(pq_a.rotation_matrix, pq_b.rotation_matrix)
