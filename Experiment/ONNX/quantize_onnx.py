"""
Apply ONNX Runtime dynamic INT8 quantization to the exported encoder.

Usage:
    python quantize_onnx.py \
        --input encoder.onnx \
        --output encoder_int8.onnx

No calibration data is needed — weights are quantized statically to INT8
and activations are quantized dynamically at runtime.
"""

import argparse
import os

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_encoder(
    input_path: str,
    output_path: str,
    weight_type: str = "int8",
    per_channel: bool = True,
):
    """Quantize an ONNX encoder model to INT8."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")

    wt = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8

    print(f"Quantizing: {input_path}")
    print(f"  Weight type : {weight_type}")
    print(f"  Per-channel : {per_channel}")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=wt,
        per_channel=per_channel,
    )

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    quant_mb = os.path.getsize(output_path) / (1024 * 1024)
    ratio = orig_mb / quant_mb if quant_mb > 0 else float("inf")

    print(f"Original  : {orig_mb:.1f} MB")
    print(f"Quantized : {quant_mb:.1f} MB")
    print(f"Compression: {ratio:.2f}x")
    print(f"Saved to  : {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize ONNX encoder to INT8"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="encoder.onnx",
        help="Path to the FP32 ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="encoder_int8.onnx",
        help="Output path for quantized model",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        default="int8",
        choices=["int8", "uint8"],
        help="Quantized weight type (default: int8)",
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use per-tensor quantization instead of per-channel",
    )
    args = parser.parse_args()

    quantize_encoder(
        input_path=args.input,
        output_path=args.output,
        weight_type=args.weight_type,
        per_channel=not args.per_tensor,
    )
