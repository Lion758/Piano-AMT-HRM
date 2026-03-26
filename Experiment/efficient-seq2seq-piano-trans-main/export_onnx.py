"""
Export the piano AMT encoder to ONNX format.

Usage:
    python export_onnx.py \
        --checkpoint path/to/checkpoint.pt \
        --output encoder.onnx \
        --config-name main_config

The encoder is exported with a dynamic batch axis so it can handle
any batch size at inference time. Sequence length and feature
dimensions are fixed to match the training configuration.
"""

import argparse
import os
import sys
import torch
import onnx

from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir

from model.T5 import Transformer
from config.utils import DictToObject


def load_model(config_dir: str, config_name: str, checkpoint_path: str):
    """Load the Transformer model from config + checkpoint."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name=config_name)

    model = Transformer(config.model)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Remove ignored layers if specified in config
    ignore = getattr(config.model, "checkpoint_ignore_layres", [])
    for key in list(state_dict.keys()):
        if key in ignore:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class EncoderWrapper(torch.nn.Module):
    """Thin wrapper that calls the encoder with an all-ones mask
    and deterministic=True, matching inference-time behaviour."""

    def __init__(self, encoder, dtype):
        super().__init__()
        self.encoder = encoder
        self.dtype = dtype

    def forward(self, encoder_input_tokens: torch.Tensor) -> torch.Tensor:
        # All-ones mask: no padding in fixed-length mel chunks
        ones = torch.ones(
            encoder_input_tokens.shape[:-1],
            device=encoder_input_tokens.device,
        )
        encoder_mask = torch.einsum("bi,bj->bij", ones, ones)
        # Expand for num_heads: [B, 1, T, T]
        encoder_mask = encoder_mask.unsqueeze(1).to(self.dtype)
        return self.encoder(encoder_input_tokens, encoder_mask, deterministic=True)


def export_encoder(
    checkpoint_path: str,
    output_path: str,
    config_dir: str,
    config_name: str,
    opset_version: int = 17,
):
    """Export the encoder to ONNX."""
    print(f"Loading model from: {checkpoint_path}")
    model, config = load_model(config_dir, config_name, checkpoint_path)
    encoder = model.encoder
    dtype = model.config.dtype

    wrapper = EncoderWrapper(encoder, dtype)
    wrapper.eval()

    # Determine input shape from config
    n_frames = config.data.n_frames
    encoder_input_dim = config.model.encoder_input_dim
    print(f"Encoder input shape: [batch, {n_frames}, {encoder_input_dim}]")
    print(f"Encoder output shape: [batch, {n_frames}, {config.model.emb_dim}]")

    # Create dummy input
    dummy_input = torch.randn(1, n_frames, encoder_input_dim)

    # Verify forward pass works
    with torch.no_grad():
        dummy_output = wrapper(dummy_input)
    print(f"Forward pass OK. Output shape: {dummy_output.shape}")

    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["encoder_input"],
        output_names=["encoded"],
        dynamic_axes={
            "encoder_input": {0: "batch_size"},
            "encoded": {0: "batch_size"},
        },
    )

    # Validate the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to: {output_path} ({file_size_mb:.1f} MB)")
    print("ONNX model validation passed.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export encoder to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--output", type=str, default="encoder.onnx", help="Output ONNX file path"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Absolute path to config directory. Defaults to ./config",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="main_config",
        help="Hydra config name (default: main_config)",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)"
    )
    args = parser.parse_args()

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))

    export_encoder(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_dir=config_dir,
        config_name=args.config_name,
        opset_version=args.opset,
    )
