"""
Hybrid inference: ONNX encoder + PyTorch decoder.

Usage:
    # Run inference on an audio file
    python inference_onnx.py \
        --checkpoint path/to/checkpoint.pt \
        --onnx-encoder encoder_int8.onnx \
        --audio path/to/audio.wav \
        --output output.mid

    # Verify ONNX encoder matches PyTorch encoder
    python inference_onnx.py \
        --checkpoint path/to/checkpoint.pt \
        --onnx-encoder encoder.onnx \
        --verify
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import onnxruntime as ort

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from model.T5 import Transformer
from model.Mask import make_attention_mask
from config.utils import DictToObject
from data.constants import *
from utils import sequence_processing


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pytorch_model(config_dir: str, config_name: str, checkpoint_path: str):
    """Load the full Transformer model from config + checkpoint."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name=config_name)

    model = Transformer(config.model)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ignore = getattr(config.model, "checkpoint_ignore_layres", [])
    for key in list(state_dict.keys()):
        if key in ignore:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


def create_onnx_session(onnx_path: str, use_gpu: bool = True):
    """Create an ONNX Runtime inference session."""
    providers = []
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()
    print(f"ONNX Runtime providers: {active}")
    return sess


def onnx_encode(session: ort.InferenceSession, encoder_input: np.ndarray) -> np.ndarray:
    """Run the ONNX encoder on a numpy array."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: encoder_input})
    return result[0]


def generate_with_onnx_encoder(
    model: Transformer,
    onnx_session: ort.InferenceSession,
    encoder_inputs: torch.Tensor,
    target_seq_length: int = 1024,
    break_on_eos: bool = True,
):
    """Autoregressive generation using ONNX encoder + PyTorch decoder.

    Mirrors Transformer.generate() but replaces self.encode() with the
    ONNX encoder session.
    """
    model.decoder.initialize_decoder_cache()

    batch_size, T, n_mel = encoder_inputs.size()
    gen_tokens = torch.ones([batch_size, target_seq_length]).to(encoder_inputs) * TOKEN_PAD
    eos_flags = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
    curr_frame_index = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)

    # --- ONNX encoder ---
    enc_np = encoder_inputs.cpu().float().numpy()
    encoded_np = onnx_encode(onnx_session, enc_np)
    encoded = torch.from_numpy(encoded_np).to(encoder_inputs.device)

    # Masks (same as Transformer.generate)
    decoder_mask = torch.tril(
        torch.ones((target_seq_length, target_seq_length), dtype=model.config.dtype),
        diagonal=0,
    ).to(encoder_inputs.device)
    decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, model.config.num_heads, -1, -1
    )
    encoder_decoder_mask = torch.ones(
        (batch_size, model.config.num_heads, target_seq_length, T),
        dtype=model.config.dtype,
    ).to(encoder_inputs.device)

    pred_step = 1
    if hasattr(model.config, "predict_onset_by_AttnMap") and model.config.predict_onset_by_AttnMap:
        pred_step = 2

    use_kv_cache = True
    curr_token = torch.ones([batch_size, pred_step]).to(encoder_inputs) * TOKEN_START

    for i in tqdm(range(0, target_seq_length, pred_step), desc="Generating tokens"):
        encoded_i = encoded[:, :T, :]

        if use_kv_cache:
            decoder_input_tokens_i = curr_token
            encoder_decoder_mask_i = encoder_decoder_mask[:, :, i : i + pred_step, :T]
            decoder_mask_i = decoder_mask[:, :, i : i + pred_step, : i + pred_step]
            decoder_positions_i = (
                torch.arange(i, i + pred_step, device=encoder_inputs.device)
                .unsqueeze(0)
                .expand([batch_size, -1])
            )
        else:
            decoder_input_tokens_i = model._shift_right(gen_tokens[:, : i + pred_step], shift_step=1)
            encoder_decoder_mask_i = encoder_decoder_mask[:, :, : i + pred_step, :T]
            decoder_mask_i = decoder_mask[:, :, : i + pred_step, : i + pred_step]
            decoder_positions_i = None

        # Slide window for encoder-decoder attention
        if (
            i % 3 != 0
            and hasattr(model.config, "encoder_decoder_slide_window_size")
            and model.config.encoder_decoder_slide_window_size > 0
        ):
            enc_len, emb_dim = encoded.size(1), encoded.size(2)
            encoded_fold = encoded.view(
                batch_size, -1, model.config.encoder_decoder_slide_window_size, emb_dim
            )
            curr_window_index = torch.floor(
                curr_frame_index // model.config.encoder_decoder_slide_window_size
            ).int()
            batch_indices = torch.arange(batch_size, device=encoded.device)
            encoded_i = encoded_fold[batch_indices, curr_window_index, :, :]
            encoded_i = encoded_i.view(
                batch_size, model.config.encoder_decoder_slide_window_size, emb_dim
            )
            encoder_decoder_mask_i = encoder_decoder_mask_i[
                :, :, :, : model.config.encoder_decoder_slide_window_size
            ]

        # Hierarchical pooling
        encoded_pooling_dict = None
        if (
            hasattr(model.config, "cross_attention_hierarchy_pooling")
            and model.config.cross_attention_hierarchy_pooling
        ):
            encoded_pooling_dict = {}
            for pooling in set(model.config.pooling_sizes):
                encoded_pool = encoded_i
                if pooling > 1:
                    encoded_pool = encoded_pool.reshape(
                        batch_size,
                        encoded_pool.size(1) // pooling,
                        pooling,
                        encoded_pool.size(2),
                    ).mean(dim=2)
                encoded_pooling_dict[pooling] = encoded_pool
            encoded_i = None

        decoder_output_dict = model.decoder(
            encoded_i,
            encoded_pooling_dict=encoded_pooling_dict,
            decoder_input_tokens=decoder_input_tokens_i,
            decoder_positions=decoder_positions_i,
            decoder_mask=decoder_mask_i,
            encoder_decoder_mask=encoder_decoder_mask_i,
            deterministic=True,
            decode=use_kv_cache,
        )

        logits = decoder_output_dict["decoder_outputs"]
        probs = torch.softmax(logits[:, -pred_step:, :], dim=-1)[..., : model.pad_token]

        # Force token type ordering if hybrid attention is enabled
        if (
            hasattr(model.config, "hybrid_global_local_cross_attn")
            and model.config.hybrid_global_local_cross_attn
        ):
            if i % 3 == 0:
                begin, end = Tokenizer.TokenOnset.get_bound()
                probs[:, :, begin:end] += 1
                probs[:, :, model.eos_token] += 1
            elif i % 3 == 1:
                begin, end = Tokenizer.TokenPitch.get_bound()
                probs[:, :, begin:end] += 1
                begin, end = Tokenizer.TokenNoteOff.get_bound()
                probs[:, :, begin:end] += 1
            elif i % 3 == 2:
                begin, end = Tokenizer.TokenVel.get_bound()
                probs[:, :, begin:end] += 1
                probs[:, :, TOKEN_BLANK] += 1

        curr_token = torch.argmax(probs, dim=-1)
        gen_tokens[:, i : i + pred_step] = curr_token

        eos_flags = eos_flags | (curr_token[:, 0] == model.eos_token)
        if break_on_eos and eos_flags.int().sum() == batch_size:
            break

        # Update frame index for slide-window cross-attention
        gen_onsets = curr_token[:, 0]
        frames_per_second = DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH
        eos_flags_int = eos_flags.int()
        for b in range(batch_size):
            onset_i = gen_onsets[b].item()
            if eos_flags_int[b] == 1:
                curr_frame_index[b] = T - 1
                continue
            if Tokenizer.TokenOnset.is_instance(onset_i):
                onset_val = Tokenizer.TokenOnset.get_value(onset_i)
                onset_sec = onset_val / Tokenizer.ONSET_SEC_UP_SAMPLING
                onset_frame_index = int(onset_sec * frames_per_second)
                curr_frame_index[b] = min(T - 1, onset_frame_index)

    return gen_tokens


# ── Verification ──────────────────────────────────────────────────────


def verify_onnx_encoder(
    model: Transformer,
    onnx_session: ort.InferenceSession,
    config,
    n_runs: int = 10,
):
    """Compare ONNX encoder output against PyTorch encoder output."""
    n_frames = config.data.n_frames
    encoder_input_dim = config.model.encoder_input_dim

    dummy = torch.randn(1, n_frames, encoder_input_dim)

    # PyTorch forward
    with torch.no_grad():
        pt_out = model.encode(dummy.to(device), enable_dropout=False).cpu()

    # ONNX forward
    onnx_out = onnx_encode(onnx_session, dummy.numpy())
    onnx_out_t = torch.from_numpy(onnx_out)

    max_diff = (pt_out - onnx_out_t).abs().max().item()
    mean_diff = (pt_out - onnx_out_t).abs().mean().item()
    close = torch.allclose(pt_out, onnx_out_t, atol=1e-4)

    print(f"\n{'='*50}")
    print("Encoder Output Comparison")
    print(f"{'='*50}")
    print(f"  Max  abs diff : {max_diff:.6e}")
    print(f"  Mean abs diff : {mean_diff:.6e}")
    print(f"  allclose(atol=1e-4): {close}")

    # Latency benchmark
    dummy_np = dummy.numpy()
    dummy_dev = dummy.to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.encode(dummy_dev, enable_dropout=False)
        onnx_encode(onnx_session, dummy_np)

    if device == "cuda":
        torch.cuda.synchronize()

    # PyTorch timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model.encode(dummy_dev, enable_dropout=False)
        if device == "cuda":
            torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ONNX timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        onnx_encode(onnx_session, dummy_np)
    onnx_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nLatency ({n_runs} runs, batch=1)")
    print(f"  PyTorch encoder : {pt_ms:.1f} ms")
    print(f"  ONNX encoder    : {onnx_ms:.1f} ms")
    print(f"  Speedup         : {pt_ms / onnx_ms:.2f}x")

    # Model sizes
    onnx_path = onnx_session._model_path if hasattr(onnx_session, "_model_path") else None
    pt_params = sum(p.numel() * p.element_size() for p in model.encoder.parameters())
    print(f"\nModel size")
    print(f"  PyTorch encoder params : {pt_params / 1024 / 1024:.1f} MB")
    if onnx_path and os.path.isfile(onnx_path):
        print(f"  ONNX file              : {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    return close


# ── Audio inference ───────────────────────────────────────────────────


def run_inference(
    model: Transformer,
    onnx_session: ort.InferenceSession,
    config,
    audio_path: str,
    midi_path: str,
):
    """Run full audio → MIDI inference with the hybrid pipeline."""
    from train import MT3Trainer

    # Build a trainer just to reuse its feature extractor
    trainer = MT3Trainer(config)
    trainer.model = model.to(device).eval()

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.to(device).mean(dim=0, keepdim=True)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE).to(device)(wav)

    clip_len = config.data.n_frames * config.data.hop_length
    clip_len_in_second = clip_len / DEFAULT_SAMPLE_RATE

    with torch.no_grad():
        encoder_inputs = trainer.features_extracter.to(device)(wav[:, :-1]).transpose(-1, -2).detach()
        if config.data.amplitude_to_db and config.data.features != "mel":
            encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(encoder_inputs)

    if encoder_inputs.shape[1] % config.data.n_frames != 0:
        pad_len = config.data.n_frames - (encoder_inputs.shape[1] % config.data.n_frames)
        encoder_inputs = F.pad(encoder_inputs, (0, 0, 0, pad_len), value=0.0)

    clip_num = encoder_inputs.shape[1] // config.data.n_frames
    encoder_inputs = encoder_inputs.view(clip_num, config.data.n_frames, -1)

    batch_size = config.training.batch_inference
    num_batches = int(np.ceil(encoder_inputs.shape[0] / batch_size))
    notes_list = []
    max_seq_len = config.data.max_token_length

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for i in tqdm(range(num_batches), desc="Inference (ONNX hybrid)"):
        start = i * batch_size
        end = min((i + 1) * batch_size, encoder_inputs.shape[0])
        encoder_inputs_i = encoder_inputs[start:end].to(device)

        with torch.no_grad():
            decoder_output_tokens = generate_with_onnx_encoder(
                model,
                onnx_session,
                encoder_inputs_i,
                target_seq_length=max_seq_len,
                break_on_eos=True,
            )

        output_eos_tokens_flag = (decoder_output_tokens == sm_tokenizer.EOS).int() * sm_tokenizer.EOS
        output_seq_lens = sequence_processing.get_sequence_lengths(output_eos_tokens_flag, sm_tokenizer.EOS)

        for b in range(encoder_inputs_i.shape[0]):
            output_seq_len = output_seq_lens[b].item()
            if output_seq_len == 0:
                continue
            output_seq_b = decoder_output_tokens[b, :output_seq_len]
            offsets_sec_b = (start + b) * clip_len_in_second
            events_list_b = sm_tokenizer.detokenize(output_seq_b.cpu().numpy(), offsets_sec=offsets_sec_b)
            notes_list_b = sm_tokenizer.midi_events_to_notes(events_list_b)
            notes_list.extend(notes_list_b)

    os.makedirs(os.path.dirname(midi_path) or ".", exist_ok=True)
    sm_tokenizer.save_midi(notes_list, midi_path)
    print(f"MIDI saved to: {midi_path}")


# ── CLI ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import data.symbolic_music_tokenizer as Tokenizer

    parser = argparse.ArgumentParser(description="Hybrid ONNX encoder + PyTorch decoder inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--onnx-encoder", type=str, required=True, help="Path to ONNX encoder model")
    parser.add_argument("--config-dir", type=str, default=None, help="Absolute path to config directory")
    parser.add_argument("--config-name", type=str, default="main_config", help="Hydra config name")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file")
    parser.add_argument("--output", type=str, default=None, help="Path to output MIDI file")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX vs PyTorch encoder output")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution for ONNX")
    args = parser.parse_args()

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))

    print(f"Loading PyTorch model from: {args.checkpoint}")
    model, config = load_pytorch_model(config_dir, args.config_name, args.checkpoint)
    model = model.to(device).eval()

    print(f"Loading ONNX encoder from: {args.onnx_encoder}")
    onnx_session = create_onnx_session(args.onnx_encoder, use_gpu=not args.cpu)

    if args.verify:
        verify_onnx_encoder(model, onnx_session, config)

    if args.audio:
        midi_path = args.output
        if midi_path is None:
            basename = os.path.basename(args.audio)
            midi_path = os.path.splitext(basename)[0] + ".onnx_output.mid"
            midi_path = os.path.join("outputs", midi_path)

        run_inference(model, onnx_session, config, args.audio, midi_path)
