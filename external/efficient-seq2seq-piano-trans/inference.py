import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from data.constants import *
from train import MT3Trainer
from utils import sequence_processing


device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = None


def load_config(config_path: str | None, config_name: str, overrides: list[str]) -> OmegaConf:
    if config_path is not None:
        config_file = Path(config_path).expanduser().resolve()
        config_dir = str(config_file.parent)
        config_stem = config_file.stem
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name=config_stem, overrides=overrides)
    else:
        config_dir = str((Path(__file__).resolve().parent / "config").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
    return config


def run_inference(config: OmegaConf):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_path = config.model.checkpoint_path

    global trainer
    if trainer is None:
        trainer = MT3Trainer(config)
        print("Loading model from checkpoint:", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        for key in list(state_dict.keys()):
            if key in config.model.checkpoint_ignore_layres:
                print(f"Removing key {key} from state_dict")
                del state_dict[key]

        trainer.model.load_state_dict(state_dict, strict=False)
        trainer.model = trainer.model.to(device).eval()

    audio_path = config.audio_path
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.to(device).mean(dim=0, keepdim=True)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE).to(device)(wav)

    clip_len = config.data.n_frames * config.data.hop_length
    clip_len_in_second = clip_len / DEFAULT_SAMPLE_RATE

    with torch.no_grad():
        encoder_inputs = trainer.features_extracter.to(device)(wav[:, :-1]).transpose(-1, -2)
        encoder_inputs = encoder_inputs.detach()
        if trainer.config.data.amplitude_to_db and trainer.config.data.features != "mel":
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
    if torch.cuda.is_available():
        torch.cuda.synchronize("cuda")
        torch.cuda.empty_cache()
    gc.collect()

    for i in tqdm(range(num_batches), desc="Inference"):
        start = i * batch_size
        end = min((i + 1) * batch_size, encoder_inputs.shape[0])
        encoder_inputs_i = encoder_inputs[start:end].to(device)

        with torch.no_grad():
            decoder_output_tokens = trainer.model.generate(
                encoder_inputs_i,
                target_seq_length=max_seq_len,
                berak_on_eos=True,
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

    midi_path = getattr(config, "midi_path", None)
    if not midi_path:
        basename = os.path.basename(audio_path)
        midi_path = os.path.join("outputs", os.path.splitext(basename)[0] + ".output.mid")
    os.makedirs(os.path.dirname(midi_path), exist_ok=True)
    sm_tokenizer.save_midi(notes_list, midi_path)
    print("Saved MIDI to:", midi_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with either the standard config tree or an explicit Hydra YAML config path.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to a Hydra config file such as root_extras/Config_Files/experiment_T5_V0.yaml or a saved experiment_config.yaml")
    parser.add_argument("--config-name", type=str, default="main_config", help="Config name under ./config when --config-path is not used")
    parser.add_argument("overrides", nargs="*", help="Hydra/OmegaConf dotlist overrides such as model.checkpoint_path=... audio_path=...")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path, args.config_name, args.overrides)
    run_inference(config)
