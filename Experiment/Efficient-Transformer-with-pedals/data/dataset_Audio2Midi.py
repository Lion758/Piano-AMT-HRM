
from pathlib import Path
import math
import os
import sys


work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)
import torch
from torch.utils.data import IterableDataset, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as Functional
import librosa
import librosa.display
import music21
import torchaudio

import hashlib

import numpy as np 
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import hydra
import random
import h5py
import matplotlib.pyplot as plt
from glob import glob
import json
import pandas as pd

from data.constants import *

from data.symbolic_music_tokenizer import SymbolicMusicTokenizer, TokenOnset, ONSET_SEC_UP_SAMPLING
import data.symbolic_music_tokenizer as Tokenizer
from collections import defaultdict


def _build_waveform_augmenter(config):
    """Construct an audiomentations.Compose from config.data.augmentation.waveform.

    Returns None if augmentation is disabled, audiomentations is unavailable, or
    no enabled stages produce a valid transform (e.g. reverb requested but no
    IRs present on disk). Safe to call from any subset; callers gate on subset.
    """
    aug_cfg = config.data.get("augmentation", None)
    if aug_cfg is None or not aug_cfg.get("enabled", False):
        return None

    wave_cfg = aug_cfg.get("waveform", None)
    if wave_cfg is None:
        return None

    try:
        import audiomentations
    except ImportError:
        print("[augmentation] audiomentations not installed; skipping waveform augmentations.")
        return None

    transforms = []

    ps = wave_cfg.get("pitch_shift", None)
    if ps is not None and ps.get("enabled", False):
        transforms.append(audiomentations.PitchShift(
            min_semitones=float(ps.get("min_semitones", -0.1)),
            max_semitones=float(ps.get("max_semitones", 0.1)),
            p=float(ps.get("p", 0.8)),
        ))

    rv = wave_cfg.get("reverb", None)
    if rv is not None and rv.get("enabled", False):
        ir_dir = rv.get("ir_dir", None)
        if ir_dir is not None:
            ir_path = ir_dir
            if not os.path.isabs(ir_path):
                ir_path = os.path.join(work_dir, ir_path)
            ir_files = []
            if os.path.isdir(ir_path):
                ir_files = sorted(glob(os.path.join(ir_path, "*.wav")))
            if len(ir_files) > 0:
                transforms.append(audiomentations.ApplyImpulseResponse(
                    ir_path=ir_path,
                    p=float(rv.get("p", 0.5)),
                ))
            else:
                print(f"[augmentation] reverb enabled but no .wav IRs found in {ir_path}; skipping.")

    if len(transforms) == 0:
        return None
    return audiomentations.Compose(transforms)



# Hash string to int 0~9.
def stable_hash_to_digit(s: str) -> int:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h, 16) % 10


PEDAL_BOUNDARY_KERNEL = ((0, 1.0), (-1, 0.5), (1, 0.5), (-2, 0.25), (2, 0.25))


def _config_get(config, key, default):
    if config is None:
        return default
    try:
        if key in config:
            return config[key]
    except TypeError:
        pass
    return getattr(config, key, default)


def build_pedal_boundary_kernel(config=None):
    data_config = _config_get(config, "data", config)
    kernel_type = str(_config_get(data_config, "pedal_boundary_kernel_type", "triangular")).lower()

    if kernel_type == "triangular":
        return PEDAL_BOUNDARY_KERNEL

    if kernel_type == "gaussian":
        radius = int(_config_get(data_config, "pedal_boundary_kernel_radius", 5))
        sigma_frames = float(_config_get(data_config, "pedal_boundary_kernel_sigma_frames", 3.0))
        if radius < 0:
            raise ValueError(f"pedal_boundary_kernel_radius must be >= 0, got {radius}.")
        if sigma_frames <= 0.0:
            raise ValueError(
                f"pedal_boundary_kernel_sigma_frames must be > 0, got {sigma_frames}."
            )
        return tuple(
            (
                offset,
                float(math.exp(-0.5 * (offset / sigma_frames) ** 2)),
            )
            for offset in range(-radius, radius + 1)
        )

    raise ValueError(
        f"Unsupported pedal_boundary_kernel_type {kernel_type!r}; "
        "expected 'triangular' or 'gaussian'."
    )


class SingleWavDataset(Dataset):
    def __init__(self,config, dataset_dir:str, dataset_index:int, audio_idx, midi_path, audio_h5_path, random_clip = True, augmenter=None) -> None:
        self.config = config
        self.pedal_boundary_kernel = build_pedal_boundary_kernel(config)
        self.dataset_dir = dataset_dir
        self.dataset_index = dataset_index
        self.dataset_sequence_type = config.data.dataset_sequence_types[dataset_index]
        self.dataset_name = os.path.split(dataset_dir)[-1]
        self.midi_path = midi_path
        self.data_loaded = False
        n_frames = config.data.n_frames
        max_token_length = config.data.max_token_length
        hop_length = config.data.hop_length


        frames_per_second=DEFAULT_SAMPLE_RATE / hop_length
        
        # load cache
        # root_dir = config.data.dataset_dir
        # token_cache_path = os.path.join(root_dir, "cache", os.path.relpath(midi_path, root_dir) + ".pth")
        # os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
        # if os.path.exists(token_cache_path):
        #     data_dict = torch.load(token_cache_path)
        #     tokens_ary = data_dict["tokens_ary"]
        #     seconds_ary = data_dict["seconds_ary"]
        # else:
        #     tokens_ary, seconds_ary = cp_tokenizer.tokenize_midi(midi_path)
        #     data_dict = {
        #         "tokens_ary": tokens_ary,
        #         "seconds_ary": seconds_ary
        #     }
        #     torch.save(data_dict, token_cache_path)
        
        # self.tokens_ary = torch.tensor(tokens_ary)
        # self.seconds_ary = torch.tensor(seconds_ary)
        
        # Lazy Loading
        self.tokens_ary = None
        self.seconds_ary = None
        self.data_loaded = False
        

        ###########
        # audio
        
        self.audio_idx = str(audio_idx)
        self.audio_h5 = h5py.File(audio_h5_path)
        duration = self.audio_h5[self.audio_idx]["duration"][()]
        assert duration <= 3000 # 1600 # <= 10 min
        # Check audio name and midi name.
        audio_path = self.audio_h5[self.audio_idx]["path"][()].decode()
        self.audio_name = os.path.split(audio_path)[1].replace(".flac", "")
        # assert midi_name == audio_name
        self.audio_size = self.audio_h5[self.audio_idx]["audio"].shape[0]
        
        self.total_frames = np.ceil( self.audio_size / hop_length ).astype(int)
        # self.total_frames = int(duration * frames_per_second)
        self.length = np.ceil( self.audio_size / (hop_length*n_frames) ).astype(int)
        # assert self.total_frames >= n_frames

        self.n_frames = n_frames
        self.hop_length = hop_length
        self.max_token_length = max_token_length

        self.random_clip = random_clip
        self.random = np.random.RandomState() #Don't use constand seed here!!! (Overfitting problems).
        self.augmenter = augmenter

    def __len__(self):
        # if self.random_clip == True:
        #     return 1 # self.length
        # return 100 # 
        return self.length
        
    def __load_cache(self):
        # load cache
        
        if True:
            tsv_path = os.path.splitext(self.midi_path)[0] + ".midi-notes.tsv"
            source_dataframe_midi = pd.read_csv(tsv_path, sep="\t")
            dataframe_midi = sm_tokenizer.notes_to_midi_events(
                source_dataframe_midi,
                use_truth_offsets=self.config.data.get("use_truth_offsets", False),
                emit_pedal_tokens=self.config.data.get("emit_pedal_tokens", True),
            )
        else:
            source_dataframe_midi = sm_tokenizer.midi_to_dataframe(self.midi_path)
            dataframe_midi = sm_tokenizer.notes_to_midi_events(
                source_dataframe_midi,
                use_truth_offsets=self.config.data.get("use_truth_offsets", False),
                emit_pedal_tokens=self.config.data.get("emit_pedal_tokens", True),
            )

        self.dataframe_midi = dataframe_midi.sort_values(
            by=["onset_sec", "type_id", "pitch"],
            kind="mergesort",
        ).reset_index(drop=True)
        self.dataframe_pedal_events = self._get_pedal_rows(source_dataframe_midi)
        self._prepare_pedal_cache()
            
        self.data_loaded = True

    def _get_pedal_rows(self, pedal_source_df=None):
        if pedal_source_df is None:
            pedal_source_df = getattr(self, "dataframe_pedal_events", None)
            if pedal_source_df is None:
                pedal_source_df = self.dataframe_midi

        if pedal_source_df is None or "type" not in pedal_source_df.columns:
            return pd.DataFrame(columns=["type", "onset_sec", "pitch", "velocity"])

        pedal_rows = pedal_source_df[
            pedal_source_df["type"].isin(["PedalOn", "PedalOff"])
        ].copy()
        if len(pedal_rows) == 0:
            return pedal_rows

        # Same-time repedaling must close the previous span before opening the next one.
        pedal_rows["_pedal_sort"] = np.where(pedal_rows["type"] == "PedalOff", 0, 1)
        pedal_rows = pedal_rows.sort_values(
            by=["onset_sec", "_pedal_sort"],
            kind="mergesort",
        ).drop(columns=["_pedal_sort"]).reset_index(drop=True)
        return pedal_rows

    def _prepare_pedal_cache(self):
        pedal_rows = self._get_pedal_rows()
        self.dataframe_pedal_events = pedal_rows
        if len(pedal_rows) == 0:
            self.pedal_event_times = np.empty(0, dtype=np.float64)
            self.pedal_event_states = np.empty(0, dtype=np.int8)
            return

        self.pedal_event_times = pedal_rows["onset_sec"].to_numpy(dtype=np.float64, copy=True)
        pedal_types = pedal_rows["type"].to_numpy()
        self.pedal_event_states = np.where(pedal_types == "PedalOn", 1, 0).astype(np.int8, copy=False)

    def _build_pedal_targets_slow(self, begin_sec, end_sec, second_per_frame):
        pedal_rows = self._get_pedal_rows()
        pedal_boundary_kernel = getattr(self, "pedal_boundary_kernel", PEDAL_BOUNDARY_KERNEL)

        pedal_frame_target = torch.zeros(self.n_frames, dtype=torch.float32)
        pedal_onset_target = torch.zeros(self.n_frames, dtype=torch.float32)
        pedal_offset_target = torch.zeros(self.n_frames, dtype=torch.float32)

        state = 0
        for _, ev in pedal_rows.iterrows():
            if ev["onset_sec"] >= begin_sec:
                break
            state = 1 if ev["type"] == "PedalOn" else 0

        events_in_clip = pedal_rows[
            (pedal_rows["onset_sec"] >= begin_sec) & (pedal_rows["onset_sec"] < end_sec)
        ].to_dict("records")

        cursor_frame = 0
        for ev in events_in_clip:
            ev_frame = int(round((ev["onset_sec"] - begin_sec) / second_per_frame))
            ev_frame = max(0, min(self.n_frames, ev_frame))
            pedal_frame_target[cursor_frame:ev_frame] = state
            new_state = 1 if ev["type"] == "PedalOn" else 0
            target_arr = pedal_onset_target if new_state == 1 else pedal_offset_target
            for offset, val in pedal_boundary_kernel:
                f = ev_frame + offset
                if 0 <= f < self.n_frames:
                    if val > target_arr[f].item():
                        target_arr[f] = val
            state = new_state
            cursor_frame = ev_frame
        pedal_frame_target[cursor_frame:] = state

        return pedal_frame_target, pedal_onset_target, pedal_offset_target

    def _build_pedal_targets_fast(self, begin_sec, end_sec, second_per_frame):
        if not hasattr(self, "pedal_event_times") or not hasattr(self, "pedal_event_states"):
            self._prepare_pedal_cache()
        pedal_boundary_kernel = getattr(self, "pedal_boundary_kernel", PEDAL_BOUNDARY_KERNEL)

        pedal_frame_target = np.zeros(self.n_frames, dtype=np.float32)
        pedal_onset_target = np.zeros(self.n_frames, dtype=np.float32)
        pedal_offset_target = np.zeros(self.n_frames, dtype=np.float32)

        event_times = self.pedal_event_times
        event_states = self.pedal_event_states
        if event_times.size == 0:
            return (
                torch.from_numpy(pedal_frame_target),
                torch.from_numpy(pedal_onset_target),
                torch.from_numpy(pedal_offset_target),
            )

        state_index = np.searchsorted(event_times, begin_sec, side="left") - 1
        state = int(event_states[state_index]) if state_index >= 0 else 0

        clip_begin_index = np.searchsorted(event_times, begin_sec, side="left")
        clip_end_index = np.searchsorted(event_times, end_sec, side="left")
        clip_times = event_times[clip_begin_index:clip_end_index]
        clip_states = event_states[clip_begin_index:clip_end_index]

        if clip_times.size == 0:
            pedal_frame_target[:] = state
            return (
                torch.from_numpy(pedal_frame_target),
                torch.from_numpy(pedal_onset_target),
                torch.from_numpy(pedal_offset_target),
            )

        event_frames = np.rint((clip_times - begin_sec) / second_per_frame).astype(np.int64)
        event_frames = np.clip(event_frames, 0, self.n_frames)

        cursor_frame = 0
        for ev_frame, new_state in zip(event_frames, clip_states):
            pedal_frame_target[cursor_frame:ev_frame] = state
            state = int(new_state)
            cursor_frame = int(ev_frame)
        pedal_frame_target[cursor_frame:] = state

        on_frames = event_frames[clip_states == 1]
        off_frames = event_frames[clip_states == 0]
        for target_arr, frames in (
            (pedal_onset_target, on_frames),
            (pedal_offset_target, off_frames),
        ):
            if frames.size == 0:
                continue
            for offset, val in pedal_boundary_kernel:
                kernel_frames = frames + offset
                valid_frames = kernel_frames[(0 <= kernel_frames) & (kernel_frames < self.n_frames)]
                if valid_frames.size > 0:
                    np.maximum.at(target_arr, valid_frames, val)

        return (
            torch.from_numpy(pedal_frame_target),
            torch.from_numpy(pedal_onset_target),
            torch.from_numpy(pedal_offset_target),
        )
        
    def __getitem__(self, index, begin=None, end=None):
        if self.data_loaded == False:
            self.__load_cache()
            
        row = {}
        
        if (begin is None) and (end is None):
            if self.random_clip == False:
                begin = index * self.n_frames # begin frame idx
                end = begin + self.n_frames # end frame idx
            else:
                # begin = index * self.n_frames + self.random.randint(-self.n_frames//2, self.n_frames//2)
                # begin = np.clip(0, self.total_frames - self.n_frames//4)
                random = np.random.RandomState()
                begin = random.randint(0, self.total_frames - self.n_frames//4)
                
                end = begin + self.n_frames
        else:
            assert self.random_clip == False
            assert (begin is not None) and (end is not None)
            
        audio_begin = begin * self.hop_length
        audio_end = end * self.hop_length
        # assert end <= self.total_frames
        # assert audio_end <= self.audio_size

        # audio = torch.tensor(self.data_dict["wav"][audio_begin:audio_end])
        audio = torch.tensor(self.audio_h5[self.audio_idx]["audio"][audio_begin:audio_end])

        # Apply waveform augmentations (training only — gated by Audio2Midi_Dataset).
        # Augmentations here are label-preserving for note-onset prediction:
        # ±10-cent pitch shift does not cross a semitone boundary, and convolution
        # reverb does not move onset times beyond the mir_eval 50ms tolerance.
        if self.augmenter is not None and len(audio) > 0:
            audio_np = audio.numpy().astype(np.float32, copy=False)
            audio_np = self.augmenter(samples=audio_np, sample_rate=DEFAULT_SAMPLE_RATE)
            peak = float(np.max(np.abs(audio_np))) if audio_np.size > 0 else 0.0
            if peak > 1.0:
                audio_np = audio_np / peak
            audio = torch.from_numpy(audio_np)

        # convert time resolution from frame to OUTPUT_TIME_STEP_PER_SECOND
        second_per_frame = self.config.data.hop_length/DEFAULT_SAMPLE_RATE
        begin_sec = begin * second_per_frame # * OUTPUT_TIME_STEP_PER_SECOND
        end_sec = end * second_per_frame # * OUTPUT_TIME_STEP_PER_SECOND

        # Auxiliary per-frame sustain-pedal targets, aligned 1:1 with the encoder time axis
        # (length = self.n_frames). Three signals:
        #   - pedal_frame_target: binary state (0=up, 1=down) at each frame center
        #   - pedal_onset_target: configurable soft kernel around each PedalOn frame
        #   - pedal_offset_target: configurable soft kernel around each PedalOff frame
        # The encoder gets direct supervision for pedal cues; the existing decoder pedal
        # tokens are unchanged.
        pedal_frame_target, pedal_onset_target, pedal_offset_target = self._build_pedal_targets_fast(
            begin_sec,
            end_sec,
            second_per_frame,
        )

        pad_len = self.n_frames*self.hop_length - len(audio)
        if pad_len > 0:
            audio = Functional.pad(audio, [0, pad_len])
    
        # max_token_len = self.n_frames // self.downsample_rate
        max_token_len = self.config.data.max_token_length # * cp_tokenizer.compound_word_size
        # assert max_token_len < TOKEN_END
        decoder_targets = torch.ones([max_token_len], dtype=torch.long) * TOKEN_PAD
        decoder_targets_frame_index = torch.ones([max_token_len], dtype=torch.long) * (self.n_frames - 1) # max frame index
        encoder_decoder_mask = torch.zeros([max_token_len, self.n_frames], dtype=torch.long)
        seq_len = 0
        sel_tokens = torch.tensor([], dtype=torch.long)
        
        seconds = []
        
        # Add EOS
        cp_eos = torch.ones([sm_tokenizer.compound_word_size], dtype=sel_tokens.dtype) * sm_tokenizer.PAD
        # cp_eos[Tokenizer.TokenFamily.cp_index] = cp_tokenizer.EOS
        cp_eos[0] = sm_tokenizer.EOS #Tokenizer.TokenPitch.cp_index
        
        # MIDI Sequence
        df_midi = self.dataframe_midi[(self.dataframe_midi["onset_sec"] >= begin_sec) & (self.dataframe_midi["onset_sec"] <= end_sec)].copy(deep=True)
        if len(df_midi) > 0:
            # make a offset for onset_sec
            df_midi["onset_sec"] = df_midi["onset_sec"] - begin_sec # Fix offset when using clip for training.
            sel_cp_tokens, seconds = sm_tokenizer.tokenize_dataframe(df_midi, sequence_type="performance")
            sel_cp_tokens = torch.tensor(sel_cp_tokens, dtype=torch.long)
            sel_tokens = sel_cp_tokens.flatten()
            
        # Eos
        sel_tokens = torch.concat([sel_tokens, cp_eos], dim=0)
        seconds.extend([second_per_frame * self.n_frames] * sm_tokenizer.compound_word_size) # Add EOS seconds
        
        # Limit the token len to max_len
        if sel_tokens.size()[0] > max_token_len:
            print("Decoder target length(%d) > max(%d)."%(sel_tokens.size()[0], max_token_len))
            sel_tokens = sel_tokens[:max_token_len]
            
        
        seq_len = sel_tokens.size()[0]
        
        decoder_targets[:seq_len] = sel_tokens
        
        for i, sec in enumerate(seconds[:seq_len-1]):
            frame_index = int(  np.round(sec / second_per_frame ) )
            frame_index = np.clip(frame_index, 0, self.n_frames - 1) # Clip to valid frame index
            decoder_targets_frame_index[i] = frame_index
        
        # Encoder Decoder Mask
        seconds = seconds[:seq_len-1]
        if hasattr(self.config.model, "encoder_decoder_slide_window_size") and self.config.model.encoder_decoder_slide_window_size > 0:
            for i, sec in enumerate(seconds):
                if i % 3 == 0:
                    encoder_decoder_mask[i, :] = 1 # Set the mask to 1 for onset tokens
                    assert Tokenizer.TokenOnset.is_instance(sel_tokens[i]) or sel_tokens[i] == sm_tokenizer.EOS or sel_tokens[i] == sm_tokenizer.PAD, "Onset token expected, but got %s"%(sel_tokens[i])
                    continue
                frame_index = int(  np.round(sec / second_per_frame ) )
                frame_index = np.clip(frame_index, 0, self.n_frames - 1) # Clip to valid frame index
                # decoder_targets_frame_index[i] = frame_index
                # begin_index = max(0, frame_index - int(self.config.model.encoder_decoder_slide_window_size//2))
                begin_index = frame_index - (frame_index % self.config.model.encoder_decoder_slide_window_size)
                end_index = begin_index + self.config.model.encoder_decoder_slide_window_size
                encoder_decoder_mask[i, begin_index:end_index] = 1
        else:
            encoder_decoder_mask[:,:] = 1
            
    
        # remove invalid vocab
        invalid_vocab_mask = decoder_targets >= sm_tokenizer.vocab_size
        if invalid_vocab_mask.int().sum() > 0:
            print("decoder_targets >= VOCAB_SIZE")
            decoder_targets[invalid_vocab_mask] = sm_tokenizer.PAD
            
        decoder_targets_mask = (decoder_targets != sm_tokenizer.PAD).long()

        
        audio_name = [ord(x) for x in self.audio_name][:256]
        audio_name = torch.tensor(audio_name, dtype=torch.long)
        audio_name = torch.nn.functional.pad(audio_name, pad=[0, 256 - audio_name.size()[0]], value=ord(" "))
        
        midi_path = [ord(x) for x in self.midi_path][:1024]
        midi_path = torch.tensor(midi_path, dtype=torch.long)
        midi_path = torch.nn.functional.pad(midi_path, pad=[0, 1024 - midi_path.size()[0]], value=ord(" "))
        
        dataset_name = [ord(x) for x in self.dataset_name][:32]
        dataset_name = torch.tensor(dataset_name, dtype=torch.long)
        dataset_name = torch.nn.functional.pad(dataset_name, pad=[0, 32 - dataset_name.size()[0]], value=ord(" "))
        
        pedal_target_mask = torch.ones(self.n_frames, dtype=torch.long)

        row.update({
            "inputs": audio,
            "decoder_targets": decoder_targets,
            "decoder_targets_mask": decoder_targets_mask,
            "decoder_targets_len": torch.tensor(seq_len, dtype=torch.long),
            "decoder_targets_frame_index": decoder_targets_frame_index,

            "encoder_decoder_mask": encoder_decoder_mask[None],

            "audio_ids": torch.tensor(int(self.audio_idx), dtype=torch.long),
            "audio_name": audio_name,
            "frame_offsets":torch.tensor(begin, dtype=torch.long),
            "total_frames": torch.tensor(int(self.total_frames), dtype=torch.long),
            "midi_path": midi_path,
            "dataset_name": dataset_name,
            "dataset_index": torch.tensor(self.dataset_index, dtype=torch.long),

            "pedal_frame_target": pedal_frame_target,
            "pedal_frame_target_mask": pedal_target_mask,
            "pedal_onset_target": pedal_onset_target,
            "pedal_onset_target_mask": pedal_target_mask,
            "pedal_offset_target": pedal_offset_target,
            "pedal_offset_target_mask": pedal_target_mask,
        })

        return  row
    
    
def get_subset(subset_list_path):
    paths = []
    with open(subset_list_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            assert len(line) == 2
            idx, path = line
            paths.append([idx, path])
        
    return paths

class Audio2Midi_Dataset(Dataset):
    def __init__(self, config, dataset_dir:str, dataset_index:int, subset='train', dataset_size = -1, random_clip = True) -> None:
        root_dir = dataset_dir
        
        subset_train = get_subset(root_dir + '/subset_train.tsv')
        subset_validation = get_subset(root_dir + '/subset_validation.tsv')
        subset_test = get_subset(root_dir + '/subset_test.tsv')
        
        if subset == "train":
            subset_data = subset_train
        elif subset == "validation":
            subset_data = subset_validation
        elif subset == "test":
            subset_data = subset_test
        elif subset == "all":
            subset_data = subset_train + subset_validation + subset_test
        else:
            raise "Unknown subset: " + subset
        
        
        self.audio_relative_paths = []
        self.mid_relative_paths = []
        self.dataset_dir = root_dir
        
        idx_list = []
        mid_paths = []
        
        for idx, path in subset_data:
            
            if( len(idx_list) >= 5) and subset != "test": # set mini set for debug
                # break
                pass
            
            mid_path = os.path.join(root_dir, config.data.cache_dir_name, os.path.splitext(path)[0].replace("/", ">") + ".mid")
            
            if os.path.exists(mid_path):
                idx_list.append(idx)
                mid_paths.append(mid_path)
                
                self.audio_relative_paths.append(path)
                self.mid_relative_paths.append(mid_path)
                
        audio_h5_path = root_dir + "/audio.h5"
        
        if subset == "test":
            pass
            if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
                pass
                idx_list = idx_list[::10]
                mid_paths = mid_paths[::10]   
            else:
                pass
                idx_list = idx_list[:] # Test set, only use 1/5 of the data.
                mid_paths = mid_paths[:]
        else:
            if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
                idx_list = idx_list[::5] # Test set, only use 1/5 of the data.
                mid_paths = mid_paths[::5]
                
        

        augmenter = _build_waveform_augmenter(config) if subset == "train" else None
        if augmenter is not None:
            print(f"[augmentation] waveform augmenter active for subset={subset} with {len(augmenter.transforms)} transform(s).")

        self.dataset_list = [ SingleWavDataset(config, dataset_dir, dataset_index, i, path, audio_h5_path, random_clip=random_clip, augmenter=augmenter)
            for i, path in tqdm(zip(idx_list, mid_paths), total=len(mid_paths))
        ]
        self.datasets = ConcatDataset(self.dataset_list)
        if dataset_size > 0:
            self.length = dataset_size
        else:
            self.length = len(self.datasets)
        # self.random = np.random.RandomState(42)
    def __len__(self):
        # length = np.sum([dataset.length for dataset in self.dataset_list])
        return self.length
        # return 100 * len(self.dataset_list)
    def __getitem__(self, index):
        return self.datasets[index]
    

    
    
    
def visualize(data_dict, output_fig_path):
    sr = DEFAULT_SAMPLE_RATE
    y = data_dict["inputs"].cpu().numpy()
    tokens = data_dict["decoder_targets"]
    cp_tokens = tokens.reshape([-1, sm_tokenizer.compound_word_size])
    cp_tokens = cp_tokens.cpu().numpy()
    cp_n = cp_tokens.shape[0]
    
    C = librosa.cqt(y, sr=sr)
    C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    dur_mins = len(y)/sr/60
    fig_width = 12 * min(dur_mins, 5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Draw Spectrogram
    librosa.display.specshow(C_dB, sr=sr, x_axis="time", y_axis="cqt_note", ax=ax)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    # Draw  Notes
    notes = []
    def get_val(cp_word, TokenType):
        return cp_word[TokenType.cp_index] - TokenType.begin_index

    beat_times = []
    downbeat_times = []
    tempo = 120 # 
    for idx in range(cp_n):
        cp_word = cp_tokens[idx]
        family = cp_word[0]
        start = get_val(cp_word, Tokenizer.TokenOnset) / Tokenizer.ONSET_SEC_UP_SAMPLING
        if family == Tokenizer.FamilyType.NOTE.value:
            pitch = get_val(cp_word, Tokenizer.TokenPitch)
            
            sec_per_quarter = 60 / tempo
            
            staff = get_val(cp_word, Tokenizer.TokenStaff)
            if Tokenizer.TokenBeat in sm_tokenizer.token_tpye_set:
                beat = get_val(cp_word, Tokenizer.TokenBeat)
                if beat % Tokenizer.BEAT_UP_SAMPLING == 0:
                    beat_times.append(start)
            
            end = start + get_val(cp_word, Tokenizer.TokenDur) / Tokenizer.DUR_TATUM_UP_SAMPLING * sec_per_quarter
            
            notes.append({
                "pitch": librosa.midi_to_note(pitch),
                "freq": librosa.midi_to_hz(pitch),
                "start": start,
                "end": end,
                "staff":staff
            })
        elif family == Tokenizer.FamilyType.MEASURE.value:
            if Tokenizer.TokenTempo in sm_tokenizer.token_tpye_set:
                tempo = get_val(cp_word, Tokenizer.TokenTempo) * Tokenizer.TEMPO_DOWN_SAMPLING
            downbeat_times.append(start)
        elif family == Tokenizer.FamilyType.EOS.value:
            break
        
    for n in notes:
        if n["staff"] == 0:
            color = "lime"
        else:
            color = "cyan"
        ax.hlines(
            y=n["freq"], xmin=n["start"], xmax=n["end"],
            colors=color, linewidth=2, label=n["pitch"]
        )
    
    # 避免重复 legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # ax.legend(unique.values(), unique.keys(), loc="upper right")
    ax.legend(list(unique.values())[:2], ["Staff 1", "Staff 2"], loc="upper right")
    
    
    
    # Draw Beat & Down Beat
    height = (ylim[1] - ylim[0])/4
    ymin = ylim[0]
    for t in beat_times:
        ax.vlines(t, ymin=ymin, ymax=ymin+height * 0.25, color='cyan', linestyle='dashed', linewidth=1)

    for t in downbeat_times:
        db_color = "magenta"
        db_color = "white"
        ax.vlines(t, ymin=ymin, ymax=ymin+height * 0.5, color=db_color, linestyle='solid', linewidth=1.5)
    
    
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    fig.savefig(output_fig_path, format="pdf", dpi=300)
    

    
## Test
@hydra.main(config_path="../config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    
    datasets = []
    for dataset_dir in config.data.dataset_dirs:
        if not os.path.exists(dataset_dir):
            raise ValueError("Dataset dir %s not exists!"%dataset_dir)
        datasets.append( Audio2Midi_Dataset(config, dataset_dir, dataset_index=0, subset="test", random_clip=True) )
        print(dataset_dir, " Len:", len(datasets[-1]))
    dataset = ConcatDataset(datasets)
    
    prev_audio_name = None
    ratios = defaultdict(list)
    for i, data in enumerate(tqdm(dataset)):
        continue
        
        
        if i % 5 != 0:
            pass
            # continue
        # print(i, data)
        # score_str = token_to_score(data["decoder_targets_onset"].cpu().numpy())
        # sy_score = dur_token_to_midi(data["decoder_targets_onset"].cpu().numpy())
        audio_name = data["audio_name"]
        frame_offsets_ratio = data["frame_offsets_ratio"]
        
        audio_name = "".join([chr(x) for x in audio_name.numpy()]).strip()
        onset_pitch_dict = defaultdict(list)
        targets = data["decoder_targets"].cpu().numpy()
        for i in range(0, data["decoder_targets"].size()[0], 2):
            onset = targets[i]
            pitch = targets[i+1]
            onset_pitch_dict[onset].append(pitch)
        if prev_audio_name == audio_name:
            continue
        if i % 1000 == 0:
            # print("Audio name:", audio_name)
            # print("Frame offset ratio:", frame_offsets_ratio)
            # print("Ratio mean:", np.mean(ratios))
            # ratios = []
            print(audio_name)
        
        continue
        
        audio_path = "preview/%05d_%s.wav"%(i, audio_name)
        audio = data["inputs"].cpu().numpy()
        torchaudio.save(audio_path, torch.tensor(audio)[None,], DEFAULT_SAMPLE_RATE)
        
        
        if i >  50000:
            break
        prev_audio_name = audio_name
        

if __name__ == "__main__":
    my_main()
