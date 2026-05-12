import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from omegaconf import OmegaConf
import hydra
from model.T5 import Transformer
from model.HPPNet import HPPNet
from model.GPT import GPT

from data.dataset_Audio2Midi import Audio2Midi_Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import wandb
import json
import pandas as pd
from torch.utils.data import IterableDataset, Dataset, ConcatDataset
# from torchaudio.transforms import MelSpectrogram

# from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from data.constants import *
import gc
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data.mel import MelSpectrogram
from data.spec_augment import SpecAugment
# from nnAudio.features import CQT, STFT
# import nnAudio.utils
from PIL import Image

import torchaudio
import metrics.transcription_metrics as transcription_metrics
import numpy as np

from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
import mir_eval

import wandb
from datetime import datetime
import time as std_time
import socket
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import importlib
from itertools import chain
from tqdm import tqdm
from glob import glob
import pandas as pd
#from line_profiler import LineProfiler
from symusic import Score, TimeUnit
from collections import defaultdict

import visualize.transcription_visualizer as transcription_visualizer
import utils.log_memory_usage as log_memory_usage
import utils.sequence_processing as sequence_processing
from model.Layers import normalize_transformer_ffn_activation

import shutil

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None


def set_training_process_name(process_name):
    process_name = str(process_name).strip()
    if not process_name:
        return

    if setproctitle is not None:
        try:
            setproctitle(process_name)
        except Exception:
            pass

    try:
        libc = ctypes.CDLL(None)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        libc.prctl(15, process_name[:15].encode("utf-8"), 0, 0, 0)
    except Exception:
        pass


def detect_checkpoint_format(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        lightning_keys = {
            "optimizer_states",
            "lr_schedulers",
            "loops",
            "callbacks",
            "pytorch-lightning_version",
        }
        if any(key in checkpoint for key in lightning_keys):
            return "lightning"
    if isinstance(checkpoint, dict):
        return "state_dict"
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def extract_model_state_dict(checkpoint, checkpoint_format):
    if checkpoint_format == "lightning":
        model_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("model."):
                model_state_dict[key[len("model."):]] = value
        if not model_state_dict:
            raise ValueError("Lightning checkpoint did not contain any model.* weights.")
        return model_state_dict
    return checkpoint


def remove_ignored_layers(state_dict, ignore_layers):
    removed_layers = []
    if ignore_layers is None:
        return removed_layers

    for key in ignore_layers:
        normalized_key = key[len("model."):] if key.startswith("model.") else key
        if normalized_key in state_dict:
            del state_dict[normalized_key]
            removed_layers.append(normalized_key)

    return removed_layers


def infer_transformer_ffn_activation_from_state_dict(state_dict):
    relu_keys = {
        "encoder.encoder_layers.0.mlp.intermediate_layers.0.weight",
        "decoder.decoder_layers.0.mlp.intermediate_layers.0.weight",
    }
    swiglu_keys = {
        "encoder.encoder_layers.0.mlp.value_layer.weight",
        "encoder.encoder_layers.0.mlp.gate_layer.weight",
        "decoder.decoder_layers.0.mlp.value_layer.weight",
        "decoder.decoder_layers.0.mlp.gate_layer.weight",
    }

    has_relu_keys = any(key in state_dict for key in relu_keys)
    has_swiglu_keys = any(key in state_dict for key in swiglu_keys)

    if has_relu_keys and has_swiglu_keys:
        raise ValueError("Checkpoint contains both ReLU and SwiGLU FFN keys; refusing to guess activation mode.")
    if has_relu_keys:
        return "relu"
    if has_swiglu_keys:
        return "swiglu"

    raise ValueError(
        "Unable to infer transformer FFN activation from checkpoint. "
        "Expected legacy ReLU keys like "
        "'encoder.encoder_layers.0.mlp.intermediate_layers.0.weight' "
        "or SwiGLU keys like "
        "'encoder.encoder_layers.0.mlp.value_layer.weight'."
    )


def validate_transformer_ffn_activation_compatibility(state_dict, configured_activation):
    config_activation = normalize_transformer_ffn_activation(configured_activation)
    checkpoint_activation = infer_transformer_ffn_activation_from_state_dict(state_dict)

    if checkpoint_activation != config_activation:
        raise ValueError(
            "Checkpoint FFN activation mismatch: "
            f"config.model.mlp_activations={config_activation!r} "
            f"but checkpoint uses {checkpoint_activation!r}. "
            "Use a checkpoint trained with the same FFN activation."
        )

    return checkpoint_activation


class MT3Trainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.model.model_name == "HPPNet":
            self.model = HPPNet(config=config.model)
        else:
            self.model = Transformer(config=config.model)
        self.last_time_stamp = std_time.time()
        self.validation_step_count = 0
        self.criterion_list = []
        for loss_name, loss_spec in config.training.losses.items():
            # Loss spec is [outputs_name, targets_name, criterion_module] or
            # [outputs_name, targets_name, criterion_module, weight]. Default weight = 1.0
            # keeps existing 3-element entries unchanged.
            if len(loss_spec) == 4:
                outputs_name, targets_name, criterion_module, weight = loss_spec
            else:
                outputs_name, targets_name, criterion_module = loss_spec
                weight = 1.0
            criterion_class_name = criterion_module.split(".")[-1]
            criterion_class = getattr(importlib.import_module(criterion_module), criterion_class_name)
            self.criterion_list.append({
                "loss_name": loss_name,
                "outputs_name": outputs_name,
                "targets_name": targets_name,
                "criterion": criterion_class(config),
                "weight": float(weight),
            }
            )

        self.feature_extracters = {}
        # mel_extracter = MelSpectrogram(sample_rate=DEFAULT_SAMPLE_RATE, n_mels=DEFAULT_NUM_MEL_BINS, n_fft=FFT_SIZE, hop_length=DEFAULT_HOP_WIDTH,  f_min=MEL_FMIN, f_max=MEL_FMAX)
        mel_extracter = MelSpectrogram(config.data.num_mel_bins, config.data.sample_rate, config.data.fft_size, config.data.hop_length,  mel_fmin=config.data.mel_fmin, mel_fmax=config.data.mel_fmax)
        # cqt_extracter = CQT(sr=config.data.sample_rate, hop_length=config.data.hop_length, fmin=config.data.cqt_fmin, n_bins=config.data.num_cqt_bins, bins_per_octave=config.data.bins_per_octave)
        # log_stft_extracter = STFT(n_fft=config.data.fft_size, freq_bins=config.data.num_cqt_bins, hop_length=config.data.hop_length, freq_scale="log", fmin=config.data.mel_fmin, fmax=config.data.mel_fmax, sr=config.data.sample_rate, output_format="Magnitude")
        
        # self.feature_extracters["mel"] = mel_extracter.to(self.device)
        # self.feature_extracters["cqt"] = cqt_extracter.to(self.device)
        if config.data.features == "mel":
            self.features_extracter = mel_extracter
        elif config.data.features == "cqt":
            self.features_extracter = cqt_extracter
        elif config.data.features == "log-stft":
            self.features_extracter = log_stft_extracter

        aug_cfg = config.data.get("augmentation", None)
        sa_cfg = aug_cfg.get("spec_augment", None) if aug_cfg is not None else None
        if aug_cfg is not None and aug_cfg.get("enabled", False) and sa_cfg is not None and sa_cfg.get("enabled", False):
            self.spec_augment = SpecAugment(
                freq_mask_param=sa_cfg.get("freq_mask_param", 13),
                n_freq_masks=sa_cfg.get("n_freq_masks", 2),
                time_mask_param=sa_cfg.get("time_mask_param", 40),
                n_time_masks=sa_cfg.get("n_time_masks", 2),
            )
        else:
            self.spec_augment = None
    

    def forward(self, encoder_input_tokens, decoder_target_tokens, decode, decoder_input_tokens = None, decoder_positions=None, decoder_targets_frame_index=None, encoder_decoder_mask=None):
        return self.model.forward(encoder_input_tokens, decoder_target_tokens, decode=decode, decoder_input_tokens=decoder_input_tokens, decoder_positions=decoder_positions, 
                decoder_targets_frame_index=decoder_targets_frame_index,
                encoder_decoder_mask=encoder_decoder_mask)
    
    def log_time_event(self, event_name):
        if self.global_rank == 0:
            curr_time = std_time.time()
            dur = curr_time - self.last_time_stamp
            if self.global_step % 200 == 0:
                self.log("time/%s"%event_name, dur)
                print("time/%s"%event_name, "%.3f"%dur)
            self.last_time_stamp = curr_time
    
    def forward_step(self, batch, batch_idx, forward_type, cal_metrics=False):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            outputs_dict: _description_
            loss_dict:
            metrics_dict: 
        """
        # concat combined dataloader manually.
        batch_list = list(batch.values())
        batch = {}
        for k in batch_list[0].keys():
            batch[k] = torch.cat([b[k] for b in batch_list], dim=0)
        

        outputs_dict = {}
        targets_dict = {}
        loss_dict = {}
        metrics_dict = {}

        n_steps = max(batch_idx, self.global_step)

        self.log_time_event("backword_done")
        with torch.no_grad():
            inputs = batch["inputs"]
            inputs = self.features_extracter(inputs[:, :-1]).transpose(-1, -2)
            inputs = inputs.detach()
            if self.config.data.amplitude_to_db:
                # assert self.config.data.features != "mel"
                # if self.config.data.features != "mel":
                inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(inputs)
            if self.spec_augment is not None and self.training:
                inputs = self.spec_augment(inputs)
        decoder_inputs = None

        targets_dict["input_features"] = inputs
        targets = batch["decoder_targets"]
        targets_mask = batch["decoder_targets_mask"]
        targets_len = batch["decoder_targets_len"]
        decoder_target_tokens = targets.clone()
        
        decoder_targets_frame_index = batch["decoder_targets_frame_index"]
        encoder_decoder_mask = batch["encoder_decoder_mask"]
        
        # Clip seq len if it is shorter than the max.
        if False:
            _, seq_len = targets.size()
            max_seq_lens = batch["decoder_targets_len"].max()
            if max_seq_lens < seq_len:
                targets = targets[:, :max_seq_lens]
                batch["decoder_targets_mask"] = batch["decoder_targets_mask"][:, :max_seq_lens]
        

        batch_size, seq_len = targets.size()

        targets_dict["decoder_targets"] = targets
        targets_dict["decoder_targets_mask"] = targets_mask
        targets_dict["decoder_targets_len"] = targets_len

        # Auxiliary per-frame pedal targets (state, onset, offset). Optional in batch:
        # only registered if the dataset emitted them, so older datasets / inference
        # paths that don't build these fields keep working.
        for pedal_key in (
            "pedal_frame_target",
            "pedal_onset_target",
            "pedal_offset_target",
        ):
            if pedal_key in batch:
                targets_dict[pedal_key] = batch[pedal_key]
                targets_dict[pedal_key + "_mask"] = batch[pedal_key + "_mask"]

        self.log_time_event("data_load_done")
        # CNN encoder still use input shape of [B, T, F].
        # => [B*T, n_token, vocab_size], [B, T, 128]
        outputs_dict = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=decoder_target_tokens, decode=False, decoder_input_tokens = decoder_inputs, 
                                    decoder_targets_frame_index=decoder_targets_frame_index,
                                    encoder_decoder_mask=encoder_decoder_mask)
        self.log_time_event("forward_done")

        # cal losses
        if "loss" in outputs_dict:
            # If model has already calculated loss.
            loss_dict["loss"] = outputs_dict["loss"]
        else:
            total_loss = 0
            for dic in self.criterion_list:
                outputs = outputs_dict[dic["outputs_name"]]
                _targets = targets_dict[dic["targets_name"]]
                _targets_mask = targets_dict[dic["targets_name"] + "_mask"]
                loss = dic["criterion"](outputs, _targets, _targets_mask)
                if not torch.isfinite(loss):
                    print("Error! Loss is inf!!!", loss)
                    loss = loss * 0
                if type(loss) is dict:
                    for k, v in loss.items():
                        loss_dict[dic["loss_name"] +  "_" + k] = v
                        total_loss += dic["weight"] * v
                else:
                    loss_dict[dic["loss_name"]] = loss
                    total_loss += dic["weight"] * loss
            loss_dict["loss"] = total_loss
        self.log_time_event("loss_cal_done")
        
        decoder_outputs = outputs_dict["decoder_outputs"]
        decoder_outputs_probs = torch.softmax(decoder_outputs, dim=-1)[..., :VOCAB_SIZE]
        decoder_targets_token = targets_dict["decoder_targets"]
        decoder_targets_mask = targets_dict["decoder_targets_mask"]
        
        # cal metrics
        if cal_metrics:
            decoder_outputs_token = torch.argmax(decoder_outputs_probs, dim=-1)
            decoder_outputs_token[ decoder_targets_mask==0 ] = TOKEN_PAD # ignore masked tokens.
            outputs_dict["decoder_outputs_token"] = decoder_outputs_token
            self.cal_decoder_token_metrics(decoder_outputs_token, decoder_targets_token, batch, metrics_dict, metric_prefix="")
            
        # Visualize decoder outputs.
        steps_list = self.config.training.save_pianoroll_every_n_steps
        if self.global_rank == 0 and (n_steps in steps_list or n_steps % steps_list[-1] == 0):
            img_path = os.path.join(self.logger.save_dir, "img") + "/decoder_outputs_%06d_stack_%s.png"%(n_steps, forward_type)
            decoder_targets_onehot = Functional.one_hot(decoder_targets_token, num_classes=VOCAB_SIZE)
            transcription_visualizer.save_frame_pianoroll(img_path, decoder_outputs_probs, decoder_targets_onehot)
                

        # Save attention weights to local dir.
        steps_list = self.config.training.save_pianoroll_every_n_steps
        if self.global_rank == 0 and ( n_steps < 1 or (n_steps in steps_list or n_steps % steps_list[-1] == 0) ):
            if "cross_attention_weights" in outputs_dict:
                self.visualize_attention_weights(outputs_dict["cross_attention_weights"], n_steps, forward_type, batch)
            # layer_index = 0
            for k,v in outputs_dict.items():
                if k.startswith("decoder_layer_cross_attention_weights"):
                    layer_index = int(k.split("_")[-1])
                    self.visualize_attention_weights(v, n_steps, forward_type, batch, layer_index=layer_index)
                    # layer_index += 1

                    
        # Save input features to local dir for preview.
        if self.global_rank == 0 and n_steps < 5:
            if "input_features" in targets_dict:
                B, T, F = targets_dict["input_features"].size()
                flatten_input_features = targets_dict["input_features"].reshape([B*T, F]).cpu().numpy().T
                img_path = os.path.join(self.logger.save_dir, "img") + "/input_features_%s_%03d.png"%(forward_type, n_steps)
                wav_path = os.path.join(self.logger.save_dir, "img") + "/input_features_%s_%03d.mp3"%(forward_type, n_steps)
                plt.imsave(img_path, flatten_input_features)
                torchaudio.save(wav_path, batch["inputs"].cpu().detach().flatten()[None,:], sample_rate=self.config.data.sample_rate, format="mp3")

        return outputs_dict, targets_dict, loss_dict, metrics_dict
    
    def visualize_attention_weights(self, attention_weights, n_steps, forward_type, batch, layer_index = -1):
        """save attention weights img to local dir."""
        cross_attention_weights = attention_weights.detach().cpu()
        # normalize attention weights to range [0, 1]
        batch_size, n_heads, output_seq_len, input_seq_len = cross_attention_weights.size()
        for i in range(batch_size):
            if i > 4:
                break
            decoder_targets_len = batch["decoder_targets_len"][i].item() // SEQUENCE_POOLING_SIZE
            img_path = os.path.join(self.logger.save_dir, "img") + "/decoder_cross_attention_%s_%05d_idx=%02d_layer=%d.png"%(forward_type, n_steps, i, layer_index)
            # Save attention weights to local dir.
            weight_i = cross_attention_weights[i, :, :decoder_targets_len, :]
            min_i = weight_i.min()
            max_i = weight_i.max()
            if max_i > 0:
                weight_i = (weight_i - min_i) / (max_i - min_i)
            else:
                weight_i = weight_i - min_i
            if min_i < 0 or max_i > 1:
                print("Error! Attention weights out of range [0, 1]!", min_i, max_i)
            transcription_visualizer.save_frame_pianoroll(img_path, weight_i, weight_i*0)
    
    @rank_zero_only
    def cal_decoder_token_metrics(self, decoder_outputs_token, decoder_targets_token, batch, metrics_dict, metric_prefix=""):
        """Visualize decoder outputs.

        Args:
            decoder_outputs_token (_type_): [batch, seq_len]
            decoder_targets_token (_type_): [batch, seq_len]
            batch (_type_): _description_
            metric_prefix (_type_): _description_
        """

        self.log_time_event("metrics_cal_begin")
        metrict_ret = transcription_metrics.cal_decoder_token_metrics(decoder_outputs_token, decoder_targets_token, metric_prefix=metric_prefix)
        
        # cal metrics for different datasets.
        if len(self.config.data.dataset_dirs) > 1:
            dataset_indexs = set(batch["dataset_index"].cpu().numpy())
            for dataset_index in dataset_indexs:
                indexs = batch["dataset_index"] == dataset_index
                sel_target_tokens = decoder_targets_token[indexs]
                sel_output_tokens = decoder_outputs_token[indexs]
                sel_metrict_ret = transcription_metrics.cal_decoder_token_metrics(sel_output_tokens, sel_target_tokens, metric_prefix=metric_prefix)
                for k, v in sel_metrict_ret.items():
                    metrict_ret["%s_%d"%(k, dataset_index)] = v
        
        metrics_dict.update(metrict_ret)
    

    def training_step(self, batch, batch_idx, dataloader_idx = None): # This batch_indx will be reseted to 0 in a new epoch.
        cal_metrics = False
        if self.global_rank == 0 and self.global_step % self.config.training.cal_metrics_every_n_steps == 0:
            cal_metrics = True
        outputs_dict, targets_dict, loss_dict, metrics_dict = self.forward_step(batch, batch_idx, "train", cal_metrics)

        if self.global_rank == 0:
            # save checkpoint.
            if self.global_step > 0:
                if self.global_step in [1000, 2000, 3000, 4000, 5000, 8000, 10000] or self.global_step % self.config.training.checking_steps == 0:
                    cpt_dir = os.path.join(os.path.join(self.logger.save_dir, "cpt"))
                    os.system("mkdir -p %s"%cpt_dir)
                    checkpoint_path = cpt_dir + '/steps_' + str(self.global_step)+'.ckpt'
                    torch.save(self.model.state_dict(), checkpoint_path)
                    if hasattr(self, "previous_cpt_path"):
                        if self.prev_checkpoint_step in [10000, 50000, 100000, 200000]:
                            pass
                        else:
                            if os.path.exists(self.previous_cpt_path):
                                try:
                                    os.remove(self.previous_cpt_path)
                                except Exception as e:
                                    print("Error removing previous checkpoint:", e)
                    self.previous_cpt_path = checkpoint_path
                    self.prev_checkpoint_step = self.global_step
            
            # log loss and metrics.
            log_dict = metrics_dict.copy()
            log_dict.update(loss_dict)
            if self.global_step % 10 == 0:
                for k,v in log_dict.items():
                    if isinstance(v, (int, float, bool, torch.Tensor)):
                        self.log("train/%s"%k, v, sync_dist=False)
                # log memory usage
                if self.global_step % 200 == 0:
                    log_memory_usage.log_gpu_memory_usage(self)

        self.log_time_event("forward_step_done")
        
        return loss_dict["loss"]
    
    @rank_zero_only
    def on_train_epoch_end(self,):
        # Save checkpoint.
        cpt_dir = os.path.join(os.path.join(self.logger.save_dir, "cpt"))
        os.system("mkdir -p %s"%cpt_dir)
        torch.save(self.model.state_dict(), cpt_dir + '/latest.ckpt')
        txt_path = cpt_dir + '/latest_epoch.txt'
        with open(txt_path, "w") as f:
            f.write('latest epoch=%d , global step=%d\n'%(self.current_epoch, self.global_step))
                
        
    @torch.no_grad()
    def on_validation_start(self):
        
        # Manually call test_steps (online testing).
        if self.global_step > 1 and self.config.training.online_testing: # skip the validation step in the init epoch.
            self.test_outputs_dict = defaultdict(list)
            test_dataloader = self.test_dataloader()
            for batch_idx, batch in tqdm( enumerate(test_dataloader), desc="Testing ...", total=len(test_dataloader), disable=(self.global_rank != 0) ):
                new_batch = {}
                for k,v in batch.items():
                    new_batch[k] = v.to(self.device) # GPU memory keak problem here!
                self.test_step(new_batch, batch_idx)
            self.on_test_epoch_end()
        
        return super().on_validation_start()
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        outputs_dict, targets_dict, loss_dict, metrics_dict = self.forward_step(batch, batch_idx, "validation", cal_metrics=True)
        log_dict = {}
        log_dict.update(loss_dict)
        log_dict.update(metrics_dict)
        for k,v in log_dict.items():
            if isinstance(v, (int, float, bool, torch.Tensor)):
                self.log("validation/%s"%k, v, on_epoch=True, sync_dist=False) #
        
        return loss_dict["loss"]
    
    
    # @torch.no_grad()
    # def test_step_(self, batch, batch_idx):
    #     return self.test_step(batch, batch_idx)
    #     lp = LineProfiler()
    #     lp_wrapper = lp(self.test_step_parallel)
    #     lp_wrapper(batch, batch_idx)
    #     lp.print_stats()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        metrics_dict = {}
        # [B, n_wave_samples]
        input_waves = batch["inputs"]
        with torch.no_grad():
            # => [B, T, F]
            encoder_inputs = self.features_extracter(input_waves[:, :-1]).transpose(-1, -2)
            encoder_inputs = encoder_inputs.detach()
            if self.config.data.amplitude_to_db:
                # assert self.config.data.features != "mel"
                if self.config.data.features != "mel":
                    encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(encoder_inputs)

        pedal_head_outputs = {}
        pedal_head_specs = (
            ("pedal_frame", "pedal_frame_head", "pedal_frame_target"),
            ("pedal_onset", "pedal_onset_head", "pedal_onset_target"),
            ("pedal_offset", "pedal_offset_head", "pedal_offset_target"),
        )
        available_pedal_head_specs = [
            spec
            for spec in pedal_head_specs
            if spec[2] in batch and hasattr(self.model, spec[1])
        ]
        if len(available_pedal_head_specs) > 0:
            pedal_encoder_outputs = self.model.encode(encoder_inputs, enable_dropout=False)
            for pedal_name, pedal_head_name, _ in available_pedal_head_specs:
                pedal_head = getattr(self.model, pedal_head_name)
                pedal_head_outputs[f"{pedal_name}_output"] = torch.sigmoid(
                    pedal_head(pedal_encoder_outputs).squeeze(-1)
                ).detach()

        # [B, n_tokens]
        target_tokens = batch["decoder_targets"] 
        n_tokens = target_tokens.size()[1]
        
        # inference speed
        curr_time_sec = std_time.time()
        decoder_output_tokens = self.model.generate(encoder_inputs, target_seq_length= n_tokens, berak_on_eos=True, global_rank=self.global_rank)
        dur_time_sec = std_time.time() - curr_time_sec
        generate_token_len = max(1, decoder_output_tokens.size(1) )
        tokens_per_sec = generate_token_len / dur_time_sec
        
        # get_output_seq_lens
        output_eos_tokens_flag = decoder_output_tokens
        output_eos_tokens_flag = (output_eos_tokens_flag == sm_tokenizer.EOS).int() * sm_tokenizer.EOS
        output_seq_lens = sequence_processing.get_sequence_lengths(output_eos_tokens_flag, sm_tokenizer.EOS)
        target_sequence_lens = batch["decoder_targets_len"]
        assert len(output_seq_lens) == len(target_sequence_lens), f"Output sequence length {len(output_seq_lens)} does not match target sequence length {len(target_sequence_lens)}."
        
        def pad_to_max_size(tensor, max_size, value = -1):
            pad_size = max_size - tensor.size(0)
            if pad_size > 0:
                padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)  # Only pad batch dimension
                tensor = torch.nn.functional.pad(tensor, padding, value=value)
            return tensor
        
        # Pad the tensors of different ranks to the maximum size
        batch_size = batch["audio_ids"].size(0)
        batch_sizes = self.all_gather(batch_size)  # Gather batch sizes first
        max_batch_size = int(batch_sizes.max())
        
        gather_payload = {
            "target_tokens": pad_to_max_size(target_tokens, max_batch_size),
            "output_tokens": pad_to_max_size(decoder_output_tokens, max_batch_size),
            "target_tokens_lens": pad_to_max_size(target_sequence_lens, max_batch_size),
            "output_tokens_lens": pad_to_max_size(output_seq_lens, max_batch_size),
            
            "audio_ids": pad_to_max_size(batch["audio_ids"], max_batch_size),
            "audio_name":pad_to_max_size(batch["audio_name"], max_batch_size),
            "midi_path": pad_to_max_size(batch["midi_path"], max_batch_size),
            "frame_offsets": pad_to_max_size(batch["frame_offsets"], max_batch_size),
            
        }
        if "total_frames" in batch:
            gather_payload["total_frames"] = pad_to_max_size(batch["total_frames"], max_batch_size)
        for pedal_output_name, pedal_output in pedal_head_outputs.items():
            pedal_target_name = pedal_output_name.replace("_output", "_target")
            gather_payload[pedal_output_name] = pad_to_max_size(pedal_output, max_batch_size)
            gather_payload[pedal_target_name] = pad_to_max_size(batch[pedal_target_name], max_batch_size)

        gather_data = self.all_gather(gather_payload)

        # On master rank, process or aggregate the gathered outputs as needed
        if self.global_rank == 0:
            # => [n_rank, B, ...]
            world_size, B, T = gather_data["output_tokens"].size()
            
            # assert len(output_seq_lens) == B * world_size # 
            for w in range(world_size):
                for b in range(B):
                    audio_id = gather_data["audio_ids"][w][b].item()
                    if audio_id < 0: # Skip padding audio_id
                        continue
                    target_len_b = gather_data["target_tokens_lens"][w][b].item()
                    output_len_b = gather_data["output_tokens_lens"][w][b].item()
                    
                    audio_name = "".join([chr(x) for x in gather_data["audio_name"][w][b].detach().cpu().numpy()]).strip()
                    midi_path = "".join([chr(x) for x in gather_data["midi_path"][w][b].detach().cpu().numpy()]).strip()
                    
                    target_tokens_i = gather_data["target_tokens"][w][b, :target_len_b].cpu().numpy()
                    output_tokens_i = gather_data["output_tokens"][w][b, :output_len_b].cpu().numpy()
                    
                    output_row = {
                        "target_tokens": target_tokens_i.tolist(),
                        "output_tokens": output_tokens_i.tolist(),
                        "audio_name": audio_name,
                        "midi_path": midi_path,
                        "frame_offsets": gather_data["frame_offsets"][w][b].item(),
                        "audio_ids": audio_id,
                        "target_tokens_lens": target_len_b,
                        "output_tokens_lens": output_len_b,
                        "batch_size": int(max_batch_size),
                    }
                    if "total_frames" in gather_data:
                        output_row["total_frames"] = gather_data["total_frames"][w][b].item()
                    for pedal_output_name in (
                        "pedal_frame_output",
                        "pedal_onset_output",
                        "pedal_offset_output",
                    ):
                        if pedal_output_name in gather_data:
                            pedal_target_name = pedal_output_name.replace("_output", "_target")
                            output_row[pedal_output_name] = gather_data[pedal_output_name][w][b].detach().cpu().numpy().tolist()
                            output_row[pedal_target_name] = gather_data[pedal_target_name][w][b].detach().cpu().numpy().tolist()

                    self.test_outputs_dict[audio_id].append(output_row)
                    

        return metrics_dict
    

        
    @rank_zero_only
    def on_test_epoch_start(self,):
        # self.test_outputs_dict = defaultdict(lambda: defaultdict(list))
        self.test_outputs_dict = defaultdict(list)
    
    @rank_zero_only
    def on_test_epoch_end(self,): #  outputs
        
        print("on_test_epoch_end, test_outputs_dict size:", len(self.test_outputs_dict))
        
        metric_dict = {
            "note_precision": [],
            "note_recall": [],
            "note_f1": [],
            "target_length": [],
            "midi_path": [],
        }
        
        metric_dict = defaultdict(list)

        evaluation_config = self.config.get("evaluation", {})
        save_track_json = evaluation_config.get("save_track_json", True)
        save_output_midi = evaluation_config.get("save_output_midi", True)
        copy_reference_midi = evaluation_config.get("copy_reference_midi", True)
        report_pedal_metrics = evaluation_config.get("report_pedal_metrics", True)
        report_pedal_extended = evaluation_config.get("report_pedal_extended", False)
        pedal_event_source = str(evaluation_config.get("pedal_event_source", "decoder"))
        midi_pedal_event_source = str(evaluation_config.get("midi_pedal_event_source", "decoder"))
        frame_head_threshold_on = float(evaluation_config.get("frame_head_threshold_on", 0.5))
        frame_head_threshold_off = float(evaluation_config.get("frame_head_threshold_off", 0.4))
        frame_head_min_down_frames = int(evaluation_config.get("frame_head_min_down_frames", 3))
        frame_head_min_up_frames = int(evaluation_config.get("frame_head_min_up_frames", 2))
        pedal_reference_metric_names = (
            "pedal_precision",
            "pedal_recall",
            "pedal_f1",
        )
        pedal_diagnostic_metric_names = (
            "pedal_precision",
            "pedal_recall",
            "pedal_f1",
            "pedal+offset_precision",
            "pedal+offset_recall",
            "pedal+offset_f1",
            "pedal_on_precision",
            "pedal_on_recall",
            "pedal_on_f1",
            "pedal_off_precision",
            "pedal_off_recall",
            "pedal_off_f1",
        )
        valid_pedal_event_sources = {"decoder", "frame_head"}
        if pedal_event_source not in valid_pedal_event_sources:
            raise ValueError(f"evaluation.pedal_event_source must be one of {sorted(valid_pedal_event_sources)}, got {pedal_event_source!r}.")
        if midi_pedal_event_source not in valid_pedal_event_sources:
            raise ValueError(f"evaluation.midi_pedal_event_source must be one of {sorted(valid_pedal_event_sources)}, got {midi_pedal_event_source!r}.")

        # Save test tokens to json
        if self.config.training.mode == "test":
            test_output_dir = self.test_output_dir
        else:
            test_output_dir = self.logger.save_dir + "/test/step=%06d"%(self.global_step)
        os.makedirs(test_output_dir, exist_ok=True)
        
        for audio_id, data_list in tqdm(self.test_outputs_dict.items(), desc="Saving test results", disable=self.global_rank != 0):
            data_list = sorted(data_list, key=lambda x: x["frame_offsets"])
            audio_name = data_list[0]["audio_name"]
            target_midi_path = data_list[0]["midi_path"]
            # audio_name = audio_name.replace(" ", "_")
            
            # copy midi file to test output dir.
            if copy_reference_midi:
                shutil.copy(target_midi_path, test_output_dir + "/" + os.path.basename(target_midi_path))
            
            metric_dict["batch_size"].append( data_list[0]["batch_size"] )
            
            concat_output_tokens = sum([row["output_tokens"] for row in data_list], [])
            concat_target_tokens = sum([row["target_tokens"] for row in data_list], [])
            
            # Calculate note and pedal metrics
            output_event_data_list = []
            target_event_data_list = []
            sec_per_frame = DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
            for row in data_list:
                offsets_sec = row["frame_offsets"] * sec_per_frame
                output_event_data_list += sm_tokenizer.detokenize(row["output_tokens"], offsets_sec)
                target_event_data_list += sm_tokenizer.detokenize(row["target_tokens"], offsets_sec)
            output_note_data_list, decoder_token_pedal_event_list = sm_tokenizer.midi_events_to_notes(output_event_data_list)
            target_note_data_list, target_pedal_event_list = sm_tokenizer.midi_events_to_notes(target_event_data_list)

            frame_head_pedal_event_list = None
            if len(data_list) > 0 and "pedal_frame_output" in data_list[0]:
                frame_head_pedal_frame_outputs = transcription_metrics.collect_trimmed_frame_arrays(
                    data_list,
                    "pedal_frame_output",
                )
                frame_head_pedal_event_list = transcription_metrics.pedal_frame_output_to_events(
                    frame_head_pedal_frame_outputs,
                    sec_per_frame=sec_per_frame,
                    threshold_on=frame_head_threshold_on,
                    threshold_off=frame_head_threshold_off,
                    min_pedal_down_frames=frame_head_min_down_frames,
                    min_pedal_up_frames=frame_head_min_up_frames,
                )

            output_pedal_event_list, pedal_event_source_used = transcription_metrics.choose_pedal_event_list(
                decoder_token_pedal_event_list,
                frame_head_pedal_event_list,
                pedal_event_source,
            )
            midi_pedal_event_list, midi_pedal_event_source_used = transcription_metrics.choose_pedal_event_list(
                decoder_token_pedal_event_list,
                frame_head_pedal_event_list,
                midi_pedal_event_source,
            )
            metric_dict["pedal_event_source_used"].append(pedal_event_source_used)
            metric_dict["midi_pedal_event_source_used"].append(midi_pedal_event_source_used)

            tsv_path = os.path.splitext(target_midi_path)[0] + ".midi-notes.tsv"
            if os.path.exists(tsv_path):
                reference_df = pd.read_csv(tsv_path, sep="\t")
            else:
                reference_df = sm_tokenizer.midi_to_dataframe(target_midi_path)
            reference_pedal_event_list = transcription_metrics.reference_pedal_events_from_dataframe(reference_df)

            piece_end_time = transcription_metrics.infer_piece_end_time(
                note_lists=[output_note_data_list, target_note_data_list],
                pedal_event_lists=[
                    decoder_token_pedal_event_list,
                    frame_head_pedal_event_list or [],
                    output_pedal_event_list,
                    midi_pedal_event_list,
                    target_pedal_event_list,
                    reference_pedal_event_list,
                ],
            )
            source_pedal_event_lists = {
                "decoder_token": decoder_token_pedal_event_list,
                "frame_head": frame_head_pedal_event_list,
            }

            if len(reference_pedal_event_list) > 0 and report_pedal_metrics:
                metric_dict["pedal_metrics_status"].append("available")
                pedal_metrics, output_pedal_spans, _ = transcription_metrics.cal_reference_pedal_metrics(
                    output_pedal_event_list,
                    reference_pedal_event_list,
                    piece_end_time=piece_end_time,
                )
                for metric_name, metric_value in pedal_metrics.items():
                    metric_dict[metric_name].append(metric_value)

                pedal_diagnostics, _, _ = transcription_metrics.cal_pedal_metrics(
                    output_pedal_event_list,
                    reference_pedal_event_list,
                    piece_end_time=piece_end_time,
                )
                for metric_name, metric_value in pedal_diagnostics.items():
                    metric_dict["diagnostic_" + metric_name].append(metric_value)

                for source_prefix, source_events in source_pedal_event_lists.items():
                    if source_events is None:
                        for metric_name in pedal_reference_metric_names:
                            metric_dict[f"{source_prefix}_reference_{metric_name}"].append(np.nan)
                        for metric_name in pedal_diagnostic_metric_names:
                            metric_dict[f"{source_prefix}_{metric_name}"].append(np.nan)
                        continue

                    source_reference_metrics, _, _ = transcription_metrics.cal_reference_pedal_metrics(
                        source_events,
                        reference_pedal_event_list,
                        piece_end_time=piece_end_time,
                    )
                    for metric_name, metric_value in source_reference_metrics.items():
                        metric_dict[f"{source_prefix}_reference_{metric_name}"].append(metric_value)

                    source_diagnostic_metrics, _, _ = transcription_metrics.cal_pedal_metrics(
                        source_events,
                        reference_pedal_event_list,
                        piece_end_time=piece_end_time,
                    )
                    for metric_name, metric_value in source_diagnostic_metrics.items():
                        metric_dict[f"{source_prefix}_{metric_name}"].append(metric_value)
            else:
                if report_pedal_metrics:
                    metric_dict["pedal_metrics_status"].append("not_available_no_reference_pedals")
                else:
                    metric_dict["pedal_metrics_status"].append("disabled")
                output_pedal_spans = transcription_metrics.pedal_events_to_spans(
                    output_pedal_event_list,
                    piece_end_time=piece_end_time,
                )
                for metric_name in pedal_reference_metric_names:
                    metric_dict[metric_name].append(np.nan)
                for metric_name in pedal_diagnostic_metric_names:
                    metric_dict["diagnostic_" + metric_name].append(np.nan)
                for source_prefix in source_pedal_event_lists:
                    for metric_name in pedal_reference_metric_names:
                        metric_dict[f"{source_prefix}_reference_{metric_name}"].append(np.nan)
                    for metric_name in pedal_diagnostic_metric_names:
                        metric_dict[f"{source_prefix}_{metric_name}"].append(np.nan)

            pedal_frame_metric_specs = (
                ("pedal_frame", "pedal_frame_output", "pedal_frame_target"),
                ("pedal_onset_frame", "pedal_onset_output", "pedal_onset_target"),
                ("pedal_offset_frame", "pedal_offset_output", "pedal_offset_target"),
            )
            for metric_prefix, output_key, target_key in pedal_frame_metric_specs:
                if len(data_list) > 0 and output_key in data_list[0]:
                    frame_outputs, frame_targets = transcription_metrics.collect_trimmed_frame_arrays(
                        data_list,
                        output_key,
                        target_key,
                    )
                    frame_metrics = transcription_metrics.cal_binary_frame_metrics(
                        metric_prefix,
                        frame_outputs,
                        frame_targets,
                    )
                    for metric_name, metric_value in frame_metrics.items():
                        metric_dict[metric_name].append(metric_value)
                else:
                    metric_dict[f"{metric_prefix}_precision"].append(np.nan)
                    metric_dict[f"{metric_prefix}_recall"].append(np.nan)
                    metric_dict[f"{metric_prefix}_f1"].append(np.nan)
            
            # Onset only Metrics
            output_interval = np.array([(note["onset"], note["offset"]) for note in output_note_data_list])
            target_interval = np.array([(note["onset"], note["offset"]) for note in target_note_data_list])
            output_pitches = np.array([note["pitch"] for note in output_note_data_list])
            target_pitches = np.array([note["pitch"] for note in target_note_data_list])
            output_velocities = np.array([note["velocity"] for note in output_note_data_list])
            target_velocities = np.array([note["velocity"] for note in target_note_data_list])
            output_pitches = mir_eval.util.midi_to_hz(output_pitches)
            target_pitches = mir_eval.util.midi_to_hz(target_pitches)
            
            precision, recall, f1, overlap = evaluate_notes(target_interval, target_pitches, output_interval, output_pitches, offset_ratio=None)
            metric_dict["note_precision"].append(precision)
            metric_dict["note_recall"].append(recall)
            metric_dict["note_f1"].append(f1)
            
            precision, recall, f1, overlap = evaluate_notes(target_interval, target_pitches, output_interval, output_pitches)
            metric_dict["note+offset_precision"].append(precision)
            metric_dict["note+offset_recall"].append(recall)
            metric_dict["note+offset_f1"].append(f1)
            
            precision, recall, f1, overlap = evaluate_notes_with_velocity(target_interval, target_pitches, target_velocities, output_interval, output_pitches, output_velocities, velocity_tolerance=0.1)
            metric_dict["note+offset+velocity_precision"].append(precision)
            metric_dict["note+offset+velocity_recall"].append(recall)
            metric_dict["note+offset+velocity_f1"].append(f1)

            # Pedal-extended evaluation: apply reference-style sustain semantics
            # to raw predicted/target offsets. Uncapped cached-TSV semantics are
            # also reported under diagnostic_*_uncapped metric names.
            if report_pedal_extended:
                pedal_extended_metrics, _ = transcription_metrics.cal_pedal_extended_note_metrics(
                    output_note_data_list,
                    output_pedal_event_list,
                    reference_df,
                    piece_end_time=piece_end_time,
                )
                for metric_name, metric_value in pedal_extended_metrics.items():
                    metric_dict[metric_name].append(metric_value)

            target_len = max(len(concat_target_tokens), 1) # Avoid division by zero.
            
            metric_dict["target_length"].append(target_len)
            target_midi_basename = os.path.basename(target_midi_path)
            metric_dict["midi_path"].append(target_midi_basename)
            
            if save_track_json:
                json_path = test_output_dir + "/" + os.path.splitext(target_midi_basename)[0] + ".output.json"
                with open(json_path, "w") as f:
                    data_dict = {
                        "audio_id": audio_id,
                        "audio_name": audio_name,
                        "midi_path": target_midi_basename,
                        "data_list": data_list
                    }
                    json.dump(data_dict, f, ensure_ascii=False, indent=2)
                
            if save_output_midi:
                try:
                    output_midi_path = test_output_dir + "/" + os.path.splitext(target_midi_basename)[0] + ".output.mid"
                    midi_output_pedal_spans = transcription_metrics.pedal_events_to_spans(
                        midi_pedal_event_list,
                        piece_end_time=piece_end_time,
                    )
                    output_pedal_events_for_midi = transcription_metrics.pedal_spans_to_event_list(midi_output_pedal_spans)
                    sm_tokenizer.save_midi(output_note_data_list, output_midi_path, pedal_event_list=output_pedal_events_for_midi)
                except Exception as e:
                    print("Error saving midi file:", e)
                
        df = pd.DataFrame(metric_dict)
        # Calculate mean and insert into the first row
        avg_value = df.select_dtypes(include="number").mean()
        df.loc["Average"] = avg_value
        df = pd.concat([df.loc[["Average"]], df.drop("Average")])
        df.to_csv(test_output_dir + "/!test_metrics_summary.csv", index=False)
        
        # Log Metrics.
        if self.config.training.mode == "train":
            for k, v in avg_value.items():
                self.log("test/%s"%k, v)


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.training.learning_rate)

        # scheduler = CosineAnnealingLR(optimizer, T_max=10)
        # scheduler.get_lr()
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        
        return optimizer
    
    
    def train_dataloader(self):
        # train_data = MIDIDataset(type='train')
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!"%dataset_dir)
                datasets.append( Audio2Midi_Dataset(self.config, dataset_dir, dataset_index=dataset_index, subset="train", random_clip=True) )
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name
        
        sampler = None
        
        # Set dataloader for multiple datasets 
        n_datasets = len(self.config.data.dataset_dirs)
        assert self.config.training.batch % n_datasets == 0, "Batch size should be divisible by the number of datasets."
        iterables = {
            str(i): DataLoader(dataset, 
                batch_size=self.config.training.batch // n_datasets,
                num_workers=self.config.training.num_workers // n_datasets,
                sampler=sampler,
                shuffle=True,
            )
            for i, dataset in enumerate(datasets)
        }
        combined_loader = CombinedLoader(iterables, mode="max_size_cycle")
        return combined_loader
    
    def val_dataloader(self):
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!"%dataset_dir)
                datasets.append( Audio2Midi_Dataset(self.config, dataset_dir, dataset_index=dataset_index, subset="validation", random_clip=False) )
            validation_data = ConcatDataset(datasets)
            # validation_data = Audio2Midi_Dataset(self.config, subset="validation", random_clip=False)
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name
            
        sampler=None
        shuffle=False
            
        # Set dataloader for multiple datasets 
        n_datasets = len(self.config.data.dataset_dirs)
        assert self.config.training.batch % n_datasets == 0, "Batch size should be divisible by the number of datasets."
        iterables = {
            str(i): DataLoader(dataset, 
                batch_size=self.config.training.batch // n_datasets,
                num_workers=self.config.training.num_workers // n_datasets,
                sampler=sampler,
                shuffle=shuffle,
            )
            for i, dataset in enumerate(datasets)
        }
        combined_loader = CombinedLoader(iterables, mode="max_size_cycle")
        return combined_loader

    def test_dataloader(self, subset="test"):
        if subset not in {"test", "validation"}:
            raise ValueError(f"test_dataloader subset must be 'test' or 'validation', got {subset!r}.")
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!"%dataset_dir)
                datasets.append( Audio2Midi_Dataset(self.config, dataset_dir, dataset_index=dataset_index, subset=subset, random_clip=False) )
            test_data = ConcatDataset(datasets)
            # test_data = Audio2Midi_Dataset(self.config, subset="test", random_clip=False)
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name

        testloader = DataLoader(test_data, batch_size=self.config.training.batch_test, num_workers=self.config.training.num_workers)
        return testloader


@rank_zero_only
def Init_rank_zero_only(log_dir):
    os.environ["MT3-LOGGER-PATH"] = log_dir
    

@hydra.main(config_path="config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    torch.cuda.empty_cache()
    gc.collect()

    # if socket.gethostname() == "sacs14":
    #     config.data.dataset_dir='/nvme1/wei/data/maestro-v3.0.0/'
        

    ####################################################
    # Get log dir.
    model_name = config.model.model_name
    experiment_name = "_".join([model_name])

    if config.training.log_dir is None:
        time_str = datetime.now().strftime('%y%m%d-%H%M%S')
        # loss_name = config.training.loss
        batch_size = "BS" + str(config.training.batch)
        token_types = str(config.data.token_types)
        log_workdir = "runs"
        if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
            log_workdir = "runs-debug"
        assert config.training.notes is not None and len(config.training.notes) > 0, "Please set config.training.notes to a non-empty string."
        run_name = "_".join([time_str, config.training.notes]) # batch_size, config.data.dataset_name
        log_dir = os.path.join(log_workdir, experiment_name , run_name)
        Init_rank_zero_only(log_dir)
        log_dir = os.environ["MT3-LOGGER-PATH"]
        config.training.log_dir=log_dir
    else:
        log_dir = config.training.log_dir
    
    process_note = str(config.training.notes).strip() if config.training.notes is not None else ""
    process_name = experiment_name if not process_note else f"{experiment_name}_{process_note}"
    set_training_process_name("group 5 will end 5/8 ~ 11:00")
    ####################################################
    # Create model.
    model = MT3Trainer(config)
    print(model)
    resume_ckpt_path = None
    # Load checkpoint.
    if config.model.checkpoint_path is None:
        assert config.model.froze_encoder == False, "If you want to train from scratch, please set config.model.froze_encoder to False and config.model.checkpoint_path to None."
    else:
        checkpoint = torch.load(config.model.checkpoint_path, map_location="cpu")
        checkpoint_format = detect_checkpoint_format(checkpoint)
        model_state_dict = extract_model_state_dict(checkpoint, checkpoint_format)
        validate_transformer_ffn_activation_compatibility(model_state_dict, config.model.mlp_activations)
        ignored_layers = remove_ignored_layers(model_state_dict, config.model.checkpoint_ignore_layres)

        if checkpoint_format == "lightning" and len(ignored_layers) == 0:
            resume_ckpt_path = config.model.checkpoint_path
            print("Resuming full Lightning checkpoint:", resume_ckpt_path)
        else:
            if checkpoint_format == "lightning":
                print("Loading model weights from Lightning checkpoint without optimizer state because ignored layers were requested:", ignored_layers)
            else:
                print("Loading model weights from state_dict checkpoint:", config.model.checkpoint_path)
            model.model.load_state_dict(model_state_dict, strict=config.model.strict_checkpoint)
    
    # model.model = torch.compile(model.model)
    
    # Create logger.
    if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
        wandb_logger = WandbLogger(name=experiment_name+config.training.notes, project="mt3-score-pytorch-debug", offline=config.training.debug_log_offline, save_dir=log_dir) #, rank_zero_only=True
    else:
        wandb_logger = WandbLogger(name=experiment_name+config.training.notes, project="AMT-audio-to-midi", offline=False, save_dir=log_dir, notes=config.training.notes) #, rank_zero_only=True
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir)

    # Save informations to log dir.
    if model.global_rank == 0:
        # save model architecture to txt
        os.makedirs(log_dir, exist_ok=True)
        with open(log_dir + "/model_architecture.txt", "w") as f:
            f.write(str(config) + "\n"*5)
            f.write("\n".join([socket.gethostname(), " "]))
            f.write(str(model))

        # image dir.
        img_dir = os.path.join(log_dir, "img")
        os.system('mkdir -p "%s"'%img_dir)
        score_dir = os.path.join(log_dir, "score")
        os.system('mkdir -p "%s"'%score_dir)
        config

        # save source code
        if False:
            src_path = os.path.join(log_dir, "src").replace("'", "\\'")
            os.makedirs(src_path, exist_ok=True)
            OmegaConf.save(config, src_path+"/experiment_config.yaml")
            os.system("rsync -av  --exclude='*/runs-debug/' --exclude='*/runs/' --include='*/' --prune-empty-dirs --progress --include='*.py' --exclude='*' $(pwd) %s"%src_path)
            os.system("zip -r %s/src.zip %s"%(log_dir, src_path))
            os.system("rm -rf %s"%src_path) # remove src dir.

        # save config yaml.
        OmegaConf.save(config, log_dir+"/experiment_config.yaml")
        
        config_dict = OmegaConf.to_container(config)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            config_dict["training"]["cuda"]= os.environ["CUDA_VISIBLE_DEVICES"]
        wandb_logger.log_hyperparams(config_dict)
    
    logger_list = [wandb_logger, tensorboard_logger]
    if config.training.mode != "train":
        logger_list = []
    trainer = pl.Trainer(
                        devices=config.devices, # 1 [1,2, 4, 5, 6,7]
                        accelerator=config.accelerator, # "gpu"
                        logger=logger_list,
                        #  val_check_interval=0.0, # 0.0:disable, None:total training batch.
                        check_val_every_n_epoch=config.training.evaluation_epochs,
                        max_steps=config.training.training_steps,
                        reload_dataloaders_every_n_epochs=5,
                        log_every_n_steps = 50,
                        #  strategy="dp",
                        # strategy='ddp_find_unused_parameters_true'
                        gradient_clip_val=0.5,
                        gradient_clip_algorithm="value",
                        # callbacks=[val_call_back,]
                        strategy=DDPStrategy(find_unused_parameters=True),
                        )

    trainer.fit(model, ckpt_path=resume_ckpt_path)
    
if __name__ == "__main__":
    my_main()
