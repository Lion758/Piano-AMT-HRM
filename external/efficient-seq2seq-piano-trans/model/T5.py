"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""
#HRM_REFINERS
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import warnings
from model.HRM_refiners import HRMEncoderAdapter


from model.Encoder import Encoder
from model.Decoder import Decoder, CompoundDecoder
from model.Layers import *
from model.Mask import *
from model.HPPNet import HPPNet
from data.constants import *

from torch.nn.utils.rnn import pad_sequence
from config.utils import DictToObject

from tqdm import tqdm
from utils.log_memory_usage import profile_cuda_memory
from model.HRM_refiners import HRMEncoderCarry

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        config =DictToObject(dict(config))
        config.dtype = eval(config.dtype)
        # config.dtype = torch.float32 # 
        self.config = config
        # select encoder
        if config.encoder_name == "TransformerEncoder":
            self.encoder = Encoder(config)
        elif config.encoder_name == "CNNEncoder":
            self.encoder = CNNEncoder(config)
        elif config.encoder_name == "HPPNetEncoder":
            self.encoder = HPPNet(config)
        elif config.encoder_name == "hFT_Transformer_Encoder":
            self.encoder = hFT_Transformer_Encoder(config)
        else:
            raise "Unknown encoder: " + config.encoder_name
        # froze encoder
        if config.froze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        

        # select decoder
        # self.decoder = Decoder(config=config)
        sub_token_names = [ t.get_class_name() for t in sm_tokenizer.token_type_list]
        
        if config.decoder_name == "TransformerDecoder":
            self.decoder = Decoder(config=config)
        elif config.decoder_name == "CompoundTransformerDecoder":
            self.decoder = CompoundDecoder(config=config, sub_token_names=sub_token_names)

        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END

        self.use_hrm_refiner = getattr(self.config, "use_hrm_refiner", False)
        self.hrm_refine_scales = set(getattr(self.config, "hrm_refine_scales", [4]))
        self.hrm_ablation_pooling_sets = getattr(self.config, "hrm_ablation_pooling_sets", [])
        self.hrm_active_ablation_idx = getattr(self.config, "hrm_active_ablation_idx", None)
        if self.hrm_ablation_pooling_sets and self.hrm_active_ablation_idx is not None:
            selected_scales = self.hrm_ablation_pooling_sets[self.hrm_active_ablation_idx]
            self.hrm_refine_scales = set(selected_scales)
        self.hrm_cross_scale_coupling = getattr(self.config, "hrm_cross_scale_coupling", False)
        self.hrm_cross_scale_bidirectional = getattr(self.config, "hrm_cross_scale_bidirectional", True)

        self.hrm_refiners = nn.ModuleDict()
        if self.use_hrm_refiner and hasattr(self.config, "pooling_sizes"):
            for p in self._ordered_poolings(self.config.pooling_sizes, order="coarse_to_fine"):
                if p in self.hrm_refine_scales:
                    self.hrm_refiners[str(p)] = HRMEncoderAdapter(
                        d_model=self.config.emb_dim,
                        H_cycles=getattr(self.config, "hrm_H_cycles", 2),
                        L_cycles=getattr(self.config, "hrm_L_cycles", 2),
                        H_layers=getattr(self.config, "hrm_H_layers", 4),
                        L_layers=getattr(self.config, "hrm_L_layers", 4),
                        num_heads=getattr(self.config, "hrm_num_heads", 8),
                        expansion=getattr(self.config, "hrm_expansion", 4.0),
                        rms_norm_eps=getattr(self.config, "hrm_rms_norm_eps", 1e-5),
                        use_rope=getattr(self.config, "hrm_use_rope", True),
                        forward_dtype=eval(getattr(self.config, "hrm_forward_dtype", "torch.bfloat16")),
                        max_seq_len=getattr(self.config, "max_len", 4096),
                        one_step_grad=getattr(self.config, "hrm_one_step_grad", True),
                        halt_max_steps=getattr(self.config, "hrm_halt_max_steps", 1),
                        halt_exploration_prob=getattr(self.config, "hrm_halt_exploration_prob", 0.0),
                        use_act_halt=getattr(self.config, "hrm_use_act_halt", False),
                    )
        self.hrm_carry_dict = {}
        self._hrm_carry_warned_scales = set()
        self.hrm_cross_scale_proj = nn.ModuleDict()
        if self.hrm_cross_scale_coupling and self.use_hrm_refiner:
            coupling_scales = [
                p for p in self._ordered_poolings(getattr(self.config, "pooling_sizes", []), order="coarse_to_fine")
                if str(p) in self.hrm_refiners
            ]
            for idx in range(len(coupling_scales) - 1):
                coarse = coupling_scales[idx]
                fine = coupling_scales[idx + 1]
                self.hrm_cross_scale_proj[f"{coarse}_to_{fine}"] = nn.Linear(self.config.emb_dim, self.config.emb_dim)
                if self.hrm_cross_scale_bidirectional:
                    self.hrm_cross_scale_proj[f"{fine}_to_{coarse}"] = nn.Linear(self.config.emb_dim, self.config.emb_dim)

    def reset_hrm_carry(self, batch_size=None):
        if batch_size is None:
            self.hrm_carry_dict = {}
            self._hrm_carry_warned_scales.clear()
            return

        keep_batch = int(batch_size)
        new_carry_dict = {}
        for pooling, carry in self.hrm_carry_dict.items():
            if carry is None:
                continue
            if carry.z_H.size(0) == keep_batch:
                new_carry_dict[pooling] = carry
        self.hrm_carry_dict = new_carry_dict
        self._hrm_carry_warned_scales.clear()

    def detach_hrm_carry(self):
        detached = {}
        for pooling, carry in self.hrm_carry_dict.items():
            if carry is None:
                continue
            detached[pooling] = HRMEncoderCarry(
                z_H=carry.z_H.detach(),
                z_L=carry.z_L.detach(),
                steps=carry.steps.detach(),
                halted=carry.halted.detach(),
                external_context=None if carry.external_context is None else carry.external_context.detach(),
            )
        self.hrm_carry_dict = detached

    def clear_hrm_carry_on_device_change(self):
        if not self.hrm_carry_dict:
            return
        model_device = next(self.parameters()).device
        for carry in self.hrm_carry_dict.values():
            if carry is None:
                continue
            if carry.z_H.device != model_device:
                self.reset_hrm_carry()
                return

    def train(self, mode: bool = True):
        was_training = self.training
        result = super().train(mode)
        if (was_training and not mode) and getattr(self.config, "hrm_auto_reset_on_eval", False):
            self.reset_hrm_carry()
        return result
        
        
    def encode(
            self, 
            encoder_input_tokens, 
            encoder_segment_ids=None, 
            enable_dropout=True
            ):
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)

        encoder_mask = make_attention_mask(
            torch.ones(encoder_input_tokens.shape[:-1]),
            torch.ones(encoder_input_tokens.shape[:-1]),
            dtype=self.config.dtype
        )

        if encoder_segment_ids is not None:
            encoder_mask = combine_masks(
                encoder_mask,
                make_attention_mask(
                    encoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )
        
        encoder_mask = encoder_mask.to(encoder_input_tokens.device)
            
        if self.config.froze_encoder:
            with torch.no_grad():
                encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)
                encoder_outputs = encoder_outputs.detach()
        else:
            encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)
        return encoder_outputs
    
    def set_requires_grad(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_encoder(self): self.set_requires_grad(self.encoder, False)
    def unfreeze_encoder(self): self.set_requires_grad(self.encoder, True)

    def freeze_decoder(self): self.set_requires_grad(self.decoder, False)
    def unfreeze_decoder(self): self.set_requires_grad(self.decoder, True)

    def freeze_hrm(self): self.set_requires_grad(self.hrm_refiners, False)
    def unfreeze_hrm(self): self.set_requires_grad(self.hrm_refiners, True)

    def _ordered_poolings(self, pooling_sizes, order="coarse_to_fine"):
        ordered = sorted({int(p) for p in pooling_sizes}, reverse=(order == "coarse_to_fine"))
        return ordered

    def set_hrm_ablation_pooling_set(self, pooling_sizes):
        self.hrm_refine_scales = set(int(p) for p in pooling_sizes)

    def _resize_context_to_length(self, context: torch.Tensor, target_len: int) -> torch.Tensor:
        if context.size(1) == target_len:
            return context
        context_t = context.transpose(1, 2)
        resized = Functional.interpolate(context_t, size=target_len, mode="nearest")
        return resized.transpose(1, 2)

    def _compute_encoder_valid_mask(self, encoder_input_tokens: torch.Tensor, encoder_segment_ids=None) -> torch.Tensor:
        valid_mask = encoder_input_tokens.abs().sum(dim=-1) > 0
        if encoder_segment_ids is not None:
            valid_mask = valid_mask & (encoder_segment_ids > 0)
        return valid_mask

    def _pool_valid_mask(self, source_mask: torch.Tensor, pooling: int, target_len: int) -> torch.Tensor:
        if pooling <= 1:
            return source_mask[:, :target_len]
        reshaped = source_mask[:, :target_len * pooling].reshape(source_mask.size(0), target_len, pooling)
        return reshaped.any(dim=-1)

    def _run_hierarchy_refinement(self, encoded_source, reset_flag=None, source_mask=None):
        encoded_pooling_dict = {}
        hrm_halting = {}
        if not (hasattr(self.config, "cross_attention_hierarchy_pooling") and self.config.cross_attention_hierarchy_pooling):
            return encoded_source, encoded_pooling_dict, hrm_halting

        ordered_poolings = self._ordered_poolings(self.config.pooling_sizes, order="coarse_to_fine")
        if source_mask is None:
            source_mask = torch.ones(
                (encoded_source.size(0), encoded_source.size(1)),
                dtype=torch.bool,
                device=encoded_source.device,
            )
        else:
            source_mask = source_mask.to(device=encoded_source.device, dtype=torch.bool)

        summary_by_pooling = {}
        for idx, pooling in enumerate(ordered_poolings):
            encoded_pool = encoded_source
            if pooling > 1:
                encoded_pool = encoded_pool.reshape(
                    encoded_source.size(0),
                    encoded_source.size(1) // pooling,
                    pooling,
                    encoded_source.size(2),
                ).mean(dim=2)

            pooled_mask = self._pool_valid_mask(source_mask, pooling=pooling, target_len=encoded_pool.size(1))

            external_context = None
            if self.hrm_cross_scale_coupling and str(pooling) in self.hrm_refiners and idx > 0:
                coarse_pooling = ordered_poolings[idx - 1]
                if coarse_pooling in summary_by_pooling:
                    proj_key = f"{coarse_pooling}_to_{pooling}"
                    if proj_key in self.hrm_cross_scale_proj:
                        coarse_ctx = self.hrm_cross_scale_proj[proj_key](summary_by_pooling[coarse_pooling])
                        external_context = self._resize_context_to_length(coarse_ctx, encoded_pool.size(1))

            encoded_i = encoded_pool
            if self.use_hrm_refiner and str(pooling) in self.hrm_refiners:
                carry = self.hrm_carry_dict.get(pooling)
                if carry is not None and (
                    carry.z_H.size(0) != encoded_pool.size(0)
                    or carry.z_H.size(1) != encoded_pool.size(1)
                ):
                    if pooling not in self._hrm_carry_warned_scales:
                        warnings.warn(
                            (
                                f"HRM carry shape mismatch at pooling={pooling}: "
                                f"carry={(carry.z_H.size(0), carry.z_H.size(1))}, "
                                f"expected={(encoded_pool.size(0), encoded_pool.size(1))}. "
                                "Resetting HRM carry."
                            ),
                            stacklevel=2,
                        )
                        self._hrm_carry_warned_scales.add(pooling)
                    self.reset_hrm_carry()
                    carry = None
                assert carry is None or (
                    carry.z_H.size(0) == encoded_pool.size(0)
                    and carry.z_H.size(1) == encoded_pool.size(1)
                ), f"HRM carry mismatch after reset for pooling={pooling}"
                encoded_i, new_carry, halting_diag = self.hrm_refiners[str(pooling)](
                    x_pool=encoded_pool,
                    carry=carry,
                    reset_flag=reset_flag,
                    external_context=external_context,
                    pool_mask=pooled_mask,
                )
                self.hrm_carry_dict[pooling] = new_carry
                hrm_halting[pooling] = halting_diag
                summary_by_pooling[pooling] = new_carry.z_H
            else:
                encoded_i = encoded_i * pooled_mask.unsqueeze(-1).to(encoded_i.dtype)
                summary_by_pooling[pooling] = encoded_i

            if (
                self.hrm_cross_scale_coupling
                and self.hrm_cross_scale_bidirectional
                and idx > 0
            ):
                coarse_pooling = ordered_poolings[idx - 1]
                reverse_key = f"{pooling}_to_{coarse_pooling}"
                if reverse_key in self.hrm_cross_scale_proj:
                    finer_summary = summary_by_pooling[pooling].mean(dim=1, keepdim=True)
                    projected_finer = self.hrm_cross_scale_proj[reverse_key](finer_summary)
                    coarse_context = self._resize_context_to_length(projected_finer, summary_by_pooling[coarse_pooling].size(1))
                    summary_by_pooling[coarse_pooling] = summary_by_pooling[coarse_pooling] + coarse_context
                    if coarse_pooling in self.hrm_carry_dict:
                        self.hrm_carry_dict[coarse_pooling].external_context = coarse_context.detach()

            encoded_pooling_dict[pooling] = encoded_i * pooled_mask.unsqueeze(-1).to(encoded_i.dtype)

        missing_poolings = set(self.config.pooling_sizes) - set(encoded_pooling_dict.keys())
        if missing_poolings:
            raise ValueError(f"encoded_pooling_dict missing pooling sizes: {sorted(missing_poolings)}")

        return None, encoded_pooling_dict, hrm_halting

    
    def decode(
            self, 
            encoded, 
            encoder_input_tokens, 
            decoder_input_tokens, 
            decoder_target_tokens, 
            encoder_segment_ids=None, 
            decoder_segment_ids=None, 
            decoder_positions=None, 
            enable_dropout=True, 
            decode=False, #decode: Whether to prepare and use an autoregressive cache
            max_decode_length=None,
            use_preframe_tokens=True,
            decoder_targets_frame_index=None,
            encoder_decoder_mask=None,
            hrm_reset_flag=None,
            ):
        
        encoder_decoder_mask_0 = encoder_decoder_mask
        self.clear_hrm_carry_on_device_change()
            
        # We use pooling to reduce decoder sequence length.
        # decoder_target_tokens here just use for generating decoder_mask and attention_mask.
        B, T = decoder_target_tokens.size()
        decoder_target_tokens = decoder_target_tokens #[:, :T//SEQUENCE_POOLING_SIZE]


        if decode:
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                torch.ones_like(decoder_target_tokens).to(encoded.device),
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype
            )
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=self.config.dtype,
                decoder_segment_ids=decoder_segment_ids
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0,
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype
            )

        if encoder_segment_ids is not None:
            if decode:
                raise ValueError('During decoding, packing should not be used but `encoder_segment_ids` was passed to `Transformer.decode`.')

            encoder_decoder_mask = combine_masks(
                encoder_decoder_mask,
                make_attention_mask(
                    decoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )
            
        
        # If the encoder_decoder_mask is provided, we use the default mask.
        if encoder_decoder_mask_0 is not None:
            if hasattr(self.config, "encoder_decoder_slide_window_size") and self.config.encoder_decoder_slide_window_size > 0:
                encoder_decoder_mask = encoder_decoder_mask_0
                
        encoded_pooling_dict = {}
        hrm_halting = {}
        reset_flag = None
        encoder_valid_mask = self._compute_encoder_valid_mask(encoder_input_tokens, encoder_segment_ids=encoder_segment_ids).to(encoded.device)
        if hrm_reset_flag is not None:
            reset_flag = hrm_reset_flag.to(encoded.device)

        # Hierarchical pooling
        encoded, encoded_pooling_dict, hrm_halting = self._run_hierarchy_refinement(
            encoded,
            reset_flag=reset_flag,
            source_mask=encoder_valid_mask,
        )


        decoder_output_dict = self.decoder(
            encoded,
            encoded_pooling_dict=encoded_pooling_dict,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            decoder_targets_frame_index=decoder_targets_frame_index,
            )
        if hrm_halting:
            decoder_output_dict["hrm_halting"] = hrm_halting

            # Optionally expose ACT diagnostics in the top-level output for training.
            # We prioritize pooling=1 when present, otherwise use the smallest pooling scale.
            preferred_pooling = 1 if 1 in hrm_halting else min(hrm_halting.keys())
            act_diag = hrm_halting[preferred_pooling]
            if "q_halt_logits" in act_diag:
                decoder_output_dict["hrm_q_halt_logits"] = act_diag["q_halt_logits"]
            if "q_continue_logits" in act_diag:
                decoder_output_dict["hrm_q_continue_logits"] = act_diag["q_continue_logits"]
            if "target_q_continue" in act_diag:
                decoder_output_dict["hrm_target_q_continue"] = act_diag["target_q_continue"]
            if "steps" in act_diag:
                decoder_output_dict["hrm_steps"] = act_diag["steps"]
        return decoder_output_dict
    
    def _shift_right(self, input_ids, shift_step=1):
        BOS = TOKEN_START

        shifted_input_ids = torch.zeros_like(input_ids)
        if shift_step is None:
            shift_step = SEQUENCE_POOLING_SIZE
        shifted_input_ids[..., shift_step:] = input_ids[..., :-shift_step].clone()
        shifted_input_ids[..., :shift_step] = BOS

        return shifted_input_ids
    
    def forward(self, 
                encoder_input_tokens=None, 
                decoder_target_tokens=None, 
                decoder_input_tokens=None, 
                encoder_segment_ids=None, 
                decoder_segment_ids=None, 
                encoder_positions=None, 
                decoder_positions=None, 
                enable_dropout=True, 
                decode=False,
                dur_inputs = None,
                dur_targets = None,
                decoder_targets_frame_index=None,
                encoder_decoder_mask=None,
                hrm_reset_flag=None,
    ):
        res_dict = {}
            
        if decoder_input_tokens == None:
            decoder_input_tokens = self._shift_right(decoder_target_tokens, shift_step=1)
            
        encoder_outputs = self.encode(encoder_input_tokens, encoder_segment_ids=encoder_segment_ids, enable_dropout=enable_dropout)
        
        decoder_output_dict = self.decode(encoder_outputs, encoder_outputs, decoder_input_tokens, decoder_target_tokens, encoder_segment_ids=encoder_segment_ids, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, enable_dropout=enable_dropout, decode=decode, 
            decoder_targets_frame_index=decoder_targets_frame_index,
            encoder_decoder_mask=encoder_decoder_mask,
            hrm_reset_flag=hrm_reset_flag)


        res_dict.update(decoder_output_dict)

        return res_dict

    def generate_0(self, primer=None, target_seq_length=1024):
        num_primer = len(primer)
        len_primer = len(primer[0])
        # -> [num_frame x vec_size]
        gen_tokens = torch.LongTensor([self.pad_token for i in range(target_seq_length-len_primer)]).expand(num_primer, target_seq_length-len_primer)
        gen_tokens = torch.concat((primer.type(torch.long), gen_tokens.device(primer.device)), dim=-1)
        

        i = num_primer
        while (i < target_seq_length):
            logits, _ = self.forward(gen_tokens[..., :i], decode=True)
            probs = self.softmax(logits)[..., :self.eos_token]
            token_probs = probs[:, i - 1, :]

            next_token = torch.argmax(token_probs)
            gen_tokens[:, i] = next_token

            if next_token == self.eos_token:
                break
            i += 1

        return gen_tokens[:, :i]
    
    # @profile_cuda_memory
    def generate(self, encoder_inputs, target_seq_length = 1024, berak_on_eos=False, global_rank=0):
        self.reset_hrm_carry(batch_size=encoder_inputs.size(0))
        self.decoder.initialize_decoder_cache()
        
        batch_size, T, n_mel = encoder_inputs.size()
        gen_tokens = torch.ones([batch_size, target_seq_length]).to(encoder_inputs) * TOKEN_PAD
        eos_flags = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        curr_frame_index = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        max_num_tokens = 0
        encoded = self.encode(encoder_inputs, enable_dropout=False)
        encoder_valid_mask = self._compute_encoder_valid_mask(encoder_inputs).to(encoded.device)
        decoder_mask = torch.tril(torch.ones((target_seq_length, target_seq_length), dtype=self.config.dtype), diagonal=0).to(encoder_inputs.device)
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_heads, -1, -1)  # [batch_size, 1, target_seq_length, target_seq_length]
        encoder_decoder_mask = torch.ones((batch_size, self.config.num_heads, target_seq_length, T), dtype=self.config.dtype).to(encoder_inputs.device)  # [batch_size, 1, target_seq_length, T]
        
        pred_step = 1
        if hasattr(self.config, "predict_onset_by_AttnMap") and self.config.predict_onset_by_AttnMap:
            pred_step = 2
            
        use_kv_cache = True
        curr_token = torch.ones([batch_size, pred_step]).to(encoder_inputs) * TOKEN_START  # [batch_size, pred_step]
        hrm_reset_flag = torch.ones(batch_size, dtype=torch.bool, device=encoder_inputs.device)
        for i in tqdm(range(0, target_seq_length, pred_step), desc="Generating tokens (rank %d)" % global_rank):
            encoded_i = encoded[:, :T, :]  # [batch_size, T, n_mel]
            encoder_valid_mask_i = encoder_valid_mask[:, :T]
            
            if use_kv_cache:
                decoder_input_tokens_i = curr_token # 
                encoder_decoder_mask_i = encoder_decoder_mask[:, :, i:i+pred_step, :T]
                decoder_mask_i = decoder_mask[:, :, i:i+pred_step, :i+pred_step]
                decoder_positions_i = torch.arange(i, i+pred_step, device=encoder_inputs.device).unsqueeze(0).expand([batch_size, -1])  # [1, 1, pred_step]
            else:
                decoder_input_tokens_i = self._shift_right(gen_tokens[:, :i+pred_step], shift_step=1)  # [batch_size, i+pred_step]
                encoder_decoder_mask_i = encoder_decoder_mask[:, :, :i+pred_step, :T]  # [batch_size, 1, i+pred_step, T]
                decoder_mask_i = decoder_mask[:, :, :i+pred_step, :i+pred_step]
                decoder_positions_i = None
            

            # Slide window for encoder-decoder attention
            if i % 3 != 0 and hasattr(self.config, "encoder_decoder_slide_window_size") and self.config.encoder_decoder_slide_window_size > 0:
                enc_len, emb_dim = encoded.size(1), encoded.size(2)
                encoded_fold = encoded.view(batch_size, -1, self.config.encoder_decoder_slide_window_size, emb_dim)  # [batch_size, num_windows, window_size, emb_dim]
                # => [batch_size, 1]
                curr_window_index = torch.floor(curr_frame_index // self.config.encoder_decoder_slide_window_size).int()#.unsqueeze(0)  # [batch_size, 1]
                batch_indices = torch.arange(batch_size, device=encoded.device) #.unsqueeze(1)  # [batch_size, 1]
                encoded_i = encoded_fold[batch_indices, curr_window_index, :, :]  # [batch_size, 1, window_size, emb_dim]
                # print(encoded_i.size())
                encoded_i = encoded_i.view(batch_size, self.config.encoder_decoder_slide_window_size, emb_dim)  # [batch_size, window_size, emb_dim]

                mask_fold = encoder_valid_mask.view(batch_size, -1, self.config.encoder_decoder_slide_window_size)
                encoder_valid_mask_i = mask_fold[batch_indices, curr_window_index, :].view(batch_size, self.config.encoder_decoder_slide_window_size)

                # encoder_decoder_mask_i = torch.stack(encoder_decoder_mask_i_list, dim=0)
                encoder_decoder_mask_i = encoder_decoder_mask_i[:, :, :, :self.config.encoder_decoder_slide_window_size]  # [batch_size, 1, i+pred_step, T]
            
            encoded_i, encoded_pooling_dict, _ = self._run_hierarchy_refinement(
                encoded_i,
                reset_flag=hrm_reset_flag,
                source_mask=encoder_valid_mask_i,
            )
            
            decoder_output_dict = self.decoder(
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
            probs = torch.softmax(logits[:, -pred_step:, :], dim=-1)[..., :self.pad_token]
            
            # Check if the output sequence is in the expected order: onset, pitch/noteOff, velocity
            if hasattr(self.config, "hybrid_global_local_cross_attn") and self.config.hybrid_global_local_cross_attn:
                tokens_ori = torch.argmax(probs, dim=-1)
                eos_flags_int = eos_flags.int()
                for b in range(batch_size):
                    token_i = tokens_ori[b, 0].item()
                    if eos_flags_int[b] == 1:
                        continue
                    if Tokenizer.TokenOnset.is_instance(token_i) or token_i == self.eos_token:
                        token_type_i = Tokenizer.TokenOnset
                    elif Tokenizer.TokenPitch.is_instance(token_i):
                        token_type_i = Tokenizer.TokenPitch
                    elif Tokenizer.TokenVel.is_instance(token_i) or token_i == TOKEN_BLANK:
                        token_type_i = Tokenizer.TokenVel
                    elif Tokenizer.TokenNoteOff.is_instance(token_i):
                        token_type_i = Tokenizer.TokenNoteOff
                    else:
                        token_type_i = "Other"
                    if i % 3 == 0:
                        dest_token_type_i = (Tokenizer.TokenOnset,)
                    elif i % 3 == 1:
                        dest_token_type_i = (Tokenizer.TokenPitch, Tokenizer.TokenNoteOff)
                    elif i % 3 == 2:
                        prev_token_i = gen_tokens[b, i-1].item()
                        if Tokenizer.TokenNoteOff.is_instance(prev_token_i) and token_i != TOKEN_BLANK:
                            # If the previous token is a NoteOff, then the current velocity should be TOKEN_BLANK
                            print(f"Warning: The generated token {token_i} at batch idx {b} position {i} is a velocity token, but the previous token is a NoteOff. The velocity should be {TOKEN_BLANK}, but got {token_i}.")
                        dest_token_type_i = (Tokenizer.TokenVel, )  
                    if not token_type_i in dest_token_type_i:
                        print(f"Warning: The generated token {token_i} at batch idx {b} position {i} is not in the expected format. Expected: {dest_token_type_i}, but got: {token_type_i}.")

                

            if hasattr(self.config, "hybrid_global_local_cross_attn") and self.config.hybrid_global_local_cross_attn:
                # Force the output sequence to in the format of onset, pitch, and velocity
                if i % 3 == 0: # Onset or EOS token
                    begin,end = Tokenizer.TokenOnset.get_bound()
                    probs[:, :, begin:end] += 1
                    probs[:, :, self.eos_token] += 1
                elif i % 3 == 1: # Pitch token, NoteOn or NoteOff
                    begin, end = Tokenizer.TokenPitch.get_bound()
                    probs[:, :, begin:end] += 1
                    begin, end = Tokenizer.TokenNoteOff.get_bound()
                    probs[:, :, begin:end] += 1
                elif i % 3 == 2: # Velocity token
                    begin, end = Tokenizer.TokenVel.get_bound()
                    probs[:, :, begin:end] += 1
                    probs[:, :, TOKEN_BLANK] += 1  # Force the velocity token to be a valid token
            
            
            token_probs = probs
            curr_token = torch.argmax(token_probs, dim=-1) # [batch_size, pred_step]
            gen_tokens[:, i:i+pred_step] = curr_token
            max_num_tokens += pred_step
            # Check  EOS
            eos_flags = eos_flags | (curr_token[:, 0] == self.eos_token)
            if berak_on_eos and eos_flags.int().sum() == batch_size:
                break
            if hrm_reset_flag is not None:
                hrm_reset_flag = torch.zeros_like(hrm_reset_flag)

            # Update current frame index
            gen_onsets = curr_token[:, 0]
            frames_per_second = DEFAULT_SAMPLE_RATE  / DEFAULT_HOP_WIDTH
            eos_flags_int = eos_flags.int()
            for b in range(batch_size):
                onset_i = gen_onsets[b].item()
                if eos_flags_int[b] == 1:
                    curr_frame_index[b] = T - 1
                    continue
                if Tokenizer.TokenOnset.is_instance(onset_i): # Check if it is a onset token
                    onset_val = Tokenizer.TokenOnset.get_value(onset_i)
                    onset_sec = onset_val / Tokenizer.ONSET_SEC_UP_SAMPLING
                    onset_frame_index = int(onset_sec * frames_per_second)
                    curr_frame_index[b] = min(T-1, onset_frame_index)

        return gen_tokens #[:, :max_num_tokens]

    
   
