import os
import sys
work_dir = os.path.split(__file__)[0] + "/../"
import torch
import numpy as np
import data.notes_parser as notes_parser
from data.constants import *
from symusic import Score, TimeUnit
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn import metrics


from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity

import utils.pianoroll_parser as pianoroll_parser

from data.notes_parser import frame_tokens_to_interval_and_pitch
from data.pedal_extension_utils import (
    PEDAL_MIN_DURATION,
    infer_piece_end_time as shared_infer_piece_end_time,
    pedal_events_to_spans as shared_pedal_events_to_spans,
    extend_notes_with_pedal_events,
    extend_notes_with_reference_pedal_events,
)

PEDAL_DUMMY_MIDI = 21
PEDAL_EVENT_TOLERANCE = 0.05
REFERENCE_PEDAL_ONSET_TOLERANCE = 0.2
REFERENCE_PEDAL_OFFSET_RATIO = 0.2
REFERENCE_PEDAL_OFFSET_MIN_TOLERANCE = 0.05

def pianoroll_frame_metrics(name, output_pianoroll_frame, target_pianoroll_frame):
    if (target_pianoroll_frame == 1).sum() == 0:
        return {}
    assert target_pianoroll_frame.max() <= 1 and target_pianoroll_frame.min() >= 0
    assert output_pianoroll_frame.max() <= 1 and output_pianoroll_frame.min() <= 0
    P = (target_pianoroll_frame == 1).sum()
    output_pianoroll_frame = (output_pianoroll_frame == 1).long()
    target_pianoroll_frame = (target_pianoroll_frame == 1).long()
    TP = torch.mul(output_pianoroll_frame, target_pianoroll_frame).sum()
    FP = torch.mul(output_pianoroll_frame, 1 - target_pianoroll_frame).sum()
    p = 0
    if TP + FP > 0:
        p = TP / (FP + TP)
    r = TP/P
    f1 = 0
    if p+r > 0:
        f1 = 2 * p * r /(p + r)
    
    return {
        name + "_p": p,
        name + "_r": r,
        name + "_f1": f1,
    }


def pianoroll_onset_metrics(name, output_pianoroll_onset, target_pianoroll_onset, threshold = 0.5, times_interval = 0.02):
    """_summary_

    Args:
        output_pianoroll_onset (_type_): shape [T, F] or [B, T, F]
        target_pianoroll_onset (_type_): shape [T, F] or [B, T, F]
    """
    if output_pianoroll_onset.ndim == 3:
        B,T,F = output_pianoroll_onset.size()
        output_pianoroll_onset = output_pianoroll_onset.reshape([B*T, F])
        target_pianoroll_onset = target_pianoroll_onset.reshape([B*T, F])
    output_pianoroll_onset = (output_pianoroll_onset >= threshold).long()

    output_pianoroll_onset = pianoroll_parser.get_onset_pianoroll(output_pianoroll_onset)
    target_pianoroll_onset = pianoroll_parser.get_onset_pianoroll(target_pianoroll_onset)

    output_times, output_onsets = pianoroll_parser.onset_pianoroll_to_note_list(output_pianoroll_onset)
    target_times, target_onsets = pianoroll_parser.onset_pianoroll_to_note_list(target_pianoroll_onset)
    output_times = np.array(output_times) * times_interval
    target_times = np.array(target_times) * times_interval
    output_hz = midi_to_hz(output_onsets)
    target_hz = midi_to_hz(target_onsets)

    num = len(target_onsets)

    output_intervals = np.stack([output_times, output_times + 0.2], axis=1)
    target_intervals = np.stack([target_times, target_times + 0.2], axis=1)
    p, r, f, o = evaluate_notes(target_intervals, target_hz, output_intervals, output_hz, offset_ratio=None, onset_tolerance=0.05)
    print("%s: p=%.3f, r=%.3f, f1=%.3f, n=%d"%(name, p, r, f, num))
    return {
        name + "_p": p,
        name + "_r": r,
        name + "_f1": f,
    }


def cal_frames_token_metrics(output_tokens, target_tokens, decoder_targets_tokens):
    """_summary_

    Args:
        output_tokens (torch.tensor.long): [B, n_tokens], n_tokens is the max token seq length.
        target_tokens (torch.tensor.long): [B, n_tokens]
        targets_delta_tokens (torch.tensor.long): [B, n_tokens]
    Returns:
        metric_dict (Dict):
    """
    metrics_dict = {}
    
    output_tokens = torch.clone(output_tokens)
    # self.log_time_event("metrics_cal_begin")
    B, n_tokens = output_tokens.size()

    # # set tokens after eos to TOKEN_PAD
    # eos_mask = torch.zeros([B], device=output_tokens.device).bool()
    # for i in range(n_tokens):
    #     output_tokens[eos_mask, i] = TOKEN_PAD
    #     eos_mask = eos_mask | (output_tokens[:, i] == TOKEN_END)
    #     # eos_mask = eos_mask | (output_tokens[:, i] >= 256) 

    output_onset_tokens = torch.clone(output_tokens)
    # output_onset_tokens[(output_onset_tokens > MIDI_MAX) | (output_onset_tokens < MIDI_MIN)] = TOKEN_BLANK
    
    output_intervals, output_pitches = frame_tokens_to_interval_and_pitch(output_onset_tokens)
    # notes_dict["output_intervals"] = output_intervals
    # notes_dict["output_pitches"] = output_pitches
    target_intervals, target_pitches = frame_tokens_to_interval_and_pitch(target_tokens)
    # notes_dict["target_intervals"] = target_intervals
    # notes_dict["target_pitches"] = target_pitches
    # => [128, BT]

    # cal metrics
    # frame idx to second.
    output_intervals_second = output_intervals * DEFAULT_HOP_WIDTH/DEFAULT_SAMPLE_RATE
    target_intervals_second = target_intervals * DEFAULT_HOP_WIDTH/DEFAULT_SAMPLE_RATE
    output_pitches_hz = midi_to_hz(output_pitches)
    target_pitches_hz = midi_to_hz(target_pitches)
    if len(target_pitches) > 0 and len(output_pitches) > 0:
        p, r, f, o = evaluate_notes(target_intervals_second, target_pitches_hz, output_intervals_second, output_pitches_hz, offset_ratio=None, onset_tolerance=0.05)
        # self.log_time_event("metrics_cal_done")
        num = len(target_pitches)
        
        metrics_dict["onset_p"] = p
        metrics_dict["onset_r"] = r
        metrics_dict["onset_f1"] = f
        metrics_dict["onset_num"] = num
        print("Onset: p=%.3f, r=%.3f, f1=%.3f, n=%d, n_tokens=%d"%(p,r,f,num, n_tokens))

        p, r, f, o = evaluate_notes(target_intervals_second, target_pitches_hz, output_intervals_second, output_pitches_hz, onset_tolerance=0.05)
        metrics_dict["note_p"] = p
        metrics_dict["note_r"] = r
        metrics_dict["note_f1"] = f
        print("Note: p=%.3f, r=%.3f, f1=%.3f"%(p,r,f))
    else:
        if len(target_pitches) == 0:
            print("Warning: len(target_pitches)=0.")
        else:
            print("Warning: len(output_pitches)=0.")

    # cal acc of tokens.
    output_tokens = output_tokens.view(-1)#.cpu().numpy()
    target_tokens = target_tokens.view(-1)#.cpu().numpy()
    decoder_targets_tokens = decoder_targets_tokens.view(-1)#.cpu().numpy()
    def event_acc(token_type, begin, end, output_tokens, target_tokens):
        # sel_idx = (target_tokens >= begin) &  (target_tokens < end)
        # output_tokens_sel = output_tokens[sel_idx]
        # target_tokens_sel =target_tokens[sel_idx]
        # if len(target_tokens_sel) == 0:
        #     return 0
        # return accuracy_score(target_tokens_sel, output_tokens_sel)
        sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
        sel_outputs = ((output_tokens >= begin) & (output_tokens < end))
        num = sel_targets.sum()
        if num == 0:
            return {}
        TP = (output_tokens[sel_targets & sel_outputs] == target_tokens[sel_targets & sel_outputs]).sum()

        FP = sel_outputs.sum() - TP
        if TP + FP == 0:
            precision= 0
        else:
            precision = TP/(TP+FP)
        recall = TP / num
        if (precision + recall) <= 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        d =  {
            "token_%s_p"%token_type:precision,
            "token_%s_r"%token_type:recall,
            "token_%s_f1"%token_type:f1,
        }
        if token_type == "on":
            print("Token_on: p=%.3f, r=%.3f, f1=%.3f"%(precision,recall,f1))
        return d

    metrics_dict.update( event_acc("all", 0, TOKEN_PAD, output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("on", 0, START_IDX["note_off"], output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("off", START_IDX["note_off"], START_IDX["time_shift"], output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("time", START_IDX["time_shift"], START_IDX["time_shift"]+128, output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("eos", TOKEN_END, TOKEN_END+1, output_tokens, decoder_targets_tokens) )

    return metrics_dict

def cal_onsets_and_frames_metrics(name, output_onset_pianoroll: torch.tensor, output_frame_pianoroll, target_onset_pianoroll, target_frame_pianoroll):
    """_summary_

    Args:
        output_onset_pianoroll (_type_): shape [B, T, F] binary value: 0, 1
        output_frame_pianoroll (_type_): _description_
        target_onset_pianoroll (_type_): _description_
        target_frame_pianoroll (_type_): _description_
    """
    assert target_frame_pianoroll.max() <= 1 and target_frame_pianoroll.min() >= 0
    assert output_onset_pianoroll.ndim == 3
    def get_offset_pianoroll(frame_pianoroll):
        offset_pianoroll  = torch.clone(frame_pianoroll)
        offset_pianoroll[:, : -1, :] = torch.clip(frame_pianoroll[:, :-1, :] - frame_pianoroll[:, 1:, :], 0, 1)
        offset_pianoroll[:, -1, :] = 1 # set the last frame as offsets.
        return offset_pianoroll

    output_offset_pianoroll = get_offset_pianoroll(output_frame_pianoroll)
    target_offset_pianoroll = get_offset_pianoroll(target_frame_pianoroll)
    B, T, F = output_onset_pianoroll.size()
    output_onset_pianoroll = output_onset_pianoroll.reshape([B*T, F])
    output_offset_pianoroll = output_offset_pianoroll.reshape([B*T, F])
    target_onset_pianoroll = target_onset_pianoroll.reshape([B*T, F])
    target_offset_pianoroll = target_offset_pianoroll.reshape([B*T, F])

    output_intervals, output_pitches_midi = notes_parser.pianoroll_to_notes_list(output_onset_pianoroll, output_offset_pianoroll)
    target_intervals, target_pitches_midi = notes_parser.pianoroll_to_notes_list(target_onset_pianoroll, target_offset_pianoroll)
    output_intervals = output_intervals * DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
    target_intervals = target_intervals * DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
    output_pitches = midi_to_hz(output_pitches_midi)
    target_pitches = midi_to_hz(target_pitches_midi)
    metrics_dict = {}
    dur_mask = output_intervals[:, 1] < output_intervals[:, 0] + 0.05 # Ensure the min duration is 50ms.
    output_intervals[:, 1][dur_mask] = (output_intervals[:, 0] + 0.05)[dur_mask]
    p, r, f, o = evaluate_notes(target_intervals, target_pitches, output_intervals, output_pitches, onset_tolerance=0.05)
    metrics_dict["%s_p"%name] = p
    metrics_dict["%s_r"%name] = r
    metrics_dict["%s_f1"%name] = f
    


    # Duration Error Analysis
    acc = -1
    err_dur_longer = -1
    if len(output_pitches_midi) > 0 and\
            len(output_pitches_midi) == len(target_pitches_midi) and\
            (output_pitches_midi != target_pitches_midi).sum() == 0:
        output_durs = output_intervals[:, 1] - output_intervals[:, 0]
        target_durs = target_intervals[:, 1] - target_intervals[:, 0]
        errors = target_durs - output_durs
        thresholds = np.clip(target_durs * 0.2, a_min = 0.05, a_max=100000)
        error_num = (np.abs(errors) > thresholds).sum()
        acc = 1 - error_num / len(output_pitches_midi)
        error_durs = target_durs[np.abs(errors) > thresholds]
        error_durs_delta = errors[np.abs(errors) > thresholds]
        err_dur_longer = (errors < -thresholds).sum() / error_num
        metrics_dict["%s_error_durs_list"%name] = error_durs
        metrics_dict["%s_error_durs_list_delta"%name] = error_durs_delta


    print("%s: p=%.3f, r=%.3f, f1=%.3f, acc=%.3f, err_dur_longer=%.3f"%(name, p,r,f, acc, err_dur_longer))


    return metrics_dict

def cal_notes_sustain_metrics(output_tokens, output_onset_pianoroll, target_onset_pianoroll, target_frame_pianoroll):
    """_summary_

    Args:
        output_tokens (_type_): shape [B*T, n_tokens]
        output_onset_pianoroll: [B, T, F]
        target_onset_pianoroll (_type_): [B, T, F]
        target_frame_pianoroll (_type_): [B, T, F]
    """
    # => [B*T, 256]
    output_on_off_pianoroll = notes_parser.frame_tokens_to_pianoroll(output_tokens)
    output_frame_pianoroll = output_on_off_pianoroll[:, :128]
    # Add offset to frame pianoroll
    output_offset_pianoroll = output_on_off_pianoroll[:, 128:256]
    # output_frame_pianoroll[(output_frame_pianoroll == 1) & (output_offset_pianoroll == 1) ] = 0
    # output_frame_pianoroll = output_offset_pianoroll
    

    B, T, F = target_onset_pianoroll.size()
    output_frame_pianoroll = output_frame_pianoroll.reshape([B, T, F])

    return cal_onsets_and_frames_metrics("Sustain", output_onset_pianoroll, output_frame_pianoroll, target_onset_pianoroll, target_frame_pianoroll)
    
    






def cal_decoder_token_metrics(output_tokens, target_tokens, metric_prefix=""):
    """_summary_

    Args:
        output_tokens (torch.tensor.long): [B, n_tokens], n_tokens is the max token seq length.
        target_tokens (torch.tensor.long): [B, n_tokens]
        target_tokens_mask (torch.tensor.long): [B, n_tokens]
    Returns:
        metric_dict (Dict):
    """
    metrics_dict = {}
    
    output_tokens = torch.clone(output_tokens)
    # self.log_time_event("metrics_cal_begin")
    # B, n_tokens = output_tokens.size()

    # # set tokens after eos to TOKEN_PAD
    # eos_mask = torch.zeros([B], device=output_tokens.device).bool()
    # for i in range(n_tokens):
    #     output_tokens[eos_mask, i] = TOKEN_PAD
    #     eos_mask = eos_mask | (output_tokens[:, i] == TOKEN_END)
    #     # eos_mask = eos_mask | (output_tokens[:, i] >= 256) 


    # cal metrics

    # cal acc of tokens.
    output_tokens = output_tokens.view(-1)#.cpu().numpy()
    target_tokens = target_tokens.view(-1)#.cpu().numpy()
    def event_acc(token_type, begin, end, output_tokens, target_tokens):
        # sel_idx = (target_tokens >= begin) &  (target_tokens < end)
        # output_tokens_sel = output_tokens[sel_idx]
        # target_tokens_sel =target_tokens[sel_idx]
        # if len(target_tokens_sel) == 0:
        #     return 0
        # return accuracy_score(target_tokens_sel, output_tokens_sel)
        if end - begin == 1:
            sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
            sel_outputs = ((output_tokens >= begin) & (output_tokens < end))
            P = sel_targets.sum()
            if P == 0:
                return {}
            TP = (output_tokens[sel_targets & sel_outputs] == target_tokens[sel_targets & sel_outputs]).sum()

            FP = sel_outputs.sum() - TP
            if TP + FP == 0:
                precision= 0
            else:
                precision = TP/(TP+FP)
            recall = TP / P
            if (precision + recall) <= 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            d =  {
                "%stoken_%s_p"%(metric_prefix, token_type): float(precision),
                "%stoken_%s_r"%(metric_prefix, token_type): float(recall),
                "%stoken_%s_f1"%(metric_prefix, token_type): float(f1),
            }
        else:
            sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
            num = sel_targets.sum()
            if num == 0:
                return {}
            TP = (output_tokens[sel_targets] == target_tokens[sel_targets]).sum()
            acc = TP / num
            d =  {
                "%stoken_%s_acc"%(metric_prefix, token_type): float(acc),
            }
        return d
    
    metrics_dict.update( event_acc("all", 0, TOKEN_PAD, output_tokens, target_tokens) )
    for token_type in sm_tokenizer.token_type_list:
        name = token_type.get_type_name()
        begin, end = token_type.get_bound()
        metrics_dict.update( event_acc(name, begin, end, output_tokens, target_tokens) )
    
    # EOS
    begin = sm_tokenizer.EOS
    end = begin + 1
    metrics_dict.update( event_acc("EOS", begin, end, output_tokens, target_tokens) )

    return metrics_dict


def cal_multihot_notes_metrics(name, decoder_outputs_note, decoder_targets_note):
    """_summary_

    Args:
        decoder_outputs_note (torch.tensor.float): [B, n_tokens, 128], n_tokens is the max token seq length.
        decoder_targets_note (torch.tensor.long): [B, n_tokens, 128]
    Returns:
        metric_dict (Dict):
    """
    B, n_tokens, n_notes = decoder_targets_note.size()
    decoder_outputs_note = (torch.sigmoid(decoder_outputs_note) > 0.4).long()
    assert (decoder_targets_note == 0).long().sum() + (decoder_targets_note == 1).long().sum() == B * n_tokens * n_notes
    note_num = decoder_targets_note.sum()
    note_p = 0
    note_r = 0
    note_f = 0
    if note_num > 0:
        TP = (decoder_targets_note * decoder_outputs_note).sum()
        FP = ((1-decoder_targets_note) * decoder_outputs_note).sum()
        note_r =  TP / note_num
        if TP + FP > 0:
            note_p = TP/(TP+FP)
        if TP > 0:
            note_f = 2*note_r*note_p /(note_p + note_r)
            
    return {
        name + "_p": note_p,
        name + "_r": note_r,
        name + "_f1": note_f,
    }


def infer_piece_end_time(note_lists=(), pedal_event_lists=(), minimum_end_time=0.05):
    return shared_infer_piece_end_time(
        note_lists=note_lists,
        pedal_event_lists=pedal_event_lists,
        minimum_end_time=minimum_end_time,
    )


def pedal_events_to_spans(pedal_event_list, piece_end_time):
    return shared_pedal_events_to_spans(
        pedal_event_list,
        piece_end_time=piece_end_time,
    )


def pedal_spans_to_event_list(pedal_spans):
    pedal_event_list = []
    for span in pedal_spans:
        onset_time = float(span["onset"])
        offset_time = max(float(span["offset"]), onset_time + PEDAL_MIN_DURATION)
        pedal_event_list.append({"time": onset_time, "type": "PedalOn"})
        pedal_event_list.append({"time": offset_time, "type": "PedalOff"})

    return sorted(
        pedal_event_list,
        key=lambda event: (float(event["time"]), 0 if event["type"] == "PedalOff" else 1),
    )


def _pedal_spans_to_intervals(pedal_spans):
    if len(pedal_spans) == 0:
        return np.empty((0, 2), dtype=np.float64)

    intervals = np.array(
        [[float(span["onset"]), float(span["offset"])] for span in pedal_spans],
        dtype=np.float64,
    )
    intervals[:, 1] = np.maximum(intervals[:, 1], intervals[:, 0] + PEDAL_MIN_DURATION)
    non_positive_duration_mask = intervals[:, 1] <= intervals[:, 0]
    if np.any(non_positive_duration_mask):
        intervals[non_positive_duration_mask, 1] = np.nextafter(
            intervals[non_positive_duration_mask, 0],
            np.inf,
        )
    return intervals


def _ensure_minimum_note_duration(intervals, minimum_duration=0.05):
    if len(intervals) > 0:
        intervals[:, 1] = np.maximum(intervals[:, 1], intervals[:, 0] + minimum_duration)
    return intervals


def _safe_interval_metrics(reference_intervals, estimated_intervals, onset_tolerance=0.05, offset_ratio="default"):
    if len(reference_intervals) == 0 or len(estimated_intervals) == 0:
        return 0.0, 0.0, 0.0, 0.0

    reference_pitches = midi_to_hz(np.full(len(reference_intervals), PEDAL_DUMMY_MIDI))
    estimated_pitches = midi_to_hz(np.full(len(estimated_intervals), PEDAL_DUMMY_MIDI))

    if offset_ratio is None:
        return evaluate_notes(
            reference_intervals,
            reference_pitches,
            estimated_intervals,
            estimated_pitches,
            offset_ratio=None,
            onset_tolerance=onset_tolerance,
        )

    return evaluate_notes(
        reference_intervals,
        reference_pitches,
        estimated_intervals,
        estimated_pitches,
        onset_tolerance=onset_tolerance,
    )


def _coerce_intervals(intervals):
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return intervals.reshape(-1, 2)


def reference_pedal_events_from_dataframe(df):
    if df is None or "type" not in df.columns or "onset_sec" not in df.columns:
        return []

    pedal_rows = df[df["type"].isin(["PedalOn", "PedalOff"])].copy()
    if len(pedal_rows) == 0:
        return []

    pedal_rows["_pedal_sort"] = np.where(pedal_rows["type"] == "PedalOff", 0, 1)
    pedal_rows = pedal_rows.sort_values(
        by=["onset_sec", "_pedal_sort"],
        kind="mergesort",
    )

    return [
        {"time": float(row["onset_sec"]), "type": str(row["type"])}
        for _, row in pedal_rows.iterrows()
    ]


def _reference_pedal_interval_metrics(
    reference_intervals,
    estimated_intervals,
    onset_tolerance=REFERENCE_PEDAL_ONSET_TOLERANCE,
    offset_ratio=REFERENCE_PEDAL_OFFSET_RATIO,
    offset_min_tolerance=REFERENCE_PEDAL_OFFSET_MIN_TOLERANCE,
):
    reference_intervals = _coerce_intervals(reference_intervals)
    estimated_intervals = _coerce_intervals(estimated_intervals)
    if len(reference_intervals) == 0 or len(estimated_intervals) == 0:
        return 0.0, 0.0, 0.0, 0.0

    reference_pitches = np.ones(reference_intervals.shape[0])
    estimated_pitches = np.ones(estimated_intervals.shape[0])
    return evaluate_notes(
        ref_intervals=reference_intervals,
        ref_pitches=reference_pitches,
        est_intervals=estimated_intervals,
        est_pitches=estimated_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )


def cal_reference_pedal_metrics(
    output_pedal_event_list,
    reference_pedal_event_list,
    piece_end_time,
):
    output_pedal_spans = pedal_events_to_spans(output_pedal_event_list, piece_end_time)
    reference_pedal_spans = pedal_events_to_spans(reference_pedal_event_list, piece_end_time)

    output_intervals = _pedal_spans_to_intervals(output_pedal_spans)
    reference_intervals = _pedal_spans_to_intervals(reference_pedal_spans)
    pedal_precision, pedal_recall, pedal_f1, _ = _reference_pedal_interval_metrics(
        reference_intervals,
        output_intervals,
    )

    metric_dict = {
        "pedal_precision": pedal_precision,
        "pedal_recall": pedal_recall,
        "pedal_f1": pedal_f1,
    }
    return metric_dict, output_pedal_spans, reference_pedal_spans


def cal_binary_frame_metrics(name, frame_output, frame_target, threshold=0.5, target_threshold=0.5):
    y_pred = np.asarray(frame_output, dtype=np.float32).reshape(-1) > threshold
    y_true = np.asarray(frame_target, dtype=np.float32).reshape(-1) > target_threshold
    if y_true.size == 0:
        return {
            f"{name}_precision": np.nan,
            f"{name}_recall": np.nan,
            f"{name}_f1": np.nan,
        }

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true.astype(np.int32),
        y_pred.astype(np.int32),
        labels=[0, 1],
        zero_division=0,
    )
    return {
        f"{name}_precision": float(precision[1]),
        f"{name}_recall": float(recall[1]),
        f"{name}_f1": float(f1[1]),
    }


def cal_pedal_frame_metrics(pedal_frame_output, pedal_frame_target, threshold=0.5):
    return cal_binary_frame_metrics(
        "pedal_frame",
        pedal_frame_output,
        pedal_frame_target,
        threshold=threshold,
        target_threshold=0.5,
    )


def collect_trimmed_frame_arrays(data_list, output_key, target_key=None):
    frame_outputs = []
    frame_targets = []
    for row in data_list:
        row_outputs = row.get(output_key, [])
        row_targets = row.get(target_key, []) if target_key is not None else []
        fallback_total = row.get("frame_offsets", 0) + len(row_outputs)
        row_total_frames = int(row.get("total_frames", fallback_total))
        row_frame_offset = int(row.get("frame_offsets", 0))
        valid_frame_candidates = [len(row_outputs), row_total_frames - row_frame_offset]
        if target_key is not None:
            valid_frame_candidates.append(len(row_targets))
        valid_frames = max(0, min(valid_frame_candidates))
        if valid_frames <= 0:
            continue
        frame_outputs.extend(row_outputs[:valid_frames])
        if target_key is not None:
            frame_targets.extend(row_targets[:valid_frames])

    frame_outputs = np.asarray(frame_outputs, dtype=np.float32)
    if target_key is None:
        return frame_outputs
    return frame_outputs, np.asarray(frame_targets, dtype=np.float32)


def collect_trimmed_pedal_frame_arrays(data_list):
    return collect_trimmed_frame_arrays(
        data_list,
        "pedal_frame_output",
        "pedal_frame_target",
    )


def pedal_frame_output_to_events(
    pedal_frame_output,
    sec_per_frame,
    threshold_on=0.5,
    threshold_off=0.4,
    min_pedal_down_frames=3,
    min_pedal_up_frames=2,
):
    raw_spans = pedal_frame_output_to_raw_spans(
        pedal_frame_output,
        threshold_on=threshold_on,
        threshold_off=threshold_off,
    )
    return pedal_frame_spans_to_events(
        raw_spans,
        sec_per_frame=sec_per_frame,
        min_pedal_down_frames=min_pedal_down_frames,
        min_pedal_up_frames=min_pedal_up_frames,
    )


def pedal_frame_output_to_raw_spans(
    pedal_frame_output,
    threshold_on=0.5,
    threshold_off=0.4,
):
    probs = np.asarray(pedal_frame_output, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return []

    state = np.zeros(probs.shape, dtype=np.bool_)
    down = False
    for frame_idx, prob in enumerate(probs):
        if down:
            if prob < threshold_off:
                down = False
        else:
            if prob >= threshold_on:
                down = True
        state[frame_idx] = down

    diff = np.diff(state.astype(np.int8), prepend=0, append=0)
    on_frames = np.where(diff == 1)[0]
    off_frames = np.where(diff == -1)[0]
    return list(zip(on_frames.tolist(), off_frames.tolist()))


def pedal_frame_spans_to_events(
    spans,
    sec_per_frame,
    min_pedal_down_frames=3,
    min_pedal_up_frames=2,
):
    merged_spans = []
    min_pedal_up_frames = max(0, int(min_pedal_up_frames))
    for on_frame, off_frame in spans:
        if merged_spans and on_frame - merged_spans[-1][1] < min_pedal_up_frames:
            merged_spans[-1] = (merged_spans[-1][0], off_frame)
        else:
            merged_spans.append((on_frame, off_frame))

    min_pedal_down_frames = max(1, int(min_pedal_down_frames))
    merged_spans = [
        (on_frame, off_frame)
        for on_frame, off_frame in merged_spans
        if off_frame - on_frame >= min_pedal_down_frames
    ]

    events = []
    for on_frame, off_frame in merged_spans:
        events.append({"time": float(on_frame) * sec_per_frame, "type": "PedalOn"})
        events.append({"time": float(off_frame) * sec_per_frame, "type": "PedalOff"})
    return events


def choose_pedal_event_list(decoder_pedal_event_list, frame_head_pedal_event_list, requested_source):
    if requested_source == "decoder":
        return decoder_pedal_event_list, "decoder"
    if requested_source == "frame_head":
        if frame_head_pedal_event_list is None:
            return decoder_pedal_event_list, "decoder_fallback_frame_head_unavailable"
        return frame_head_pedal_event_list, "frame_head"
    raise ValueError(
        f"Unknown pedal event source {requested_source!r}. Expected 'decoder' or 'frame_head'."
    )


def _count_matched_event_times(reference_times, estimated_times, tolerance=PEDAL_EVENT_TOLERANCE):
    reference_times = np.sort(np.asarray(reference_times, dtype=np.float32))
    estimated_times = np.sort(np.asarray(estimated_times, dtype=np.float32))

    ref_idx = 0
    est_idx = 0
    matched = 0
    while ref_idx < len(reference_times) and est_idx < len(estimated_times):
        delta = estimated_times[est_idx] - reference_times[ref_idx]
        if abs(delta) <= tolerance:
            matched += 1
            ref_idx += 1
            est_idx += 1
        elif delta < -tolerance:
            est_idx += 1
        else:
            ref_idx += 1

    return matched


def _precision_recall_f1(match_count, estimated_count, reference_count):
    precision = match_count / estimated_count if estimated_count > 0 else 0.0
    recall = match_count / reference_count if reference_count > 0 else 0.0
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def cal_pedal_metrics(output_pedal_event_list, target_pedal_event_list, piece_end_time, onset_tolerance=0.05, event_tolerance=PEDAL_EVENT_TOLERANCE):
    output_pedal_spans = pedal_events_to_spans(output_pedal_event_list, piece_end_time)
    target_pedal_spans = pedal_events_to_spans(target_pedal_event_list, piece_end_time)

    output_intervals = _pedal_spans_to_intervals(output_pedal_spans)
    target_intervals = _pedal_spans_to_intervals(target_pedal_spans)

    onset_precision, onset_recall, onset_f1, _ = _safe_interval_metrics(
        target_intervals,
        output_intervals,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
    )
    note_precision, note_recall, note_f1, _ = _safe_interval_metrics(
        target_intervals,
        output_intervals,
        onset_tolerance=onset_tolerance,
    )

    target_pedal_on_times = [float(event["time"]) for event in target_pedal_event_list if event.get("type") == "PedalOn"]
    output_pedal_on_times = [float(event["time"]) for event in output_pedal_event_list if event.get("type") == "PedalOn"]
    pedal_on_match_count = _count_matched_event_times(target_pedal_on_times, output_pedal_on_times, tolerance=event_tolerance)
    pedal_on_precision, pedal_on_recall, pedal_on_f1 = _precision_recall_f1(
        pedal_on_match_count,
        len(output_pedal_on_times),
        len(target_pedal_on_times),
    )

    target_pedal_off_times = [float(event["time"]) for event in target_pedal_event_list if event.get("type") == "PedalOff"]
    output_pedal_off_times = [float(event["time"]) for event in output_pedal_event_list if event.get("type") == "PedalOff"]
    pedal_off_match_count = _count_matched_event_times(target_pedal_off_times, output_pedal_off_times, tolerance=event_tolerance)
    pedal_off_precision, pedal_off_recall, pedal_off_f1 = _precision_recall_f1(
        pedal_off_match_count,
        len(output_pedal_off_times),
        len(target_pedal_off_times),
    )

    metric_dict = {
        "pedal_precision": onset_precision,
        "pedal_recall": onset_recall,
        "pedal_f1": onset_f1,
        "pedal+offset_precision": note_precision,
        "pedal+offset_recall": note_recall,
        "pedal+offset_f1": note_f1,
        "pedal_on_precision": pedal_on_precision,
        "pedal_on_recall": pedal_on_recall,
        "pedal_on_f1": pedal_on_f1,
        "pedal_off_precision": pedal_off_precision,
        "pedal_off_recall": pedal_off_recall,
        "pedal_off_f1": pedal_off_f1,
    }
    return metric_dict, output_pedal_spans, target_pedal_spans


def _note_rows_to_list(note_rows, offset_column):
    notes = []
    for _, row in note_rows.iterrows():
        onset_time = float(row["onset_sec"])
        offset_time = float(row[offset_column])
        notes.append({
            "onset": onset_time,
            "offset": offset_time,
            "duration": offset_time - onset_time,
            "pitch": int(row["pitch"]),
            "velocity": float(row["velocity"]),
        })
    return notes


def _notes_to_interval_array(notes):
    if len(notes) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(note["onset"], note["offset"]) for note in notes], dtype=np.float32)


def _notes_to_pitch_array(notes):
    if len(notes) == 0:
        return np.array([], dtype=np.float32)
    return np.array([note["pitch"] for note in notes], dtype=np.float32)


def _notes_to_velocity_array(notes):
    if len(notes) == 0:
        return np.array([], dtype=np.float32)
    return np.array([note["velocity"] for note in notes], dtype=np.float32)


def build_pedal_extended_note_metric_inputs(output_note_data_list, output_pedal_event_list, tsv_df, piece_end_time):
    tsv_notes = tsv_df[tsv_df["type"] == "note"] if "type" in tsv_df.columns else tsv_df
    output_notes_ext = extend_notes_with_reference_pedal_events(
        output_note_data_list,
        output_pedal_event_list,
        piece_end_time=piece_end_time,
    )
    output_notes_ext_uncapped = extend_notes_with_pedal_events(
        output_note_data_list,
        output_pedal_event_list,
        piece_end_time=piece_end_time,
    )

    can_rebuild_reference_target = "offset_sec_truth" in tsv_notes.columns and "type" in tsv_df.columns
    if can_rebuild_reference_target:
        reference_notes_raw = _note_rows_to_list(tsv_notes, "offset_sec_truth")
        reference_pedal_events = reference_pedal_events_from_dataframe(tsv_df)
        gt_notes_ext = extend_notes_with_reference_pedal_events(
            reference_notes_raw,
            reference_pedal_events,
            piece_end_time=piece_end_time,
        )
        pedal_extended_target_source = "offset_sec_truth+pedal_events"
    else:
        gt_notes_ext = _note_rows_to_list(tsv_notes, "offset_sec")
        pedal_extended_target_source = "offset_sec"

    gt_notes_ext_uncapped = _note_rows_to_list(tsv_notes, "offset_sec")

    gt_interval_ext = _notes_to_interval_array(gt_notes_ext)
    out_interval_ext = _notes_to_interval_array(output_notes_ext)
    diagnostic_gt_interval_ext_uncapped = _notes_to_interval_array(gt_notes_ext_uncapped)
    diagnostic_out_interval_ext_uncapped = _notes_to_interval_array(output_notes_ext_uncapped)

    gt_interval_ext = _ensure_minimum_note_duration(gt_interval_ext)
    out_interval_ext = _ensure_minimum_note_duration(out_interval_ext)
    diagnostic_gt_interval_ext_uncapped = _ensure_minimum_note_duration(diagnostic_gt_interval_ext_uncapped)
    diagnostic_out_interval_ext_uncapped = _ensure_minimum_note_duration(diagnostic_out_interval_ext_uncapped)

    gt_pitches_raw = _notes_to_pitch_array(gt_notes_ext)
    gt_velocities = _notes_to_velocity_array(gt_notes_ext)
    out_pitches_raw = _notes_to_pitch_array(output_notes_ext)
    out_velocities = _notes_to_velocity_array(output_notes_ext)
    diagnostic_gt_pitches_raw = _notes_to_pitch_array(gt_notes_ext_uncapped)
    diagnostic_gt_velocities = _notes_to_velocity_array(gt_notes_ext_uncapped)
    diagnostic_out_pitches_raw = _notes_to_pitch_array(output_notes_ext_uncapped)
    diagnostic_out_velocities = _notes_to_velocity_array(output_notes_ext_uncapped)

    return {
        "gt_interval_ext": gt_interval_ext,
        "gt_pitches_hz": midi_to_hz(gt_pitches_raw) if len(gt_pitches_raw) else np.array([], dtype=np.float32),
        "gt_velocities": gt_velocities,
        "out_interval_ext": out_interval_ext,
        "out_pitches_ext": midi_to_hz(out_pitches_raw) if len(out_pitches_raw) else np.array([], dtype=np.float32),
        "out_vel_ext": out_velocities,
        "diagnostic_gt_interval_ext_uncapped": diagnostic_gt_interval_ext_uncapped,
        "diagnostic_gt_pitches_hz_uncapped": midi_to_hz(diagnostic_gt_pitches_raw)
        if len(diagnostic_gt_pitches_raw)
        else np.array([], dtype=np.float32),
        "diagnostic_gt_velocities_uncapped": diagnostic_gt_velocities,
        "diagnostic_out_interval_ext_uncapped": diagnostic_out_interval_ext_uncapped,
        "diagnostic_out_pitches_ext_uncapped": midi_to_hz(diagnostic_out_pitches_raw)
        if len(diagnostic_out_pitches_raw)
        else np.array([], dtype=np.float32),
        "diagnostic_out_vel_ext_uncapped": diagnostic_out_velocities,
        "pedal_extended_target_source": pedal_extended_target_source,
    }


def _calculate_note_metric_values(reference_intervals, reference_pitches, reference_velocities, estimated_intervals, estimated_pitches, estimated_velocities):
    if len(reference_intervals) == 0 or len(estimated_intervals) == 0:
        note_precision = note_recall = note_f1 = 0.0
        note_vel_precision = note_vel_recall = note_vel_f1 = 0.0
    else:
        note_precision, note_recall, note_f1, _ = evaluate_notes(
            reference_intervals,
            reference_pitches,
            estimated_intervals,
            estimated_pitches,
        )
        note_vel_precision, note_vel_recall, note_vel_f1, _ = evaluate_notes_with_velocity(
            reference_intervals,
            reference_pitches,
            reference_velocities,
            estimated_intervals,
            estimated_pitches,
            estimated_velocities,
            velocity_tolerance=0.1,
        )

    return note_precision, note_recall, note_f1, note_vel_precision, note_vel_recall, note_vel_f1


def cal_pedal_extended_note_metrics(output_note_data_list, output_pedal_event_list, tsv_df, piece_end_time):
    metric_inputs = build_pedal_extended_note_metric_inputs(
        output_note_data_list,
        output_pedal_event_list,
        tsv_df,
        piece_end_time,
    )

    note_precision, note_recall, note_f1, note_vel_precision, note_vel_recall, note_vel_f1 = _calculate_note_metric_values(
        metric_inputs["gt_interval_ext"],
        metric_inputs["gt_pitches_hz"],
        metric_inputs["gt_velocities"],
        metric_inputs["out_interval_ext"],
        metric_inputs["out_pitches_ext"],
        metric_inputs["out_vel_ext"],
    )
    (
        diagnostic_note_precision,
        diagnostic_note_recall,
        diagnostic_note_f1,
        diagnostic_note_vel_precision,
        diagnostic_note_vel_recall,
        diagnostic_note_vel_f1,
    ) = _calculate_note_metric_values(
        metric_inputs["diagnostic_gt_interval_ext_uncapped"],
        metric_inputs["diagnostic_gt_pitches_hz_uncapped"],
        metric_inputs["diagnostic_gt_velocities_uncapped"],
        metric_inputs["diagnostic_out_interval_ext_uncapped"],
        metric_inputs["diagnostic_out_pitches_ext_uncapped"],
        metric_inputs["diagnostic_out_vel_ext_uncapped"],
    )

    metric_dict = {
        "note+offset_precision_pedal_extended": note_precision,
        "note+offset_recall_pedal_extended": note_recall,
        "note+offset_f1_pedal_extended": note_f1,
        "note+offset+velocity_precision_pedal_extended": note_vel_precision,
        "note+offset+velocity_recall_pedal_extended": note_vel_recall,
        "note+offset+velocity_f1_pedal_extended": note_vel_f1,
        "diagnostic_note+offset_precision_pedal_extended_uncapped": diagnostic_note_precision,
        "diagnostic_note+offset_recall_pedal_extended_uncapped": diagnostic_note_recall,
        "diagnostic_note+offset_f1_pedal_extended_uncapped": diagnostic_note_f1,
        "diagnostic_note+offset+velocity_precision_pedal_extended_uncapped": diagnostic_note_vel_precision,
        "diagnostic_note+offset+velocity_recall_pedal_extended_uncapped": diagnostic_note_vel_recall,
        "diagnostic_note+offset+velocity_f1_pedal_extended_uncapped": diagnostic_note_vel_f1,
    }
    return metric_dict, metric_inputs


def get_intervals_notes(midi_path):
    notes, pedals = pianoroll_parser.get_notes_with_pedal(midi_path)
    onsets = notes["onset"]
    offsets = notes["offset"]
    pit = notes["pitch"]
    # Remove overlap.
    overlap_num = 0
    for p in range(128):
        mask = (pit == p)
        num = mask.sum()
        if  num > 0:
            overlap = (offsets[mask][:-1] > onsets[mask][1:]).astype(int).sum()
            if overlap > 0:
                # print(overlap)
                overlap_num += overlap
                sel_offset = offsets[mask]
                offsets[mask] = np.concatenate( [np.minimum(sel_offset[:-1], onsets[mask][1:]) , sel_offset[-1:] ], axis=0 ) # Replace inplace.
    print("overlap_num:", overlap_num)
    intervals = np.stack([onsets, offsets], axis=1)
    return intervals, pit, pedals


def cal_midi_files_metrics(est_midi_paths: list, ref_midi_paths: list, result_path):
    result_dict = defaultdict(list)
    for est_path, ref_path in tqdm(zip(est_midi_paths, ref_midi_paths)):
        est_intervals, est_notes, est_pedals = get_intervals_notes(est_path)
        ref_intervals, ref_notes, ref_pedals = get_intervals_notes(ref_path)

        est_intervals += 0.01

        est_intervals[:, 1] = np.maximum(est_intervals[:, 1], est_intervals[:, 0] + 0.05) #min dur >= 0.05


        est_pitches = midi_to_hz(est_notes)
        ref_pitches = midi_to_hz(ref_notes)

        # Frames
        # evaluate_frames()

        # Onsets
        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None, onset_tolerance=0.05)
        result_dict["onset_precision"].append(p*100)
        result_dict["onset_recall"].append(r*100)
        result_dict["onset_f1"].append(f*100)
        
        # Notes
        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05)
        result_dict["note_precision"].append(p*100)
        result_dict["note_recall"].append(r*100)
        result_dict["note_f1"].append(f*100)
        result_dict["note_num"].append(len(ref_pitches))

        # ref_dur = ref_intervals[:, 1] - ref_intervals[:, 0]
        # est_dur = est_intervals[:, 1] - est_intervals[:, 0]
        # errors = np.abs(ref_dur - est_dur)
        # thresholds = np.clip(ref_dur * 0.2, a_min=0.05, a_max=99999999)
        # acc = (errors <= thresholds).sum() / len(ref_dur)


        result_dict["midi_ref"].append(os.path.split(ref_path)[1])
        result_dict["midi_est"].append(os.path.split(est_path)[1])
    # p_avg = np.mean(result_dict["onset_precision"])
    # result_dict["onset_precision"].append(p_avg)
    # r_avg = np.mea
    for k,v in result_dict.items():
        if not k.startswith("midi"):
            avg = np.mean(v)
            result_dict[k].append(avg)
        else:
            result_dict[k].append("Avg")

    df = pd.DataFrame(result_dict)
    df.to_csv(result_path, float_format="%.2f")
    # print(df.tail(10))
    
