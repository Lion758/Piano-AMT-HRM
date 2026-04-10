import os
import sys
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)

import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import shutil
import numpy as np
import symusic

dataset_dir = "/home/rachel/.group-5/Piano-AMT-HRM/Backend/efficient-seq2seq-piano-trans/dataset/maestro-v3.0.0"


def compute_pedal_extended_offsets(raw_offsets, pedals):
    extended_offsets = np.copy(raw_offsets)
    for pedal_start, pedal_end in zip(pedals["time"], pedals["end"]):
        mask = (raw_offsets >= pedal_start) & (raw_offsets <= pedal_end)
        extended_offsets[mask] = pedal_end
    return extended_offsets

def process_midi_file(midi_path):
    
    midi_path_rel = os.path.relpath(midi_path, dataset_dir)
    notes_tsv_path = os.path.join(dataset_dir, "cache", midi_path_rel.replace("/", ">").replace(".midi", "") + ".midi-notes.tsv")
    midi_path_out = notes_tsv_path.replace(".midi-notes.tsv", ".mid")
    os.makedirs(os.path.dirname(notes_tsv_path), exist_ok=True)
    shutil.copyfile(midi_path, midi_path_out)
    
    # Parse raw note truth and pedal spans from the same track so both
    # offset variants are derived from one aligned source.
    symusic_midi = symusic.Score(midi_path, ttype=symusic.TimeUnit.second)
    assert len(symusic_midi.tracks) == 1
    track = symusic_midi.tracks[0]
    note_array = track.notes.numpy()
    pedals = track.pedals.numpy()
    pedals["end"] = pedals["time"] + pedals["duration"]

    onset_sec = note_array["time"]
    offset_sec_truth = note_array["time"] + note_array["duration"]
    offset_sec_pedal_extended = compute_pedal_extended_offsets(offset_sec_truth, pedals)
    df_notes = pd.DataFrame({
        "type": "note",
        "type_id": 1,  # 1 for note
        "onset_sec": onset_sec,
        "duration_sec": offset_sec_pedal_extended - onset_sec,
        "offset_sec": offset_sec_pedal_extended,
        "duration_sec_truth": offset_sec_truth - onset_sec,
        "offset_sec_truth": offset_sec_truth,
        "duration_sec_pedal_extended": offset_sec_pedal_extended - onset_sec,
        "offset_sec_pedal_extended": offset_sec_pedal_extended,
        "pitch": note_array["pitch"],
        "velocity": note_array["velocity"],
    })

    # Add PedalOn events (pedal press times)
    df_frames = [df_notes]
    if len(pedals["time"]) > 0:
        df_pedal_on = pd.DataFrame({
            "type": "PedalOn",
            "type_id": 0,
            "onset_sec": pedals["time"],
            "duration_sec": pedals["duration"],
            "offset_sec": pedals["end"],
            "duration_sec_truth": pedals["duration"],
            "offset_sec_truth": pedals["end"],
            "duration_sec_pedal_extended": pedals["duration"],
            "offset_sec_pedal_extended": pedals["end"],
            "pitch": -1,
            "velocity": 0,
        })
        df_pedal_off = pd.DataFrame({
            "type": "PedalOff",
            "type_id": 0,
            "onset_sec": pedals["end"],
            "duration_sec": 0.0,
            "offset_sec": pedals["end"],
            "duration_sec_truth": 0.0,
            "offset_sec_truth": pedals["end"],
            "duration_sec_pedal_extended": 0.0,
            "offset_sec_pedal_extended": pedals["end"],
            "pitch": -1,
            "velocity": 0,
        })
        df_frames.append(df_pedal_on)
        df_frames.append(df_pedal_off)

    df_all = pd.concat(df_frames, ignore_index=True)
    df_all.sort_values(by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True], inplace=True)
    df_all.to_csv(notes_tsv_path, sep="\t", index=False)
    return True
    
    
if __name__ == "__main__":
    midi_paths = glob(os.path.join(dataset_dir, "**/*.midi"), recursive=True )
    results = []
    if os.environ.get("DEBUG") == "True":
        # try:
        for midi_path in tqdm(midi_paths):
            process_midi_file(midi_path)
            results.append(midi_path)
        # except Exception as e:
        #     print("Fail:", midi_path)
        #     print("Exception:", e)
    else:
        with Pool(processes=16) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_midi_file, midi_paths), total=len(midi_paths)):
                if result != False:
                    results.append(result)
    print("Total:", len(midi_paths))
    print("Success:", len(results))
    exit()
