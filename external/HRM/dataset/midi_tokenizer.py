# src/data/midi_tokenizer.py
from typing import List, Dict, Tuple
import json
import math

# Build vocab
PITCH_MIN, PITCH_MAX = 0, 127
NUM_PITCH = PITCH_MAX - PITCH_MIN + 1

TIME_BUCKET_MS = 10  # bucket size
MAX_TIME_MS = 5000   # max time shift representable
NUM_TIME_BUCKETS = (MAX_TIME_MS // TIME_BUCKET_MS) + 1

VELOCITY_BUCKETS = 32
VEL_MAX = 127

def build_vocab() -> Tuple[Dict[str,int], Dict[int,str]]:
    tokens = []
    tokens.append("<PAD>")
    tokens.append("<BOS>")
    tokens.append("<EOS>")
    tokens.append("<UNK>")
    # Note-on / Note-off
    for p in range(PITCH_MIN, PITCH_MAX+1):
        tokens.append(f"NOTE_ON_{p}")
        tokens.append(f"NOTE_OFF_{p}")
    # velocity buckets
    for v in range(VELOCITY_BUCKETS):
        tokens.append(f"VELOCITY_{v}")
    # time shifts
    for t in range(NUM_TIME_BUCKETS):
        tokens.append(f"TIME_SHIFT_{t}")
    id_to_token = {i:tok for i,tok in enumerate(tokens)}
    token_to_id = {tok:i for i,tok in id_to_token.items()}
    return token_to_id, id_to_token

TOKEN_TO_ID, ID_TO_TOKEN = build_vocab()

def pitch_on_token(pitch:int) -> int:
    return TOKEN_TO_ID.get(f"NOTE_ON_{pitch}", TOKEN_TO_ID["<UNK>"])
def pitch_off_token(pitch:int) -> int:
    return TOKEN_TO_ID.get(f"NOTE_OFF_{pitch}", TOKEN_TO_ID["<UNK>"])

def velocity_token(velocity:int) -> int:
    b = min(VELOCITY_BUCKETS-1, int(velocity / (VEL_MAX / VELOCITY_BUCKETS)))
    return TOKEN_TO_ID[f"VELOCITY_{b}"]

def time_shift_token(ms:int) -> int:
    bucket = min(NUM_TIME_BUCKETS-1, ms // TIME_BUCKET_MS)
    return TOKEN_TO_ID[f"TIME_SHIFT_{bucket}"]

def save_vocab(path:str):
    with open(path,"w") as f:
        json.dump({"token_to_id":TOKEN_TO_ID, "id_to_token":ID_TO_TOKEN}, f)
