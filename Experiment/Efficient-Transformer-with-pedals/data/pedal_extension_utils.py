import numpy as np


PEDAL_MIN_DURATION = 1e-6
MINIMUM_PIECE_END_TIME = 0.05


def infer_piece_end_time(note_lists=(), pedal_event_lists=(), minimum_end_time=MINIMUM_PIECE_END_TIME):
    piece_end_time = float(minimum_end_time)

    for note_list in note_lists:
        for note in note_list:
            piece_end_time = max(piece_end_time, float(note.get("offset", note.get("onset", 0.0))))

    for pedal_event_list in pedal_event_lists:
        for event in pedal_event_list:
            piece_end_time = max(piece_end_time, float(event.get("time", 0.0)))

    return piece_end_time


def normalize_pedal_spans(pedal_spans, minimum_duration=PEDAL_MIN_DURATION):
    normalized_spans = []

    for span in pedal_spans:
        onset_time = float(span["onset"])
        offset_time = max(float(span["offset"]), onset_time + minimum_duration)
        normalized_spans.append({
            "onset": onset_time,
            "offset": offset_time,
        })

    return sorted(normalized_spans, key=lambda span: (span["onset"], span["offset"]))


def pedal_records_to_spans(pedals, minimum_duration=PEDAL_MIN_DURATION):
    if pedals is None or len(pedals) == 0:
        return []

    pedal_dtype_names = getattr(getattr(pedals, "dtype", None), "names", None) or ()
    pedal_end_times = pedals["end"] if "end" in pedal_dtype_names else pedals["time"] + pedals["duration"]
    pedal_spans = [
        {
            "onset": float(onset_time),
            "offset": float(offset_time),
        }
        for onset_time, offset_time in zip(pedals["time"], pedal_end_times)
    ]
    return normalize_pedal_spans(pedal_spans, minimum_duration=minimum_duration)


def pedal_events_to_spans(pedal_event_list, piece_end_time=None, minimum_duration=PEDAL_MIN_DURATION):
    sorted_events = sorted(
        (
            {
                "time": float(event["time"]),
                "type": event["type"],
            }
            for event in pedal_event_list
            if event.get("type") in {"PedalOn", "PedalOff"} and "time" in event
        ),
        key=lambda event: (event["time"], 0 if event["type"] == "PedalOff" else 1),
    )

    pedal_spans = []
    pedal_on_time = None
    for event in sorted_events:
        if event["type"] == "PedalOn":
            if pedal_on_time is None:
                pedal_on_time = event["time"]
            continue

        if pedal_on_time is None:
            continue

        pedal_spans.append({
            "onset": pedal_on_time,
            "offset": event["time"],
        })
        pedal_on_time = None

    if pedal_on_time is not None and piece_end_time is not None:
        pedal_spans.append({
            "onset": pedal_on_time,
            "offset": float(piece_end_time),
        })

    return normalize_pedal_spans(pedal_spans, minimum_duration=minimum_duration)


def extend_offset_array_with_pedal_spans(raw_offsets, pedal_spans, onset_times=None):
    raw_offsets = np.asarray(raw_offsets, dtype=np.float64)
    if onset_times is None:
        onset_times = np.full_like(raw_offsets, -np.inf)
    else:
        onset_times = np.asarray(onset_times, dtype=np.float64)

    extended_offsets = np.array(raw_offsets, copy=True)
    if raw_offsets.size == 0 or len(pedal_spans) == 0:
        return np.maximum(extended_offsets, onset_times)

    span_onsets = np.array([float(span["onset"]) for span in pedal_spans], dtype=np.float64)
    span_offsets = np.array([float(span["offset"]) for span in pedal_spans], dtype=np.float64)

    for idx, raw_offset in enumerate(raw_offsets):
        matching_span_mask = (span_onsets <= raw_offset) & (raw_offset <= span_offsets)
        if np.any(matching_span_mask):
            extended_offsets[idx] = float(np.max(span_offsets[matching_span_mask]))

    return np.maximum(extended_offsets, onset_times)


def extend_notes_with_pedal_spans(notes, pedal_spans):
    if len(notes) == 0:
        return []

    onset_times = np.array([float(note["onset"]) for note in notes], dtype=np.float64)
    raw_offsets = np.array([float(note.get("offset", note["onset"])) for note in notes], dtype=np.float64)
    extended_offsets = extend_offset_array_with_pedal_spans(raw_offsets, pedal_spans, onset_times=onset_times)

    extended_notes = []
    for note, extended_offset in zip(notes, extended_offsets):
        new_note = dict(note)
        new_note["offset"] = float(extended_offset)
        new_note["duration"] = float(extended_offset - float(note["onset"]))
        extended_notes.append(new_note)

    return extended_notes


def extend_notes_with_reference_pedal_spans(notes, pedal_spans):
    """Extend note offsets using the reference pedal target semantics.

    This mirrors the reference piano_transcription target extension more closely
    than the cached-TSV helper above: a note is extended only when its raw offset
    is strictly inside a pedal span, and repeated notes of the same pitch within
    that span clip the previous extended note at the next onset.
    """
    if len(notes) == 0:
        return []

    normalized_spans = normalize_pedal_spans(pedal_spans)
    extended_records = []

    for note_index, note in enumerate(notes):
        onset_time = float(note["onset"])
        raw_offset = float(note.get("offset", onset_time))
        extended_offset = raw_offset
        matched_span_index = None

        for span_index, span in enumerate(normalized_spans):
            span_onset = float(span["onset"])
            span_offset = float(span["offset"])
            if span_onset < raw_offset < span_offset:
                extended_offset = span_offset
                matched_span_index = span_index
                break

        new_note = dict(note)
        new_note["offset"] = max(float(extended_offset), onset_time)
        new_note["duration"] = new_note["offset"] - onset_time
        extended_records.append({
            "index": note_index,
            "note": new_note,
            "span_index": matched_span_index,
        })

    previous_by_span_and_pitch = {}
    for record in sorted(
        extended_records,
        key=lambda item: (float(item["note"]["onset"]), item["index"]),
    ):
        span_index = record["span_index"]
        if span_index is None:
            continue

        pitch = record["note"].get("pitch", record["note"].get("midi_note"))
        key = (span_index, pitch)
        previous_record = previous_by_span_and_pitch.get(key)

        if previous_record is not None:
            previous_note = previous_record["note"]
            cap_time = float(record["note"]["onset"])
            if float(previous_note["offset"]) > cap_time:
                previous_note["offset"] = max(cap_time, float(previous_note["onset"]))
                previous_note["duration"] = previous_note["offset"] - float(previous_note["onset"])

        previous_by_span_and_pitch[key] = record

    return [record["note"] for record in sorted(extended_records, key=lambda item: item["index"])]


def extend_notes_with_reference_pedal_events(notes, pedal_event_list, piece_end_time=None):
    if len(notes) == 0:
        return []

    if piece_end_time is None:
        piece_end_time = infer_piece_end_time(
            note_lists=[notes],
            pedal_event_lists=[pedal_event_list],
        )

    pedal_spans = pedal_events_to_spans(
        pedal_event_list,
        piece_end_time=piece_end_time,
    )
    return extend_notes_with_reference_pedal_spans(notes, pedal_spans)


def extend_notes_with_pedal_events(notes, pedal_event_list, piece_end_time=None):
    if len(notes) == 0:
        return []

    if piece_end_time is None:
        piece_end_time = infer_piece_end_time(
            note_lists=[notes],
            pedal_event_lists=[pedal_event_list],
        )

    pedal_spans = pedal_events_to_spans(
        pedal_event_list,
        piece_end_time=piece_end_time,
    )
    return extend_notes_with_pedal_spans(notes, pedal_spans)
