export const SUSTAIN_THRESHOLD = 64 / 127;

function clamp01(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0;
  }
  return Math.max(0, Math.min(1, numericValue));
}

function getTrackSustainChanges(track, trackIndex) {
  const numericEvents = Array.isArray(track?.controlChanges?.[64]) ? track.controlChanges[64] : [];
  const namedEvents = Array.isArray(track?.controlChanges?.sustain) ? track.controlChanges.sustain : [];
  const sourceEvents = numericEvents.length > 0 ? numericEvents : namedEvents;

  return sourceEvents
    .map((event) => {
      const time = Number(event?.time);
      if (!Number.isFinite(time)) {
        return null;
      }

      const value = clamp01(event?.value);
      return {
        time,
        value,
        isActive: value >= SUSTAIN_THRESHOLD,
        trackIndex,
      };
    })
    .filter(Boolean);
}

function findLastSpanStartingBefore(time, sustainSpans) {
  let low = 0;
  let high = sustainSpans.length - 1;
  let result = -1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (sustainSpans[mid].onset <= time) {
      result = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return result;
}

function findLastSustainEventBefore(time, sustainEvents) {
  let low = 0;
  let high = sustainEvents.length - 1;
  let result = -1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (sustainEvents[mid].time <= time) {
      result = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return result;
}

export function extractSustainEvents(midi, pieceEndTime = 0) {
  const rawEvents = Array.isArray(midi?.tracks)
    ? midi.tracks.flatMap((track, trackIndex) => getTrackSustainChanges(track, trackIndex))
    : [];

  rawEvents.sort((a, b) => a.time - b.time || a.trackIndex - b.trackIndex || a.value - b.value);

  const lastSustainEventTime = rawEvents.length > 0 ? rawEvents[rawEvents.length - 1].time : 0;
  const resolvedPieceEndTime = Math.max(
    Number.isFinite(pieceEndTime) ? Number(pieceEndTime) : 0,
    lastSustainEventTime
  );

  if (rawEvents.length === 0) {
    return {
      sustainEvents: [],
      sustainSpans: [],
      pieceEndTime: resolvedPieceEndTime,
      lastSustainEventTime,
    };
  }

  const trackStates = new Map();
  const sustainEvents = [];
  const sustainSpans = [];
  let activeTrackCount = 0;
  let sustainStartTime = null;
  let eventIndex = 0;

  while (eventIndex < rawEvents.length) {
    const currentTime = rawEvents[eventIndex].time;

    // Apply every CC64 update at the same timestamp before evaluating the global pedal state.
    while (eventIndex < rawEvents.length && rawEvents[eventIndex].time === currentTime) {
      const event = rawEvents[eventIndex];
      const wasActive = trackStates.get(event.trackIndex) ?? false;

      if (wasActive !== event.isActive) {
        activeTrackCount += event.isActive ? 1 : -1;
        trackStates.set(event.trackIndex, event.isActive);
      }

      eventIndex += 1;
    }

    const isActive = activeTrackCount > 0;
    const previousEvent = sustainEvents[sustainEvents.length - 1];
    if (previousEvent?.isActive === isActive) {
      continue;
    }

    sustainEvents.push({
      time: currentTime,
      value: isActive ? 1 : 0,
      isActive,
    });

    if (isActive) {
      sustainStartTime = currentTime;
    } else if (sustainStartTime !== null) {
      sustainSpans.push({
        onset: sustainStartTime,
        offset: currentTime,
      });
      sustainStartTime = null;
    }
  }

  if (sustainStartTime !== null) {
    sustainSpans.push({
      onset: sustainStartTime,
      offset: Math.max(sustainStartTime, resolvedPieceEndTime),
    });
  }

  return {
    sustainEvents,
    sustainSpans,
    pieceEndTime: resolvedPieceEndTime,
    lastSustainEventTime,
  };
}

export function extendNotesWithSustain(notes = [], sustainSpans = []) {
  if (!Array.isArray(notes) || notes.length === 0) {
    return [];
  }

  if (!Array.isArray(sustainSpans) || sustainSpans.length === 0) {
    return notes.map((note) => ({ ...note }));
  }

  return notes.map((note) => {
    const noteStart = Number(note.time) || 0;
    const noteDuration = Math.max(0, Number(note.duration) || 0);
    const noteEnd = noteStart + noteDuration;
    const spanIndex = findLastSpanStartingBefore(noteEnd, sustainSpans);

    if (spanIndex === -1) {
      return { ...note };
    }

    const sustainSpan = sustainSpans[spanIndex];
    if (noteEnd < sustainSpan.onset || noteEnd >= sustainSpan.offset) {
      return { ...note };
    }

    const extendedEnd = Math.max(noteEnd, sustainSpan.offset);
    return {
      ...note,
      duration: extendedEnd - noteStart,
    };
  });
}

export function isSustainActiveAtTime(time, sustainEvents = []) {
  if (!Array.isArray(sustainEvents) || sustainEvents.length === 0) {
    return false;
  }

  const eventIndex = findLastSustainEventBefore(time, sustainEvents);
  return eventIndex >= 0 ? Boolean(sustainEvents[eventIndex].isActive) : false;
}
