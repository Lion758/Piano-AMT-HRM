import { useState, useEffect, useRef, useCallback } from 'react';
import * as Tone from 'tone';

const PLAYBACK_LOOKAHEAD_SECONDS = 0.02;

export function usePianoPlayer(notes, duration, _baseTempo = 120) {
  void _baseTempo;
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeedState] = useState(1);
  const [volume, setVolumeState] = useState(0.8);
  const [isLoaded, setIsLoaded] = useState(false);

  const samplerRef = useRef(null);
  const partRef = useRef(null);
  const noteEventsRef = useRef([]);
  const animFrameRef = useRef(null);
  const currentTimeRef = useRef(0);
  const durationRef = useRef(duration || 0);
  const speedRef = useRef(1);
  const scheduledSpeedRef = useRef(1);
  const isPlayingRef = useRef(false);
  const lastFrameTimeRef = useRef(null);
  const playbackStartTimeRef = useRef(null);

  const setPlayingState = useCallback((nextIsPlaying) => {
    isPlayingRef.current = nextIsPlaying;
    setIsPlaying(nextIsPlaying);
  }, []);

  const clampTime = useCallback((time) => {
    const safeTime = Number.isFinite(time) ? time : 0;
    const maxTime = durationRef.current || 0;
    return maxTime > 0
      ? Math.max(0, Math.min(maxTime, safeTime))
      : Math.max(0, safeTime);
  }, []);

  const syncCurrentTime = useCallback((time) => {
    const nextTime = clampTime(time);
    currentTimeRef.current = nextTime;
    setCurrentTime(nextTime);
    return nextTime;
  }, [clampTime]);

  const stopAnimation = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    lastFrameTimeRef.current = null;
    playbackStartTimeRef.current = null;
  }, []);

  const releaseActiveNotes = useCallback(() => {
    samplerRef.current?.releaseAll?.();
  }, []);

  const clearPlaybackSchedule = useCallback(() => {
    stopAnimation();
    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    Tone.Transport.loop = false;
    partRef.current?.cancel(0);
    releaseActiveNotes();
  }, [releaseActiveNotes, stopAnimation]);

  const advanceSongTime = useCallback((nowMs) => {
    if (!isPlayingRef.current) {
      return currentTimeRef.current;
    }

    if (playbackStartTimeRef.current !== null) {
      if (nowMs < playbackStartTimeRef.current) {
        return currentTimeRef.current;
      }
      lastFrameTimeRef.current = playbackStartTimeRef.current;
      playbackStartTimeRef.current = null;
    }

    if (lastFrameTimeRef.current === null) {
      lastFrameTimeRef.current = nowMs;
      return currentTimeRef.current;
    }

    const deltaSeconds = Math.max(0, (nowMs - lastFrameTimeRef.current) / 1000);
    if (deltaSeconds === 0) {
      return currentTimeRef.current;
    }

    lastFrameTimeRef.current = nowMs;
    return syncCurrentTime(
      currentTimeRef.current + deltaSeconds * scheduledSpeedRef.current
    );
  }, [syncCurrentTime]);

  const finishPlaybackRef = useRef(null);
  finishPlaybackRef.current = (finalTime = currentTimeRef.current) => {
    clearPlaybackSchedule();
    setPlayingState(false);
    syncCurrentTime(finalTime);
  };

  const updatePlaybackFrameRef = useRef(null);
  updatePlaybackFrameRef.current = (timestamp) => {
    if (!isPlayingRef.current) {
      return;
    }

    const nextTime = advanceSongTime(timestamp);
    if (durationRef.current > 0 && nextTime >= durationRef.current) {
      finishPlaybackRef.current(durationRef.current);
      return;
    }

    animFrameRef.current = requestAnimationFrame(updatePlaybackFrameRef.current);
  };

  const triggerHeldNotesAtOffset = useCallback((offset, playbackSpeed, startTime) => {
    if (!samplerRef.current?.loaded) {
      return;
    }

    noteEventsRef.current.forEach((event) => {
      const eventEnd = event.time + event.duration;
      if (event.time >= offset || eventEnd <= offset) {
        return;
      }

      const remainingDuration = (eventEnd - offset) / playbackSpeed;
      if (remainingDuration <= 0) {
        return;
      }

      samplerRef.current.triggerAttackRelease(
        event.note,
        Math.max(remainingDuration, 0.01),
        startTime,
        event.velocity
      );
    });
  }, []);

  const startPlaybackFromOffset = useCallback((offset, playbackSpeed = speedRef.current) => {
    const part = partRef.current;
    if (!part) {
      return false;
    }

    const startOffset = clampTime(offset);
    if (durationRef.current > 0 && startOffset >= durationRef.current) {
      clearPlaybackSchedule();
      setPlayingState(false);
      syncCurrentTime(durationRef.current);
      return false;
    }

    clearPlaybackSchedule();

    scheduledSpeedRef.current = playbackSpeed;
    part.playbackRate = playbackSpeed;
    part.start(0, startOffset);

    const startTime = Tone.now() + PLAYBACK_LOOKAHEAD_SECONDS;
    triggerHeldNotesAtOffset(startOffset, playbackSpeed, startTime);
    Tone.Transport.start(startTime, 0);

    playbackStartTimeRef.current = window.performance.now() + (PLAYBACK_LOOKAHEAD_SECONDS * 1000);
    animFrameRef.current = requestAnimationFrame(updatePlaybackFrameRef.current);
    setPlayingState(true);
    return true;
  }, [clampTime, clearPlaybackSchedule, setPlayingState, syncCurrentTime, triggerHeldNotesAtOffset]);

  // Initialize sampler once
  useEffect(() => {
    const sampler = new Tone.Sampler({
      urls: {
        A0: 'A0.mp3', C1: 'C1.mp3', 'D#1': 'Ds1.mp3', 'F#1': 'Fs1.mp3',
        A1: 'A1.mp3', C2: 'C2.mp3', 'D#2': 'Ds2.mp3', 'F#2': 'Fs2.mp3',
        A2: 'A2.mp3', C3: 'C3.mp3', 'D#3': 'Ds3.mp3', 'F#3': 'Fs3.mp3',
        A3: 'A3.mp3', C4: 'C4.mp3', 'D#4': 'Ds4.mp3', 'F#4': 'Fs4.mp3',
        A4: 'A4.mp3', C5: 'C5.mp3', 'D#5': 'Ds5.mp3', 'F#5': 'Fs5.mp3',
        A5: 'A5.mp3', C6: 'C6.mp3', 'D#6': 'Ds6.mp3', 'F#6': 'Fs6.mp3',
        A6: 'A6.mp3', C7: 'C7.mp3', 'D#7': 'Ds7.mp3', 'F#7': 'Fs7.mp3',
        A7: 'A7.mp3', C8: 'C8.mp3',
      },
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
      onload: () => setIsLoaded(true),
      release: 1,
    }).toDestination();

    samplerRef.current = sampler;

    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    Tone.Transport.loop = false;

    return () => {
      stopAnimation();
      if (partRef.current) {
        partRef.current.dispose();
        partRef.current = null;
      }
      Tone.Transport.stop();
      Tone.Transport.cancel();
      Tone.Transport.seconds = 0;
      sampler.dispose();
      samplerRef.current = null;
    };
  }, [stopAnimation]);

  useEffect(() => {
    durationRef.current = duration || 0;
  }, [duration]);

  // Build note part only when notes/duration change
  useEffect(() => {
    noteEventsRef.current = [];
    clearPlaybackSchedule();
    if (partRef.current) {
      partRef.current.dispose();
      partRef.current = null;
    }

    syncCurrentTime(0);
    setPlayingState(false);

    if (!notes.length || !samplerRef.current) {
      return;
    }

    const events = notes.map((note) => ({
      time: note.time,
      note: note.name || Tone.Frequency(note.midi, 'midi').toNote(),
      duration: note.duration,
      velocity: note.velocity ?? 0.8,
    }));
    noteEventsRef.current = events;

    const part = new Tone.Part((time, event) => {
      if (samplerRef.current && samplerRef.current.loaded) {
        samplerRef.current.triggerAttackRelease(
          event.note,
          Math.max(event.duration / scheduledSpeedRef.current, 0.01),
          time,
          event.velocity
        );
      }
    }, events.map((e) => [e.time, e]));

    part.loop = false;
    part.playbackRate = speedRef.current;
    partRef.current = part;

    Tone.Transport.loop = false;
    Tone.Transport.loopEnd = duration || 0;
  }, [notes, duration, clearPlaybackSchedule, setPlayingState, syncCurrentTime]);

  const play = useCallback(async () => {
    if (!partRef.current) {
      return;
    }

    await Tone.start();

    const startOffset = (durationRef.current > 0 && currentTimeRef.current >= durationRef.current)
      ? 0
      : currentTimeRef.current;

    syncCurrentTime(startOffset);
    startPlaybackFromOffset(startOffset, speedRef.current);
  }, [startPlaybackFromOffset, syncCurrentTime]);

  const pause = useCallback(() => {
    advanceSongTime(window.performance.now());
    clearPlaybackSchedule();
    setPlayingState(false);
  }, [advanceSongTime, clearPlaybackSchedule, setPlayingState]);

  const stop = useCallback(() => {
    clearPlaybackSchedule();
    setPlayingState(false);
    syncCurrentTime(0);
  }, [clearPlaybackSchedule, setPlayingState, syncCurrentTime]);

  const seek = useCallback((time) => {
    const clamped = syncCurrentTime(time);

    if (isPlayingRef.current) {
      startPlaybackFromOffset(clamped, speedRef.current);
    }
  }, [startPlaybackFromOffset, syncCurrentTime]);

  const setSpeed = useCallback((newSpeed) => {
    const safeSpeed = Math.max(0.25, Math.min(2, Number(newSpeed) || 1));
    if (isPlayingRef.current) {
      advanceSongTime(window.performance.now());
    }

    speedRef.current = safeSpeed;
    setSpeedState(safeSpeed);

    if (partRef.current) {
      partRef.current.playbackRate = safeSpeed;
    }

    if (isPlayingRef.current) {
      startPlaybackFromOffset(currentTimeRef.current, safeSpeed);
    }
  }, [advanceSongTime, startPlaybackFromOffset]);

  const setVolume = useCallback((vol) => {
    if (samplerRef.current) {
      const db = vol === 0 ? -Infinity : 20 * Math.log10(vol);
      samplerRef.current.volume.value = db;
    }
    setVolumeState(vol);
  }, []);

  return {
    play,
    pause,
    stop,
    seek,
    setSpeed,
    setVolume,
    currentTime,
    duration,
    isPlaying,
    speed,
    volume,
    isLoaded,
  };
}
