import { useState, useEffect, useRef, useCallback } from 'react';
import * as Tone from 'tone';

const PLAYBACK_LOOKAHEAD_SECONDS = 0.02;
const INITIAL_VOLUME = 0.8;

export function usePianoPlayer(notes, duration, _baseTempo = 120) {
  void _baseTempo;

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeedState] = useState(1);
  const [volume, setVolumeState] = useState(INITIAL_VOLUME);
  const [isLoaded, setIsLoaded] = useState(false);

  const samplerRef = useRef(null);
  const activePartRef = useRef(null);
  const noteEventsRef = useRef([]);
  const animFrameRef = useRef(null);
  const currentTimeRef = useRef(0);
  const durationRef = useRef(duration || 0);
  const speedRef = useRef(1);
  const scheduledSpeedRef = useRef(1);
  const isPlayingRef = useRef(false);
  const lastFrameTimeRef = useRef(null);
  const playbackStartTimeRef = useRef(null);
  const sessionIdRef = useRef(0);

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

  const disposeActivePart = useCallback(() => {
    if (activePartRef.current) {
      activePartRef.current.stop(0);
      activePartRef.current.cancel(0);
      activePartRef.current.dispose();
      activePartRef.current = null;
    }
  }, []);

  const releaseActiveNotes = useCallback(() => {
    samplerRef.current?.releaseAll?.();
  }, []);

  const clearPlaybackSession = useCallback(() => {
    sessionIdRef.current += 1;
    stopAnimation();
    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    Tone.Transport.loop = false;
    disposeActivePart();
    releaseActiveNotes();
  }, [disposeActivePart, releaseActiveNotes, stopAnimation]);

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
      currentTimeRef.current + (deltaSeconds * scheduledSpeedRef.current)
    );
  }, [syncCurrentTime]);

  const triggerHeldNotesAtOffset = useCallback((offset, playbackSpeed, startTime, sessionId) => {
    if (sessionId !== sessionIdRef.current || !samplerRef.current?.loaded) {
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

  const buildPlaybackPart = useCallback((playbackSpeed, sessionId) => {
    const part = new Tone.Part((time, event) => {
      if (event.sessionId !== sessionIdRef.current || !samplerRef.current?.loaded) {
        return;
      }

      samplerRef.current.triggerAttackRelease(
        event.note,
        event.playbackDuration,
        time,
        event.velocity
      );
    }, noteEventsRef.current.map((event) => [
      event.time,
      {
        ...event,
        playbackDuration: Math.max(event.duration / playbackSpeed, 0.01),
        sessionId,
      },
    ]));

    part.loop = false;
    part.playbackRate = playbackSpeed;
    return part;
  }, []);

  const finishPlaybackRef = useRef(null);
  finishPlaybackRef.current = (sessionId, finalTime = currentTimeRef.current) => {
    if (sessionId !== sessionIdRef.current) {
      return;
    }

    clearPlaybackSession();
    setPlayingState(false);
    syncCurrentTime(finalTime);
  };

  const updatePlaybackFrameRef = useRef(null);
  updatePlaybackFrameRef.current = (sessionId, timestamp) => {
    if (sessionId !== sessionIdRef.current || !isPlayingRef.current) {
      return;
    }

    const nextTime = advanceSongTime(timestamp);
    if (durationRef.current > 0 && nextTime >= durationRef.current) {
      finishPlaybackRef.current(sessionId, durationRef.current);
      return;
    }

    animFrameRef.current = requestAnimationFrame((nextTimestamp) => {
      updatePlaybackFrameRef.current(sessionId, nextTimestamp);
    });
  };

  const startPlaybackFromOffset = useCallback((offset, playbackSpeed = speedRef.current) => {
    if (!noteEventsRef.current.length || !samplerRef.current) {
      return false;
    }

    const startOffset = clampTime(offset);
    clearPlaybackSession();

    if (durationRef.current > 0 && startOffset >= durationRef.current) {
      setPlayingState(false);
      syncCurrentTime(durationRef.current);
      return false;
    }

    const sessionId = sessionIdRef.current;
    const part = buildPlaybackPart(playbackSpeed, sessionId);
    activePartRef.current = part;
    scheduledSpeedRef.current = playbackSpeed;

    part.start(0, startOffset);

    const startTime = Tone.now() + PLAYBACK_LOOKAHEAD_SECONDS;
    triggerHeldNotesAtOffset(startOffset, playbackSpeed, startTime, sessionId);
    Tone.Transport.start(startTime, 0);

    playbackStartTimeRef.current = window.performance.now() + (PLAYBACK_LOOKAHEAD_SECONDS * 1000);
    lastFrameTimeRef.current = null;
    animFrameRef.current = requestAnimationFrame((timestamp) => {
      updatePlaybackFrameRef.current(sessionId, timestamp);
    });
    setPlayingState(true);
    return true;
  }, [buildPlaybackPart, clampTime, clearPlaybackSession, setPlayingState, syncCurrentTime, triggerHeldNotesAtOffset]);

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
    sampler.volume.value = 20 * Math.log10(INITIAL_VOLUME);

    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    Tone.Transport.loop = false;

    return () => {
      clearPlaybackSession();
      sampler.dispose();
      samplerRef.current = null;
    };
  }, [clearPlaybackSession]);

  useEffect(() => {
    durationRef.current = duration || 0;
  }, [duration]);

  useEffect(() => {
    noteEventsRef.current = notes.map((note) => ({
      time: note.time,
      note: note.name || Tone.Frequency(note.midi, 'midi').toNote(),
      duration: note.duration,
      velocity: note.velocity ?? 0.8,
    }));

    clearPlaybackSession();
    syncCurrentTime(0);
    setPlayingState(false);
  }, [notes, clearPlaybackSession, setPlayingState, syncCurrentTime]);

  const play = useCallback(async () => {
    if (!noteEventsRef.current.length) {
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
    if (isPlayingRef.current) {
      advanceSongTime(window.performance.now());
    }

    clearPlaybackSession();
    setPlayingState(false);
  }, [advanceSongTime, clearPlaybackSession, setPlayingState]);

  const stop = useCallback(() => {
    clearPlaybackSession();
    setPlayingState(false);
    syncCurrentTime(0);
  }, [clearPlaybackSession, setPlayingState, syncCurrentTime]);

  const seek = useCallback((time) => {
    const clampedTime = syncCurrentTime(time);

    if (isPlayingRef.current) {
      startPlaybackFromOffset(clampedTime, speedRef.current);
    }
  }, [startPlaybackFromOffset, syncCurrentTime]);

  const setSpeed = useCallback((newSpeed) => {
    const safeSpeed = Math.max(0.25, Math.min(2, Number(newSpeed) || 1));

    if (isPlayingRef.current) {
      advanceSongTime(window.performance.now());
    }

    speedRef.current = safeSpeed;
    scheduledSpeedRef.current = safeSpeed;
    setSpeedState(safeSpeed);

    if (isPlayingRef.current) {
      startPlaybackFromOffset(currentTimeRef.current, safeSpeed);
    }
  }, [advanceSongTime, startPlaybackFromOffset]);

  const setVolume = useCallback((vol) => {
    const safeVolume = Math.max(0, Math.min(1, Number(vol) || 0));

    if (samplerRef.current) {
      samplerRef.current.volume.value = safeVolume === 0
        ? -Infinity
        : 20 * Math.log10(safeVolume);
    }

    setVolumeState(safeVolume);
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
