import { useState, useEffect, useRef, useCallback } from 'react';
import * as Tone from 'tone';

export function usePianoPlayer(notes, duration, baseTempo = 120) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeedState] = useState(1);
  const [volume, setVolumeState] = useState(0.8);
  const [isLoaded, setIsLoaded] = useState(false);

  const samplerRef = useRef(null);
  const partRef = useRef(null);
  const animFrameRef = useRef(null);
  const startOffsetRef = useRef(0);

  // Initialize sampler
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

    return () => {
      sampler.dispose();
      if (partRef.current) partRef.current.dispose();
      Tone.Transport.stop();
      Tone.Transport.cancel();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Schedule notes when they change
  useEffect(() => {
    if (!notes.length || !samplerRef.current) return;

    if (partRef.current) {
      partRef.current.dispose();
    }

    Tone.Transport.cancel();

    const events = notes.map(note => ({
      time: note.time,
      note: note.name || Tone.Frequency(note.midi, 'midi').toNote(),
      duration: note.duration,
      velocity: note.velocity,
    }));

    const part = new Tone.Part((time, event) => {
      if (samplerRef.current && samplerRef.current.loaded) {
        samplerRef.current.triggerAttackRelease(
          event.note,
          event.duration,
          time,
          event.velocity
        );
      }
    }, events.map(e => [e.time, e]));

    part.start(0);
    partRef.current = part;

    // Set transport duration
    Tone.Transport.loopEnd = duration + 1;
  }, [notes, duration]);

  // Update time via animation frame
  const updateTime = useCallback(() => {
    if (Tone.Transport.state === 'started') {
      setCurrentTime(Tone.Transport.seconds);
      animFrameRef.current = requestAnimationFrame(updateTime);
    }
  }, []);

  // Check for end of song
  useEffect(() => {
    if (isPlaying && currentTime >= duration) {
      Tone.Transport.pause();
      setIsPlaying(false);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    }
  }, [currentTime, duration, isPlaying]);

  const play = useCallback(async () => {
    await Tone.start();
    if (currentTime >= duration) {
      Tone.Transport.seconds = 0;
      setCurrentTime(0);
    }
    Tone.Transport.start();
    setIsPlaying(true);
    animFrameRef.current = requestAnimationFrame(updateTime);
  }, [currentTime, duration, updateTime]);

  const pause = useCallback(() => {
    Tone.Transport.pause();
    setIsPlaying(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
  }, []);

  const stop = useCallback(() => {
    Tone.Transport.stop();
    Tone.Transport.seconds = 0;
    setIsPlaying(false);
    setCurrentTime(0);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
  }, []);

  const seek = useCallback((time) => {
    const wasPlaying = Tone.Transport.state === 'started';
    Tone.Transport.seconds = time;
    setCurrentTime(time);
    if (!wasPlaying) {
      // If paused, just update position
    }
  }, []);

  const setSpeed = useCallback((newSpeed) => {
    // Tone.Transport.bpm controls playback rate
    // Default is based on "1 beat per second at 60bpm" but we use seconds-based scheduling
    // So we use playbackRate instead
    Tone.Transport.bpm.value = 60 * newSpeed;
    setSpeedState(newSpeed);
  }, []);

  const setVolume = useCallback((vol) => {
    if (samplerRef.current) {
      // Convert 0-1 range to decibels
      const db = vol === 0 ? -Infinity : 20 * Math.log10(vol);
      samplerRef.current.volume.value = db;
    }
    setVolumeState(vol);
  }, []);

  // Initialize transport BPM to match "1 second = 1 second"
  useEffect(() => {
    Tone.Transport.bpm.value = 60;
  }, []);

  return {
    play, pause, stop, seek,
    setSpeed, setVolume,
    currentTime, duration, isPlaying,
    speed, volume, isLoaded,
  };
}
