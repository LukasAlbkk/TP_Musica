"""Renders a PrettyMIDI song to a .wav file, purely for human listening
(the symbolic MIDI file is the actual deliverable/genotype-derived artifact).

No external binaries or soundfonts are required: a small numpy softsynth
(additive "distorted-guitar" tone for melodic tracks, noise-based drum hits)
produces a self-contained, reproducible render. If a General MIDI soundfont
and `pyfluidsynth` are available, pass `soundfont_path` for a more realistic
render instead -- see README.md for optional setup instructions.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from . import theory

SAMPLE_RATE = 22050


def _note_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _envelope(n: int, sr: int, attack=0.006, release=0.04) -> np.ndarray:
    env = np.ones(n)
    a = min(int(sr * attack), n // 2)
    r = min(int(sr * release), n // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _melodic_tone(pitch: int, duration: float, velocity: int, sr: int) -> np.ndarray:
    n = max(1, int(duration * sr))
    t = np.arange(n) / sr
    freq = _note_freq(pitch)
    wave_sig = np.zeros(n)
    for k in range(1, 8):
        wave_sig += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    peak = np.max(np.abs(wave_sig)) + 1e-9
    wave_sig /= peak
    return wave_sig * _envelope(n, sr) * (velocity / 127.0)


def _kick(sr: int, velocity: int) -> np.ndarray:
    n = int(0.15 * sr)
    t = np.arange(n) / sr
    freq = np.linspace(150.0, 45.0, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 18) * (velocity / 127.0)


def _snare(sr: int, velocity: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.15 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    noise /= np.max(np.abs(noise)) + 1e-9
    tone = np.sin(2 * np.pi * 180 * t)
    mix = 0.6 * noise + 0.4 * tone
    return mix * np.exp(-t * 14) * (velocity / 127.0)


def _hihat(sr: int, velocity: int, rng: np.random.Generator, open_: bool) -> np.ndarray:
    dur = 0.25 if open_ else 0.06
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    hp = np.diff(noise, prepend=0.0)
    hp /= np.max(np.abs(hp)) + 1e-9
    decay = 6 if open_ else 40
    return hp * np.exp(-t * decay) * (velocity / 127.0) * 0.7


def _crash(sr: int, velocity: int, rng: np.random.Generator) -> np.ndarray:
    n = int(1.0 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    hp = np.diff(noise, prepend=0.0)
    hp /= np.max(np.abs(hp)) + 1e-9
    return hp * np.exp(-t * 2.2) * (velocity / 127.0) * 0.8


def _write_wav(path: Path, buffer: np.ndarray, sr: int) -> None:
    data = np.clip(buffer, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def _softsynth_render(pmid, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = pmid.get_end_time() + 1.0
    buffer = np.zeros(int(total * sr) + sr)

    for inst in pmid.instruments:
        for note in inst.notes:
            start_sample = int(note.start * sr)
            if inst.is_drum:
                if note.pitch == theory.DRUM_KICK:
                    seg = _kick(sr, note.velocity)
                elif note.pitch == theory.DRUM_SNARE:
                    seg = _snare(sr, note.velocity, rng)
                elif note.pitch == theory.DRUM_HIHAT_CLOSED:
                    seg = _hihat(sr, note.velocity, rng, open_=False)
                elif note.pitch == theory.DRUM_HIHAT_OPEN:
                    seg = _hihat(sr, note.velocity, rng, open_=True)
                elif note.pitch == theory.DRUM_CRASH:
                    seg = _crash(sr, note.velocity, rng)
                else:
                    continue
            else:
                duration = max(note.end - note.start, 0.02)
                seg = _melodic_tone(note.pitch, duration, note.velocity, sr)

            end_sample = start_sample + len(seg)
            if end_sample > len(buffer):
                buffer = np.pad(buffer, (0, end_sample - len(buffer)))
            buffer[start_sample:end_sample] += seg

    peak = np.max(np.abs(buffer))
    if peak > 0:
        buffer = buffer / peak * 0.9
    return buffer


def render_wav(pmid, out_path, sr: int = SAMPLE_RATE, seed: int = 0, soundfont_path: str | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if soundfont_path:
        try:
            audio = pmid.fluidsynth(fs=sr, sf2_path=soundfont_path)
            _write_wav(out_path, audio, sr)
            return
        except Exception as exc:  # pragma: no cover - optional path
            print(f"[synth] FluidSynth render failed ({exc}); falling back to built-in softsynth.")

    buffer = _softsynth_render(pmid, sr, seed)
    _write_wav(out_path, buffer, sr)
