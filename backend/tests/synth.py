"""Synthetic test-song generator for the regression suite (⑨).

Produces deterministic WAV files with a known ground-truth BPM, a simple
drum pattern (kick on 1&3, snare on 2&4, hi-hats on 8ths) and a melodic
sine line.  Because the BPM is known exactly, the BPM-detection accuracy of
the analyzer can be measured objectively.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t * 28)
    freq = 110 * np.exp(-t * 18) + 45
    return np.sin(2 * np.pi * freq * t) * env


def _snare(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t * 22)
    noise = np.random.RandomState(7).randn(n)
    tone = np.sin(2 * np.pi * 190 * t)
    return (noise * 0.7 + tone * 0.3) * env


def _hat(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t * 80)
    noise = np.random.RandomState(3).randn(n)
    return noise * env * 0.4


def synth_song(
    bpm: float,
    duration: float = 30.0,
    melody_notes: list[float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Render a mono float32 waveform with the given BPM."""
    rng = np.random.RandomState(seed)
    total = int(duration * SR)
    out = np.zeros(total, dtype=np.float32)

    beat = 60.0 / bpm
    eighth = beat / 2
    hit_len = int(0.18 * SR)
    hat_len = int(0.05 * SR)

    kick = _kick(hit_len)
    snare = _snare(hit_len)
    hat = _hat(hat_len)

    # Place drums on a grid.
    n_beats = int(duration / beat)
    for b in range(n_beats):
        tpos = b * beat
        idx = int(tpos * SR)
        beat_in_bar = b % 4
        if beat_in_bar in (0, 2) and idx + hit_len < total:
            out[idx:idx + hit_len] += kick
        if beat_in_bar in (1, 3) and idx + hit_len < total:
            out[idx:idx + hit_len] += snare * 0.9
        # hats on every 8th
        for h in (0, 1):
            hidx = int((tpos + h * eighth) * SR)
            if hidx + hat_len < total:
                out[hidx:hidx + hat_len] += hat

    # Melodic sine line that changes each bar.
    scale = [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3]
    if melody_notes is None:
        melody_notes = [scale[rng.randint(0, len(scale))] for _ in range(n_beats)]
    for b in range(n_beats):
        f = melody_notes[b % len(melody_notes)]
        idx = int(b * beat * SR)
        seg = int(beat * SR)
        if idx + seg >= total:
            seg = total - idx
        if seg <= 0:
            continue
        t = np.arange(seg) / SR
        env = np.minimum(1.0, np.exp(-t * 1.5) + 0.2)
        out[idx:idx + seg] += (np.sin(2 * np.pi * f * t) * env * 0.25).astype(np.float32)

    # Normalize.
    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.9
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Genre-specific synthesis (large-scale validation)
#
# The single-style synth_song above is enough to measure BPM accuracy, but to
# exercise the analyzer and chart engine across real musical variety we need
# songs whose *spectra* genuinely differ: a piano solo has no drums and rich
# harmonics; EDM is four-on-the-floor with a sidechained bass; metal is dense,
# broadband and distorted; jazz swings; a ballad is vocal-dominant.  Each
# genre below toggles instrument layers and rhythmic feel accordingly.
# ---------------------------------------------------------------------------

_SCALE = [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3, 587.3, 659.3]
# Simple 4-chord progressions expressed as scale-degree triads (indices).
_PROGRESSIONS = [
    [(0, 2, 4), (5, 0, 2), (3, 5, 0), (4, 6, 1)],   # I–vi–IV–V-ish
    [(0, 2, 4), (3, 5, 0), (4, 6, 1), (0, 2, 4)],
    [(5, 0, 2), (3, 5, 0), (0, 2, 4), (4, 6, 1)],
]

GENRES = [
    "edm", "jpop", "kpop", "rock", "piano",
    "orchestra", "metal", "jazz", "vocal", "drums",
]


def _tone(freq: float, n: int, decay: float = 2.0,
          harmonics=(1.0, 0.5, 0.25, 0.12), attack: float = 80.0,
          vibrato: float = 0.0) -> np.ndarray:
    """Additive pitched tone with a quick attack and exponential decay."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / SR
    env = np.exp(-t * decay) * (1.0 - np.exp(-t * attack))
    vib = 1.0 + (vibrato * np.sin(2 * np.pi * 5.5 * t) if vibrato else 0.0)
    sig = np.zeros(n)
    for h, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(2 * np.pi * freq * h * vib * t)
    return (sig * env).astype(np.float32)


def _pad(freq: float, n: int, harmonics=(1.0, 0.6, 0.4, 0.3, 0.2)) -> np.ndarray:
    """Slow-attack sustained tone for strings / synth pads (soft onset)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / SR
    env = (1.0 - np.exp(-t * 6.0)) * np.exp(-t * 0.25)
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 4.5 * t)
    sig = np.zeros(n)
    for h, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(2 * np.pi * freq * h * vib * t)
    return (sig * env * 0.5).astype(np.float32)


def _vocal(freq: float, n: int, seed: int = 0) -> np.ndarray:
    """Formant-based vocal-like tone: fundamental + resonant formants + vibrato."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / SR
    env = (1.0 - np.exp(-t * 8.0)) * np.minimum(1.0, np.exp(-t * 0.6) + 0.3)
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * t)
    sig = np.zeros(n)
    # Rich glottal source.
    for h, amp in enumerate((1.0, 0.7, 0.5, 0.35, 0.22, 0.15), start=1):
        sig += amp * np.sin(2 * np.pi * freq * h * vib * t)
    # Two formant resonances in the vocal range.
    for formant, amp in ((700.0, 0.5), (1220.0, 0.35)):
        sig += amp * np.sin(2 * np.pi * formant * t) * np.exp(-t * 1.5)
    return (sig * env * 0.4).astype(np.float32)


def _distort(sig: np.ndarray, drive: float = 4.0) -> np.ndarray:
    """Soft-clip distortion → broadband harmonics (electric/metal guitar)."""
    return np.tanh(sig * drive).astype(np.float32)


def _place(out: np.ndarray, sample: np.ndarray, idx: int, gain: float = 1.0) -> None:
    if sample.size == 0:
        return
    end = idx + len(sample)
    if idx < 0 or end > len(out):
        end = min(end, len(out))
        sample = sample[: end - idx]
    if end > idx:
        out[idx:end] += sample * gain


# Per-genre layer configuration.
_GENRE_CFG: dict[str, dict] = {
    "edm":       {"drums": "four_floor", "bass": "steady", "harmony": "pad",
                  "lead": "saw", "lead_gain": 0.22, "bright": 1.0},
    "jpop":      {"drums": "backbeat",   "bass": "steady", "harmony": "piano",
                  "lead": "vocal", "lead_gain": 0.30, "bright": 0.9},
    "kpop":      {"drums": "backbeat",   "bass": "steady", "harmony": "pad",
                  "lead": "vocal", "lead_gain": 0.28, "bright": 1.0},
    "rock":      {"drums": "backbeat",   "bass": "steady", "harmony": "guitar",
                  "lead": "guitar", "lead_gain": 0.25, "bright": 0.85},
    "piano":     {"drums": "none",       "bass": "none",   "harmony": "piano",
                  "lead": "piano", "lead_gain": 0.30, "bright": 0.7},
    "orchestra": {"drums": "none",       "bass": "low_pad", "harmony": "pad",
                  "lead": "pad", "lead_gain": 0.28, "bright": 0.6},
    "metal":     {"drums": "double_kick", "bass": "steady", "harmony": "distorted",
                  "lead": "distorted", "lead_gain": 0.24, "bright": 0.9},
    "jazz":      {"drums": "swing",      "bass": "walking", "harmony": "piano",
                  "lead": "vocal", "lead_gain": 0.22, "bright": 0.75},
    "vocal":     {"drums": "soft",       "bass": "none",   "harmony": "piano",
                  "lead": "vocal", "lead_gain": 0.40, "bright": 0.8},
    "drums":     {"drums": "busy",       "bass": "steady", "harmony": "none",
                  "lead": "none", "lead_gain": 0.0, "bright": 1.0},
}


def _render_drums(out: np.ndarray, bpm: float, duration: float,
                  pattern: str, rng: np.random.RandomState) -> None:
    if pattern == "none":
        return
    beat = 60.0 / bpm
    eighth = beat / 2
    sixteenth = beat / 4
    hit = int(0.18 * SR)
    hat_len = int(0.05 * SR)
    kick, snare, hat = _kick(hit), _snare(hit), _hat(hat_len)
    crash = _hat(int(0.25 * SR)) * 1.5
    n_beats = int(duration / beat)

    for b in range(n_beats):
        tpos = b * beat
        idx = int(tpos * SR)
        bar_pos = b % 4

        if pattern == "four_floor":
            _place(out, kick, idx, 1.0)
            if bar_pos in (1, 3):
                _place(out, snare, idx, 0.8)
            for h in range(2):
                _place(out, hat, int((tpos + eighth * (h + 0.5)) * SR), 0.6)
        elif pattern in ("backbeat", "soft"):
            g = 0.5 if pattern == "soft" else 1.0
            if bar_pos in (0, 2):
                _place(out, kick, idx, g)
            if bar_pos in (1, 3):
                _place(out, snare, idx, 0.9 * g)
            for h in range(2):
                _place(out, hat, int((tpos + h * eighth) * SR), 0.4 * g)
        elif pattern == "double_kick":
            for s in range(4):
                _place(out, kick, int((tpos + s * sixteenth) * SR), 0.9)
            if bar_pos in (1, 3):
                _place(out, snare, idx, 1.0)
            if bar_pos == 0:
                _place(out, crash, idx, 0.7)
        elif pattern == "swing":
            # Ride on beats + swung offbeat, brush snare, sparse kick.
            _place(out, hat, idx, 0.5)
            _place(out, hat, int((tpos + eighth * 1.33) * SR), 0.35)
            if bar_pos in (1, 3):
                _place(out, snare, idx, 0.4)
            if bar_pos == 0:
                _place(out, kick, idx, 0.6)
        elif pattern == "busy":
            _place(out, kick, idx, 1.0)
            if bar_pos in (1, 3):
                _place(out, snare, idx, 0.9)
            for h in range(4):
                _place(out, hat, int((tpos + h * sixteenth) * SR), 0.5)
            if b % 8 == 7:  # fill
                for s in range(4):
                    _place(out, snare, int((tpos + s * sixteenth) * SR), 0.7)


def _render_bass(out: np.ndarray, bpm: float, duration: float,
                 mode: str, prog, rng: np.random.RandomState) -> None:
    if mode == "none":
        return
    beat = 60.0 / bpm
    n_beats = int(duration / beat)
    seg = int(beat * SR)
    for b in range(n_beats):
        chord = prog[(b // 4) % len(prog)]
        root_idx = chord[0]
        root = _SCALE[root_idx % len(_SCALE)] / 2.0  # one octave down
        idx = int(b * beat * SR)
        if mode == "walking":
            note = _SCALE[chord[b % len(chord)] % len(_SCALE)] / 2.0
            _place(out, _tone(note, seg, decay=3.0, harmonics=(1.0, 0.4)), idx, 0.35)
        elif mode == "low_pad":
            _place(out, _pad(root, seg, harmonics=(1.0, 0.5)), idx, 0.4)
        else:  # steady
            _place(out, _tone(root, seg, decay=4.0, harmonics=(1.0, 0.3)), idx, 0.4)


def _render_harmony(out: np.ndarray, bpm: float, duration: float,
                    mode: str, prog, bright: float) -> None:
    if mode == "none":
        return
    beat = 60.0 / bpm
    n_beats = int(duration / beat)
    seg = int(beat * SR)
    for b in range(n_beats):
        if mode in ("piano", "guitar", "distorted") and b % 2 != 0:
            continue  # strum/comp on half notes for these
        chord = prog[(b // 4) % len(prog)]
        idx = int(b * beat * SR)
        acc = np.zeros(seg, dtype=np.float32)
        for deg in chord:
            f = _SCALE[deg % len(_SCALE)] * (1.0 + 0.15 * (bright - 0.8))
            if mode == "piano":
                acc += _tone(f, seg, decay=2.2, harmonics=(1.0, 0.5, 0.3, 0.15))
            elif mode == "guitar":
                acc += _tone(f, seg, decay=3.0, harmonics=(1.0, 0.6, 0.4, 0.25))
            elif mode == "distorted":
                acc += _tone(f, seg, decay=2.5, harmonics=(1.0, 0.8, 0.6, 0.5))
            elif mode == "pad":
                acc += _pad(f, seg)
        if mode == "distorted":
            acc = _distort(acc, drive=3.5)
        _place(out, acc, idx, 0.18)


def _render_lead(out: np.ndarray, bpm: float, duration: float, cfg: dict,
                 prog, rng: np.random.RandomState) -> None:
    mode = cfg["lead"]
    if mode == "none":
        return
    gain = cfg["lead_gain"]
    beat = 60.0 / bpm
    n_beats = int(duration / beat)
    # Lead note per beat drawn from the current chord tones + passing notes.
    for b in range(n_beats):
        chord = prog[(b // 4) % len(prog)]
        deg = chord[rng.randint(0, len(chord))] + rng.choice([0, 0, 0, 1, -1])
        f = _SCALE[deg % len(_SCALE)]
        idx = int(b * beat * SR)
        seg = int(beat * SR)
        if mode == "vocal":
            _place(out, _vocal(f, seg, seed=b), idx, gain)
        elif mode == "piano":
            _place(out, _tone(f, seg, decay=2.5, harmonics=(1.0, 0.5, 0.25)), idx, gain)
        elif mode == "guitar":
            _place(out, _tone(f, seg, decay=3.0, harmonics=(1.0, 0.6, 0.4, 0.2)), idx, gain)
        elif mode == "distorted":
            _place(out, _distort(_tone(f, seg, decay=2.5,
                    harmonics=(1.0, 0.7, 0.5)), drive=4.0), idx, gain)
        elif mode == "saw":
            saw = _tone(f, seg, decay=1.5,
                        harmonics=tuple(1.0 / h for h in range(1, 8)))
            _place(out, saw, idx, gain)
        elif mode == "pad":
            _place(out, _pad(f, seg), idx, gain)


def synth_genre(genre: str, bpm: float, duration: float = 22.0,
                seed: int = 0) -> np.ndarray:
    """Render a genre-flavored mono waveform with a known BPM.

    Distinct genres produce distinct spectra (drum patterns, harmonic content,
    distortion, vocal formants) so the analyzer and chart engine are exercised
    across realistic musical variety, not just one timbre.
    """
    cfg = _GENRE_CFG.get(genre, _GENRE_CFG["kpop"])
    rng = np.random.RandomState(seed)
    total = int(duration * SR)
    out = np.zeros(total, dtype=np.float32)
    prog = _PROGRESSIONS[seed % len(_PROGRESSIONS)]

    _render_drums(out, bpm, duration, cfg["drums"], rng)
    _render_bass(out, bpm, duration, cfg["bass"], prog, rng)
    _render_harmony(out, bpm, duration, cfg["harmony"], prog, cfg["bright"])
    _render_lead(out, bpm, duration, cfg, prog, rng)

    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.9
    return out.astype(np.float32)


def write_wav(path: Path, wave_data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(wave_data, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())
    return path
