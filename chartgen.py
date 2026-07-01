#!/usr/bin/env python3
"""Generate rhythm-game charts from a YouTube link or local audio file.

The generator uses broad chart-design principles common in arcade and
mobile rhythm games: strong-beat anchoring, phrase-aware density, varied
hand patterns, restrained repetition, and long notes for sustained energy.
It does not copy any specific mapper's authored charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


YOUTUBE_PREFIXES = ("http://", "https://", "www.youtube.com", "youtube.com", "youtu.be")


@dataclass
class Note:
    time: float
    lane: int
    kind: str = "tap"
    duration: float = 0.0
    beat: float = 0.0
    weight: float = 0.0


@dataclass
class Analysis:
    source: str
    sample_rate: int
    duration: float
    bpm: float
    beat_offset: float
    frame_hop: float
    frame_times: list[float]
    onset_strength: list[float]
    rms: list[float]


def is_youtube_source(source: str) -> bool:
    lower = source.lower()
    return lower.startswith(YOUTUBE_PREFIXES) and ("youtu" in lower)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' command was not found. Install it or use a local WAV file.")
    return path


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    require_tool("yt-dlp")
    require_tool("ffmpeg")
    target = work_dir / "youtube_audio.%(ext)s"
    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--output",
        str(target),
        url,
    ]
    subprocess.run(command, check=True)
    wav_files = sorted(work_dir.glob("youtube_audio*.wav"))
    if not wav_files:
        raise RuntimeError("yt-dlp finished, but no WAV file was produced.")
    return wav_files[0]


def load_audio(source: str) -> tuple[np.ndarray, int, str]:
    path = Path(source)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    if is_youtube_source(source):
        temp_dir = tempfile.TemporaryDirectory()
        path = download_youtube_audio(source, Path(temp_dir.name))

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    try:
        sample_rate, data = read_wav(path)
    except Exception as exc:
        raise RuntimeError(
            "Only WAV files are supported without extra decoders. "
            "For YouTube links, install yt-dlp and ffmpeg so the app can convert audio to WAV."
        ) from exc

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 0:
        data /= peak

    # Keep ownership of the temp directory alive by stashing it on the function.
    # The process exits shortly after generation, so this avoids premature cleanup.
    load_audio._temp_dir = temp_dir  # type: ignore[attr-defined]
    return data, int(sample_rate), str(path)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32)
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sample_rate, data


def downsample_for_analysis(audio: np.ndarray, sample_rate: int, target_rate: int = 22050) -> tuple[np.ndarray, int]:
    if sample_rate <= target_rate:
        return audio, sample_rate
    old_times = np.arange(len(audio), dtype=np.float64) / sample_rate
    new_length = int(round(len(audio) * target_rate / sample_rate))
    new_times = np.arange(new_length, dtype=np.float64) / target_rate
    resampled = np.interp(new_times, old_times, audio).astype(np.float32)
    return resampled, target_rate


def find_peak_indices(values: np.ndarray, height: float, distance: int) -> np.ndarray:
    candidates: list[int] = []
    last = -distance
    for idx in range(1, len(values) - 1):
        if idx - last < distance:
            continue
        if values[idx] >= height and values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            candidates.append(idx)
            last = idx
    return np.array(candidates, dtype=int)


def frame_audio(audio: np.ndarray, sample_rate: int, frame_size: int = 1024, hop_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    count = 1 + max(0, (len(audio) - frame_size) // hop_size)
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, frame_size),
        strides=(audio.strides[0] * hop_size, audio.strides[0]),
        writeable=False,
    )
    times = (np.arange(count) * hop_size) / sample_rate
    return frames.copy(), times


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def analyze_audio(source: str) -> Analysis:
    raw_audio, raw_rate, actual_source = load_audio(source)
    audio, sample_rate = downsample_for_analysis(raw_audio, raw_rate)
    frames, frame_times = frame_audio(audio, sample_rate)
    window = np.hanning(frames.shape[1])
    rms = np.sqrt(np.mean((frames * window) ** 2, axis=1))

    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    flux = np.maximum(0, np.diff(spectra, axis=0)).sum(axis=1)
    flux = np.pad(flux, (1, 0))
    onset = normalize(flux) * 0.75 + normalize(np.maximum(0, np.diff(rms, prepend=rms[0]))) * 0.25

    bpm, beat_offset = estimate_tempo(frame_times, onset)
    return Analysis(
        source=actual_source,
        sample_rate=raw_rate,
        duration=len(raw_audio) / raw_rate,
        bpm=bpm,
        beat_offset=beat_offset,
        frame_hop=float(frame_times[1] - frame_times[0]) if len(frame_times) > 1 else 0.0,
        frame_times=frame_times.tolist(),
        onset_strength=onset.tolist(),
        rms=normalize(rms).tolist(),
    )


def estimate_tempo(times: np.ndarray, onset: np.ndarray) -> tuple[float, float]:
    if len(times) < 4:
        return 120.0, 0.0

    hop = float(times[1] - times[0])
    onset_centered = onset - np.mean(onset)
    autocorr = np.correlate(onset_centered, onset_centered, mode="full")[len(onset_centered) - 1 :]

    min_bpm, max_bpm = 70.0, 210.0
    min_lag = max(1, int(round((60.0 / max_bpm) / hop)))
    max_lag = min(len(autocorr) - 1, int(round((60.0 / min_bpm) / hop)))
    if max_lag <= min_lag:
        return 120.0, 0.0

    lag = int(np.argmax(autocorr[min_lag:max_lag]) + min_lag)
    bpm = 60.0 / (lag * hop)

    # Normalize into a playable range.
    while bpm < 90:
        bpm *= 2
    while bpm > 190:
        bpm /= 2

    beat_length = 60.0 / bpm
    peaks = find_peak_indices(onset, height=float(np.percentile(onset, 72)), distance=max(1, int(0.12 / hop)))
    if len(peaks) == 0:
        return round(bpm, 3), 0.0
    peak_times = times[peaks]
    phase = np.median(np.mod(peak_times, beat_length))
    return round(bpm, 3), round(float(phase), 4)


def value_at_time(times: list[float], values: list[float], t: float) -> float:
    if not times:
        return 0.0
    idx = int(np.searchsorted(times, t, side="left"))
    if idx <= 0:
        return float(values[0])
    if idx >= len(values):
        return float(values[-1])
    return float(max(values[idx - 1], values[idx]))


def nearest_peak_weight(times: list[float], onset: list[float], t: float, window: float) -> float:
    left = int(np.searchsorted(times, t - window, side="left"))
    right = int(np.searchsorted(times, t + window, side="right"))
    if right <= left:
        return 0.0
    return float(max(onset[left:right]))


def phrase_energy(analysis: Analysis, phrase_start: float, phrase_end: float) -> float:
    times = analysis.frame_times
    left = int(np.searchsorted(times, phrase_start, side="left"))
    right = int(np.searchsorted(times, phrase_end, side="right"))
    if right <= left:
        return 0.0
    local = np.array(analysis.rms[left:right]) * 0.55 + np.array(analysis.onset_strength[left:right]) * 0.45
    return float(np.mean(local))


def difficulty_settings(name: str) -> dict[str, float]:
    settings = {
        "easy": {"base": 0.27, "accent": 0.56, "subdivision": 2, "long": 0.18},
        "normal": {"base": 0.40, "accent": 0.48, "subdivision": 4, "long": 0.22},
        "hard": {"base": 0.56, "accent": 0.40, "subdivision": 4, "long": 0.27},
        "expert": {"base": 0.72, "accent": 0.33, "subdivision": 8, "long": 0.33},
    }
    return settings[name]


def lane_for_pattern(step_index: int, keys: int, prev_lanes: list[int], accent: float, rng: random.Random) -> int:
    if keys <= 1:
        return 0

    mirrored = [0, keys - 1, 1, keys - 2, keys // 2, max(0, keys // 2 - 1)]
    stair = list(range(keys)) + list(range(keys - 2, 0, -1))
    pattern = mirrored if accent > 0.72 else stair
    lane = pattern[step_index % len(pattern)]

    if prev_lanes and lane == prev_lanes[-1] and rng.random() < 0.78:
        choices = [i for i in range(keys) if i != lane]
        lane = choices[(step_index + len(prev_lanes)) % len(choices)]

    if len(prev_lanes) >= 2 and prev_lanes[-1] == prev_lanes[-2] == lane:
        lane = (lane + 1 + step_index) % keys
    return int(lane)


def generate_chart(analysis: Analysis, keys: int, difficulty: str, seed: int) -> list[Note]:
    rng = random.Random(seed)
    cfg = difficulty_settings(difficulty)
    beat = 60.0 / analysis.bpm
    grid = beat / float(cfg["subdivision"])
    window = min(grid * 0.58, 0.08)
    notes: list[Note] = []
    lanes: list[int] = []

    total_steps = int((analysis.duration - analysis.beat_offset) / grid)
    phrase_beats = 16
    phrase_len = phrase_beats * beat
    phrase_cache: dict[int, float] = {}

    for step in range(max(0, total_steps)):
        t = analysis.beat_offset + step * grid
        if t < 0.15 or t > analysis.duration - 0.05:
            continue

        beat_pos = (t - analysis.beat_offset) / beat
        beat_frac = beat_pos - math.floor(beat_pos)
        strong = min(abs(beat_frac), abs(beat_frac - 1.0)) < 0.045
        half = abs(beat_frac - 0.5) < 0.05
        offbeat = abs(beat_frac - 0.25) < 0.05 or abs(beat_frac - 0.75) < 0.05

        phrase_idx = int(t // phrase_len)
        if phrase_idx not in phrase_cache:
            start = phrase_idx * phrase_len
            phrase_cache[phrase_idx] = phrase_energy(analysis, start, start + phrase_len)
        section = phrase_cache[phrase_idx]

        onset = nearest_peak_weight(analysis.frame_times, analysis.onset_strength, t, window)
        energy = value_at_time(analysis.frame_times, analysis.rms, t)
        contour = 0.6 * onset + 0.25 * energy + 0.15 * section
        threshold = float(cfg["accent"])
        place = contour >= threshold

        if strong and energy > 0.12 and rng.random() < cfg["base"]:
            place = True
        if half and onset > threshold * 0.92 and rng.random() < cfg["base"] * 0.85:
            place = True
        if offbeat and difficulty in {"hard", "expert"} and onset > threshold * 0.8 and rng.random() < 0.46:
            place = True

        # Leave breathing room in low-energy intros and outros.
        if energy < 0.08 and not strong:
            place = False
        if not place:
            continue

        lane = lane_for_pattern(step, keys, lanes, onset, rng)
        kind = "tap"
        duration = 0.0

        sustain = energy > 0.62 and onset < 0.58 and strong and rng.random() < cfg["long"]
        if sustain:
            kind = "hold"
            duration = round(beat * rng.choice([1.0, 1.5, 2.0]), 4)

        notes.append(Note(time=round(t, 4), lane=lane, kind=kind, duration=duration, beat=round(beat_pos, 4), weight=round(contour, 4)))
        lanes.append(lane)

        # Add occasional chords on climactic strong beats, with restrained density.
        if (
            keys >= 4
            and strong
            and difficulty in {"hard", "expert"}
            and onset > 0.82
            and energy > 0.45
            and rng.random() < (0.12 if difficulty == "hard" else 0.22)
        ):
            second = keys - 1 - lane
            if second != lane:
                notes.append(
                    Note(time=round(t, 4), lane=second, kind="tap", duration=0.0, beat=round(beat_pos, 4), weight=round(contour, 4))
                )
                lanes.append(second)

    return sorted(notes, key=lambda note: (note.time, note.lane))


def chart_metadata(analysis: Analysis, keys: int, difficulty: str, notes: list[Note]) -> dict[str, object]:
    taps = sum(1 for note in notes if note.kind == "tap")
    holds = sum(1 for note in notes if note.kind == "hold")
    return {
        "generator": "rhythm-chart-generator",
        "source": analysis.source,
        "duration": round(analysis.duration, 3),
        "bpm": analysis.bpm,
        "beat_offset": analysis.beat_offset,
        "keys": keys,
        "difficulty": difficulty,
        "note_count": len(notes),
        "tap_count": taps,
        "hold_count": holds,
        "design_notes": [
            "Strong beats are anchored when energy supports them.",
            "Onset peaks drive accents and chord candidates.",
            "Lane selection avoids excessive repeated hits and alternates hand shapes.",
            "Phrase energy raises density during climactic sections.",
        ],
    }


def write_json(path: Path, metadata: dict[str, object], notes: list[Note]) -> None:
    payload = {"metadata": metadata, "notes": [asdict(note) for note in notes]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, notes: Iterable[Note]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "lane", "kind", "duration", "beat", "weight"])
        writer.writeheader()
        for note in notes:
            writer.writerow(asdict(note))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a rhythm-game chart from YouTube or a local WAV file.")
    parser.add_argument("source", help="YouTube URL or local WAV file path")
    parser.add_argument("--keys", type=int, default=4, choices=[4, 5, 6, 7, 8], help="Number of playable lanes")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "normal", "hard", "expert"])
    parser.add_argument("--seed", type=int, default=20260629, help="Seed for repeatable pattern variation")
    parser.add_argument("--out", default="chart.json", help="Output chart JSON path")
    parser.add_argument("--csv", default="", help="Optional CSV note list path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_audio(args.source)
        notes = generate_chart(analysis, args.keys, args.difficulty, args.seed)
        metadata = chart_metadata(analysis, args.keys, args.difficulty, notes)

        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, metadata, notes)
        if args.csv:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(csv_path, notes)

        print(f"Generated {len(notes)} notes")
        print(f"BPM: {analysis.bpm} / Beat offset: {analysis.beat_offset}s")
        print(f"JSON: {output.resolve()}")
        if args.csv:
            print(f"CSV: {Path(args.csv).resolve()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
