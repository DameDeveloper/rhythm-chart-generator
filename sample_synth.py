#!/usr/bin/env python3
"""Create a tiny synthetic WAV file for testing the chart generator."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def main() -> None:
    sample_rate = 44100
    bpm = 128
    duration = 24.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    beat = 60.0 / bpm

    audio = 0.08 * np.sin(2 * np.pi * 220 * t)
    audio += 0.04 * np.sin(2 * np.pi * 440 * t)

    for i in range(int(duration / beat)):
        start = int(i * beat * sample_rate)
        length = int(0.08 * sample_rate)
        env = np.exp(-np.linspace(0, 8, length))
        tone = np.sin(2 * np.pi * (70 if i % 4 == 0 else 180) * np.arange(length) / sample_rate)
        audio[start : start + length] += 0.75 * tone * env

    for i in range(int(duration / (beat / 2))):
        if i % 4 in (1, 3):
            start = int(i * beat / 2 * sample_rate)
            length = int(0.035 * sample_rate)
            env = np.exp(-np.linspace(0, 10, length))
            noise = np.random.default_rng(i).normal(0, 1, length)
            audio[start : start + length] += 0.2 * noise * env

    audio /= np.max(np.abs(audio))
    output = Path("sample_128bpm.wav")
    pcm = (audio * 32767).astype("<i2")
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    print(output.resolve())


if __name__ == "__main__":
    main()
