"""Raw chart metrics for the regression suite (⑨).

These are objective, audio-independent measurements taken directly from a
generated chart.  They complement chart_evaluator's 0–100 scores with the
concrete numbers the user asked to track: density, diversity, repetition,
hold ratio, hand-movement distance and max jack (consecutive same-lane) run.
"""

from __future__ import annotations

import math
from collections import Counter


def note_density(notes: list[dict], duration: float) -> float:
    if not notes or duration <= 0:
        return 0.0
    return len(notes) / max(1.0, duration)


def hold_ratio(notes: list[dict]) -> float:
    if not notes:
        return 0.0
    holds = sum(1 for n in notes if n.get("kind") == "hold")
    return holds / len(notes)


def pattern_diversity(notes: list[dict], keys: int) -> float:
    """Fraction of unique 3-grams of lanes (0..1)."""
    if len(notes) < 4:
        return 0.0
    lanes = [n["lane"] for n in sorted(notes, key=lambda n: n["time"])]
    grams = [tuple(lanes[i:i + 3]) for i in range(len(lanes) - 2)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_rate(notes: list[dict]) -> float:
    """Fraction of 4-grams that repeat at least once (0..1)."""
    if len(notes) < 8:
        return 0.0
    lanes = [n["lane"] for n in sorted(notes, key=lambda n: n["time"])]
    grams = [tuple(lanes[i:i + 4]) for i in range(len(lanes) - 3)]
    if not grams:
        return 0.0
    counter = Counter(grams)
    repeated = sum(1 for c in counter.values() if c > 1)
    return repeated / len(counter)


def avg_hand_distance(notes: list[dict]) -> float:
    """Average absolute lane jump between consecutive notes."""
    s = sorted(notes, key=lambda n: n["time"])
    if len(s) < 2:
        return 0.0
    dists = [abs(s[i]["lane"] - s[i - 1]["lane"]) for i in range(1, len(s))]
    return sum(dists) / len(dists)


def max_jack_run(notes: list[dict]) -> int:
    """Longest run of consecutive notes on the same lane."""
    s = sorted(notes, key=lambda n: n["time"])
    best = run = 0
    prev = -1
    for n in s:
        if n["lane"] == prev:
            run += 1
        else:
            run = 1
        prev = n["lane"]
        best = max(best, run)
    return best


def max_burst_nps(notes: list[dict], window: float = 0.5) -> float:
    """Peak local note rate (notes/sec) over a sliding window."""
    s = sorted(n["time"] for n in notes)
    if len(s) < 2:
        return 0.0
    best = 0
    j = 0
    for i in range(len(s)):
        while s[i] - s[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return best / window


def grid_alignment(notes: list[dict], bpm: float, beat_offset: float) -> float:
    """Fraction of notes landing on a clean 16th/triplet subdivision."""
    if not notes or bpm <= 0:
        return 0.0
    beat = 60.0 / bpm
    subs = [i / 16.0 for i in range(16)] + [i / 12.0 for i in range(12)]
    tol = 0.05
    aligned = 0
    for n in notes:
        bp = (n["time"] - beat_offset) / beat
        frac = bp - math.floor(bp)
        dist = min(min(abs(frac - x), abs(frac - x - 1.0)) for x in subs)
        if dist < tol:
            aligned += 1
    return aligned / len(notes)


def hold_tap_overlaps(notes: list[dict]) -> int:
    """Count notes that fall inside a hold on the same lane (must be 0)."""
    overlaps = 0
    holds = [n for n in notes if n.get("kind") == "hold" and n.get("duration", 0) > 0]
    for h in holds:
        h_end = h["time"] + h["duration"]
        for m in notes:
            if m is h:
                continue
            if m["lane"] == h["lane"] and h["time"] < m["time"] < h_end:
                overlaps += 1
    return overlaps


def collect_metrics(chart: dict, keys: int) -> dict:
    notes = chart.get("notes", [])
    meta = chart.get("metadata", {})
    bpm = meta.get("bpm", 0.0)
    return {
        "note_count": len(notes),
        "nps": round(note_density(notes, meta.get("duration", 0)), 2),
        "hold_ratio": round(hold_ratio(notes), 3),
        "diversity": round(pattern_diversity(notes, keys), 3),
        "repetition": round(repetition_rate(notes), 3),
        "avg_hand_dist": round(avg_hand_distance(notes), 2),
        "max_jack": max_jack_run(notes),
        "max_burst_nps": round(max_burst_nps(notes), 1),
        "grid_align": round(grid_alignment(notes, bpm, meta.get("beat_offset", 0.0)), 3),
        "overlaps": hold_tap_overlaps(notes),
    }
