"""Chart generation engine — Steps 11-15, 19-20.

Produces rhythm-game charts with structure-aware density, varied patterns,
hand-movement optimization, and humanization.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

from audio_pipeline import AnalysisResult, FocusSegment, Section, TempoPoint, bpm_at, focus_at
from pattern_loader import get_patterns, get_transition_matrix

import numpy as np


@dataclass
class Note:
    time: float
    lane: int
    kind: str = "tap"
    duration: float = 0.0
    beat: float = 0.0
    weight: float = 0.0


# ---------------------------------------------------------------------------
# Lane colour scheme (DJMAX-style)
# ---------------------------------------------------------------------------

LANE_COLORS: dict[int, list[str]] = {
    4: ["#4a9eff", "#ffffff", "#ffffff", "#4a9eff"],
    5: ["#4a9eff", "#ffffff", "#ffcc00", "#ffffff", "#4a9eff"],
    6: ["#4a9eff", "#ffffff", "#ff4444", "#ff4444", "#ffffff", "#4a9eff"],
    7: ["#4a9eff", "#ffffff", "#ff4444", "#ffcc00", "#ff4444", "#ffffff", "#4a9eff"],
    8: ["#4a9eff", "#ffffff", "#ff4444", "#ffcc00", "#ffcc00", "#ff4444", "#ffffff", "#4a9eff"],
}

# ---------------------------------------------------------------------------
# Difficulty configs (Step 13)
# ---------------------------------------------------------------------------

DIFFICULTIES = {
    "easy":   {"base": 0.25, "accent": 0.58, "subdiv": 2,  "long": 0.18, "chord": 0.00, "max_density": 3},
    "normal": {"base": 0.40, "accent": 0.48, "subdiv": 4,  "long": 0.22, "chord": 0.04, "max_density": 5},
    "hard":   {"base": 0.56, "accent": 0.40, "subdiv": 4,  "long": 0.27, "chord": 0.12, "max_density": 7},
    "expert": {"base": 0.72, "accent": 0.33, "subdiv": 8,  "long": 0.33, "chord": 0.22, "max_density": 10},
    "master": {"base": 0.85, "accent": 0.25, "subdiv": 16, "long": 0.38, "chord": 0.30, "max_density": 14},
}


# ---------------------------------------------------------------------------
# Charting style profiles (Step 12/19)
#
# These do NOT copy any artist's authored charts. They encode the *design
# philosophy* that distinguishes how different games' chart artists translate
# a song into notes:
#
#   djmax  — instrument-driven (DJMAX style). Notes track the rhythm/instrument
#            layers: kick & snare anchor strong beats, hi-hats fill streams,
#            the lead melody steers lane motion. Generous chords, jacks, and
#            trills in climaxes; balanced two-hand reading.
#   sekai  — vocal-driven (Project Sekai style). Notes hug the vocal line: the
#            melody contour steers lanes strongly, sustained vocals become long
#            notes, melodic runs become flowing stairs. Fewer chords, more holds,
#            phrase-end emphasis.
#   auto   — a balance of both.
# ---------------------------------------------------------------------------

STYLES = {
    "djmax": {
        "pitch_follow": 0.55,   # how strongly the lead melody steers lanes
        "drum_anchor": 0.75,    # kick/snare drive strong-beat placement
        "vocal_follow": 0.30,   # weight given to the vocal layer
        "chord_mult": 1.4,      # density of chords/jacks
        "hold_mult": 0.8,       # fewer holds, more crisp taps
        "trill_bias": 0.55,     # trill/zigzag tendency in dense parts
        "buildup": 0.5,         # pre-chorus density ramp
    },
    "sekai": {
        "pitch_follow": 0.85,
        "drum_anchor": 0.40,
        "vocal_follow": 0.90,
        "chord_mult": 0.55,
        "hold_mult": 1.6,
        "trill_bias": 0.20,
        "buildup": 0.7,
    },
    "auto": {
        "pitch_follow": 0.65,
        "drum_anchor": 0.55,
        "vocal_follow": 0.55,
        "chord_mult": 1.0,
        "hold_mult": 1.0,
        "trill_bias": 0.35,
        "buildup": 0.55,
    },
}


# Per-focus-instrument style overrides.  Each maps to deltas applied on top
# of the base style so the chart naturally shifts character when the lead
# instrument changes — vocals get more holds and pitch following, drums get
# anchored hits and trills, guitar solos get streams, etc.
FOCUS_STYLE_DELTAS: dict[str, dict[str, float]] = {
    "vocal":  {"pitch_follow": +0.20, "vocal_follow": +0.25, "hold_mult": +0.5,
               "drum_anchor": -0.15, "trill_bias": -0.15, "chord_mult": -0.3},
    "drums":  {"drum_anchor": +0.25, "trill_bias": +0.20, "pitch_follow": -0.15,
               "vocal_follow": -0.20, "hold_mult": -0.3, "chord_mult": +0.2},
    "guitar": {"pitch_follow": +0.15, "trill_bias": +0.15, "hold_mult": -0.2,
               "drum_anchor": -0.10, "vocal_follow": -0.10, "chord_mult": +0.1},
    "bass":   {"drum_anchor": +0.10, "pitch_follow": -0.10, "hold_mult": +0.2,
               "trill_bias": -0.10, "vocal_follow": -0.05, "chord_mult": 0.0},
    "keys":   {"pitch_follow": +0.10, "chord_mult": +0.15, "hold_mult": +0.1,
               "drum_anchor": -0.05, "trill_bias": -0.05, "vocal_follow": 0.0},
    "mixed":  {},
}


def focus_adjusted_style(base_style: dict, focus: FocusSegment) -> dict:
    """Return a copy of *base_style* adjusted for the current focus instrument."""
    deltas = FOCUS_STYLE_DELTAS.get(focus.instrument, {})
    if not deltas:
        return base_style
    adjusted = dict(base_style)
    blend = focus.confidence
    for key, delta in deltas.items():
        if key in adjusted:
            adjusted[key] = max(0.0, min(1.6, adjusted[key] + delta * blend))
    return adjusted


def pitch_to_lane(melody_val: float, keys: int) -> int:
    lane = int(round(melody_val * (keys - 1)))
    return max(0, min(keys - 1, lane))


# ---------------------------------------------------------------------------
# Instrument classification & hand model
# ---------------------------------------------------------------------------

def classify_hit(low: float, mid: float, high: float) -> str:
    """Classify the dominant instrument from frequency band energies."""
    if low > 0.45 and low >= mid and low >= high:
        return "kick"
    if low > 0.3 and mid > 0.3 and mid >= high:
        return "snare"
    if mid > 0.4 and mid > low:
        return "vocal"
    if high > 0.4 and high > mid:
        return "hihat"
    return "other"


def instrument_anchors(instrument: str, keys: int) -> list[int]:
    """Preferred lanes per instrument — creates natural hand separation."""
    mid = keys // 2
    if instrument == "kick":
        return [0, keys - 1]
    elif instrument == "snare":
        if keys <= 4:
            return [1, 2] if keys == 4 else [1, keys - 2]
        return list(range(max(1, mid - 1), min(keys - 1, mid + 2)))
    elif instrument == "hihat":
        return list(range(mid, keys))
    return list(range(keys))


def chord_partner(lane: int, instrument: str, keys: int) -> int:
    """Pick a second lane that forms a natural two-hand chord shape."""
    mid = keys // 2
    if instrument == "kick":
        return keys - 1 if lane < mid else 0
    elif instrument == "snare":
        if keys <= 4:
            return 2 if lane <= 1 else 1
        adj = lane + 1 if lane < mid else lane - 1
        return max(0, min(keys - 1, adj))
    return keys - 1 - lane


@dataclass
class HandState:
    last_left: int = -1
    last_right: int = -1
    last_hand: int = -1
    direction: int = 1
    dir_steps: int = 0

    def which_hand(self, lane: int, keys: int) -> int:
        return 0 if lane < keys / 2 else 1

    def preferred_hand(self) -> int:
        if self.last_hand == -1:
            return 0
        return 1 - self.last_hand

    def update(self, lane: int, keys: int) -> None:
        hand = self.which_hand(lane, keys)
        if hand == 0:
            self.last_left = lane
        else:
            self.last_right = lane
        self.last_hand = hand

    def update_direction(self, lane: int, prev_lane: int) -> None:
        if lane > prev_lane:
            new_dir = 1
        elif lane < prev_lane:
            new_dir = -1
        else:
            return
        if new_dir == self.direction:
            self.dir_steps += 1
        else:
            self.direction = new_dir
            self.dir_steps = 1


# ---------------------------------------------------------------------------
# Pattern library — multi-note phrase templates
#
# Each pattern is a sequence of normalised lane positions (0.0 = leftmost,
# 1.0 = rightmost).  A single float is one note; a tuple of floats is a
# chord.  The scheduler maps them to concrete lanes at runtime.
#
# Patterns carry metadata so the scheduler can pick the right one for the
# current musical context (section, intensity, style, difficulty).
# ---------------------------------------------------------------------------

DIFF_IDX = {"easy": 0, "normal": 1, "hard": 2, "expert": 3, "master": 4}


@dataclass
class PatternDef:
    name: str
    steps: list                     # float | tuple[float, ...]
    sections: tuple[str, ...]       # matching section labels
    onset_range: tuple[float, float]  # (min, max) onset intensity
    styles: tuple[str, ...]         # matching styles, or ("all",)
    min_diff: int                   # minimum DIFF_IDX
    weight: float = 1.0            # selection probability weight


def _p(name: str, steps: list, sections: tuple, onset: tuple,
       styles: tuple = ("all",), md: int = 0, w: float = 1.0) -> PatternDef:
    return PatternDef(name, steps, sections, onset, styles, md, w)


PATTERN_LIBRARY: list[PatternDef] = [
    # ── 계단 (stairs) ──
    _p("stairs_up",   [0.0, 0.33, 0.67, 1.0],
       ("verse", "chorus", "bridge", "intro", "outro"), (0.0, 1.0), ("all",), 0, 1.5),
    _p("stairs_down", [1.0, 0.67, 0.33, 0.0],
       ("verse", "chorus", "bridge", "intro", "outro"), (0.0, 1.0), ("all",), 0, 1.5),
    _p("stairs_long", [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0],
       ("chorus",), (0.5, 1.0), ("all",), 2, 1.0),

    # ── 역계단 + 꺾기 (reverse with direction change) ──
    _p("stairs_bounce", [0.0, 0.33, 0.67, 1.0, 0.67, 0.33],
       ("verse", "chorus"), (0.3, 1.0), ("all",), 1, 1.2),
    _p("stairs_rev_bounce", [1.0, 0.67, 0.33, 0.0, 0.33, 0.67],
       ("verse", "chorus"), (0.3, 1.0), ("all",), 1, 1.2),

    # ── 교차 (cross / outside-inside) ──
    _p("cross_out_in", [0.0, 1.0, 0.33, 0.67],
       ("verse", "chorus"), (0.3, 1.0), ("djmax", "auto"), 1, 1.3),
    _p("cross_in_out", [0.33, 0.67, 0.0, 1.0],
       ("verse", "chorus"), (0.3, 1.0), ("djmax", "auto"), 1, 1.3),
    _p("cross_alt",    [0.0, 0.67, 0.33, 1.0, 0.0, 0.67],
       ("chorus",), (0.5, 1.0), ("djmax",), 2, 1.0),

    # ── 계단 + 동시 (stairs with chords) ──
    _p("stairs_chord_up",   [0.0, (0.33, 1.0), 0.67, (0.0, 1.0)],
       ("chorus",), (0.6, 1.0), ("djmax", "auto"), 2, 1.2),
    _p("stairs_chord_down", [1.0, (0.0, 0.67), 0.33, (0.0, 1.0)],
       ("chorus",), (0.6, 1.0), ("djmax", "auto"), 2, 1.2),
    _p("stairs_chord_alt",  [(0.0, 0.67), 0.33, (0.33, 1.0), 0.0],
       ("chorus",), (0.7, 1.0), ("djmax",), 3, 1.0),

    # ── 트릴 (trill) ──
    _p("trill_outer",  [0.0, 1.0, 0.0, 1.0],
       ("chorus", "verse"), (0.5, 1.0), ("djmax", "auto"), 2, 1.4),
    _p("trill_inner",  [0.33, 0.67, 0.33, 0.67],
       ("chorus", "verse", "bridge"), (0.4, 1.0), ("all",), 1, 1.2),
    _p("trill_wide",   [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
       ("chorus",), (0.7, 1.0), ("djmax",), 3, 1.0),

    # ── 잭 (jack — intentional same-lane repeat) ──
    _p("jack_left_2",  [0.0, 0.0, 0.33, 0.67],
       ("chorus",), (0.7, 1.0), ("djmax",), 3, 0.8),
    _p("jack_right_2", [1.0, 1.0, 0.67, 0.33],
       ("chorus",), (0.7, 1.0), ("djmax",), 3, 0.8),
    _p("jack_inner",   [0.33, 0.33, 0.67, 0.67],
       ("chorus",), (0.6, 1.0), ("djmax", "auto"), 2, 0.7),

    # ── 폭타 (burst / mash — dense chord spam) ──
    _p("burst_lr", [(0.0, 0.33), (0.67, 1.0), (0.0, 0.33), (0.67, 1.0)],
       ("chorus",), (0.8, 1.0), ("djmax",), 3, 1.0),
    _p("burst_converge", [(0.0, 1.0), (0.33, 0.67), (0.0, 1.0), 0.5],
       ("chorus",), (0.85, 1.0), ("djmax",), 4, 0.8),
    _p("burst_spread",   [0.5, (0.0, 1.0), (0.33, 0.67), (0.0, 0.33, 0.67, 1.0)],
       ("chorus",), (0.9, 1.0), ("djmax",), 4, 0.6),

    # ── 롱노트 위 멜로디 (melody over hold) ──
    _p("hold_melody_l", [0.0, 0.33, 0.67, 0.33, 0.67, 1.0],
       ("verse", "chorus", "bridge"), (0.25, 0.7), ("sekai", "auto"), 1, 1.3),
    _p("hold_melody_r", [1.0, 0.67, 0.33, 0.67, 0.33, 0.0],
       ("verse", "chorus", "bridge"), (0.25, 0.7), ("sekai", "auto"), 1, 1.3),

    # ── 나선 (spiral — expanding outward then contracting) ──
    _p("spiral_out", [0.5, 0.33, 0.67, 0.0, 1.0],
       ("chorus",), (0.5, 1.0), ("all",), 2, 1.0),
    _p("spiral_in",  [0.0, 1.0, 0.33, 0.67, 0.5],
       ("chorus", "bridge"), (0.4, 0.9), ("all",), 2, 1.0),

    # ── 원핸드 (one-hand stream) ──
    _p("onehand_l", [0.0, 0.33, 0.0, 0.33],
       ("verse", "bridge"), (0.3, 0.7), ("sekai", "auto"), 2, 0.9),
    _p("onehand_r", [0.67, 1.0, 0.67, 1.0],
       ("verse", "bridge"), (0.3, 0.7), ("sekai", "auto"), 2, 0.9),

    # ── 더블스텝 (double step — each position twice) ──
    _p("double_step", [0.0, 0.0, 0.67, 0.67, 0.33, 0.33, 1.0, 1.0],
       ("chorus",), (0.6, 1.0), ("djmax",), 3, 0.7),

    # ── 보컬 웨이브 (vocal wave — gentle melodic flow) ──
    _p("vocal_wave",      [0.33, 0.5, 0.67, 0.5, 0.33, 0.17],
       ("verse", "bridge"), (0.15, 0.6), ("sekai",), 0, 1.4),
    _p("vocal_rise_fall", [0.17, 0.33, 0.5, 0.67, 0.83, 0.67, 0.5, 0.33],
       ("verse", "chorus"), (0.2, 0.7), ("sekai",), 1, 1.2),
]


def _norm_to_lane(val: float, keys: int) -> int:
    return max(0, min(keys - 1, int(round(val * (keys - 1)))))


# ---------------------------------------------------------------------------
# Pattern candidate scoring (⑧ "AI score" — feature-based ranking)
#
# Instead of a flat weighted-random pick, we gather every eligible pattern,
# score each one against the live musical features (onset, energy, treble,
# melody direction, intensity) plus flow heuristics (recently-used penalty),
# and select the best — with a touch of softmax randomness so identical
# inputs don't always yield the identical pattern.  This is a learned-style
# scorer expressed as hand-tuned features; it makes pattern choice feel
# intentional rather than random.
# ---------------------------------------------------------------------------

def _pattern_category(pat) -> str:
    if isinstance(pat, PatternDef):
        name = pat.name
        # Infer category from inline pattern names.
        for key in ("stairs", "cross", "trill", "jack", "burst", "hold",
                    "spiral", "onehand", "double", "vocal"):
            if key in name:
                return {
                    "stairs": "stair", "cross": "stair", "spiral": "stream",
                    "onehand": "swing", "double": "stream", "vocal": "swing",
                }.get(key, key)
        return "misc"
    return pat.get("category", "misc")


def _pattern_steps(pat) -> list:
    return pat.steps if isinstance(pat, PatternDef) else pat["steps"]


def _pattern_name(pat) -> str:
    return pat.name if isinstance(pat, PatternDef) else pat["name"]


def _transform_steps(steps: list, tx: str) -> list:
    """Apply a spatial transform to a pattern's normalized (0–1) step sequence.

    Works on both single notes (float) and chords (tuple of floats):
      identity — unchanged
      reverse  — play the sequence backwards
      mirror   — flip left↔right (v → 1 - v)
    These are the same musical transforms a human mapper uses to generate a
    fresh-feeling variation of a familiar lane figure.
    """
    if tx == "identity":
        return list(steps)

    def flip(v):
        if isinstance(v, tuple):
            return tuple(round(1.0 - x, 4) for x in v)
        return round(1.0 - float(v), 4)

    if tx == "reverse":
        return list(reversed(steps))
    if tx == "mirror":
        return [flip(v) for v in steps]
    if tx == "rev_mirror":
        return [flip(v) for v in reversed(steps)]
    return list(steps)


# How well each category fits a given musical situation.  Each entry is a
# function of the live feature context returning a 0–1 affinity.
def score_pattern_candidate(pat, ctx: dict) -> float:
    """Return a score for *pat* given the musical context *ctx*.

    ctx keys: onset, energy, high, low, mid, intensity, melody_dir,
              recent_names (list), difficulty_idx
    """
    return _score_features(
        _pattern_name(pat), _pattern_category(pat), _pattern_steps(pat), ctx,
    )


def _score_features(name: str, cat: str, steps: list, ctx: dict) -> float:
    """Score a candidate given its *name*, *category*, and *steps*.

    Split out from ``score_pattern_candidate`` so the candidate search can
    score transformed step variants directly, not just whole PatternDefs.
    """
    onset = ctx.get("onset", 0.5)
    energy = ctx.get("energy", 0.5)
    high = ctx.get("high", 0.0)
    mid = ctx.get("mid", 0.0)
    intensity = ctx.get("intensity", 0.5)
    melody_dir = ctx.get("melody_dir", 0.0)   # -1 falling .. +1 rising
    recent = ctx.get("recent_names", [])

    score = 1.0

    # --- 1. Category ↔ instrument/feature affinity ---
    if cat == "burst":
        score += 1.6 * onset + 1.0 * energy            # bursts want loud hits
    elif cat == "jack":
        score += 1.4 * onset + 0.8 * (1.0 if intensity > 0.7 else 0.0)
    elif cat == "trill":
        score += 1.2 * high + 0.6 * onset              # trills track hi-hats
    elif cat == "stream":
        score += 1.0 * energy + 0.8 * intensity
    elif cat == "stair":
        score += 0.9 + 0.8 * abs(melody_dir)           # stairs follow melody slope
    elif cat == "hold":
        score += 1.4 * mid + 0.8 * (1.0 - onset)       # holds want sustained vocal
    elif cat == "chord":
        score += 1.1 * energy + 0.7 * onset
    elif cat == "swing":
        score += 0.8 + 0.6 * (1.0 - onset)             # swing = gentler feel

    # --- 2. Melody-direction match for directional patterns ---
    rising = steps and isinstance(steps[0], (int, float)) and isinstance(steps[-1], (int, float)) and steps[-1] > steps[0]
    falling = steps and isinstance(steps[0], (int, float)) and isinstance(steps[-1], (int, float)) and steps[-1] < steps[0]
    if rising and melody_dir > 0.15:
        score += 0.8 * melody_dir
    elif falling and melody_dir < -0.15:
        score += 0.8 * (-melody_dir)
    elif (rising and melody_dir < -0.3) or (falling and melody_dir > 0.3):
        score -= 0.5   # wrong-way pattern vs melody

    # --- 3. Intensity-appropriate length ---
    length = len(steps)
    if intensity > 0.75 and length >= 6:
        score += 0.4
    elif intensity < 0.4 and length <= 4:
        score += 0.3
    elif intensity < 0.4 and length >= 7:
        score -= 0.4   # too busy for a quiet part

    # --- 4. Diversity: penalize recently-used patterns ---
    if name in recent:
        idx = recent[::-1].index(name)
        score -= max(0.2, 1.2 - idx * 0.25)

    # --- 5. Transition matrix: favor natural pattern flow ---
    prev_cat = ctx.get("prev_category", "")
    if prev_cat:
        tm = get_transition_matrix()
        row = tm.get(prev_cat, {})
        if row:
            tw = row.get(cat, 0.5)
            score *= (0.4 + tw * 0.4)

    # --- 6. Category fatigue: after 3+ same-category in a row, nudge away ---
    cat_run = ctx.get("cat_run", 0)
    if cat_run >= 3 and cat == prev_cat:
        score *= max(0.25, 1.0 - (cat_run - 2) * 0.2)

    # --- 7. Extended feature-vector affinity (⑩) ---
    # Pull the richer descriptors when present; default to neutral otherwise.
    vocal_p = ctx.get("vocal_presence", 0.0)
    drum_f = ctx.get("drum_fill", 0.0)
    tension = ctx.get("tension", 0.0)
    harmonic = ctx.get("harmonic_ratio", 0.5)
    chord_chg = ctx.get("chord_change", 0.0)
    inst_chg = ctx.get("instrument_change", 0.0)

    if vocal_p or drum_f or tension:  # only when features were supplied
        # Vocal line → sustained/gentle shapes (holds, swing, stairs).
        if cat in ("hold", "swing", "stair"):
            score += 0.9 * vocal_p
        elif cat in ("burst", "jack"):
            score -= 0.5 * vocal_p     # avoid mashing over a soft vocal

        # Drum fill → dense percussive shapes (bursts, jacks, trills).
        if cat in ("burst", "jack", "trill"):
            score += 1.1 * drum_f
        elif cat in ("hold", "swing"):
            score -= 0.6 * drum_f

        # Tension build → longer, denser patterns feel like a ramp.
        length = len(steps)
        if tension > 0.55 and length >= 6:
            score += 0.8 * tension
        elif tension < 0.35 and length <= 4:
            score += 0.3 * (1.0 - tension)

        # Harmonic (tonal) vs percussive character.
        if cat in ("stair", "hold"):
            score += 0.5 * harmonic          # melodic shapes over tonal audio
        if cat in ("trill", "jack"):
            score += 0.5 * (1.0 - harmonic)  # rolls over percussive audio

        # A fresh chord / instrument change is a natural place for a new
        # pattern *category* — reward switching away from the previous one.
        if (chord_chg > 0.5 or inst_chg > 0.5) and cat != prev_cat:
            score += 0.4 * max(chord_chg, inst_chg)

    return max(0.01, score)


class PatternState:
    """Runtime state for the pattern scheduler.

    Manages two pattern sources:
    * Inline ``PATTERN_LIBRARY`` — the hardcoded 29 patterns
    * JSON-loaded patterns from ``patterns/*.json`` — loaded once at startup

    Section Memory (④): remembers which patterns were used in each section
    type+occurrence.  When the same section reappears (e.g. 2nd chorus), it
    reuses the same pattern sequence with small variations, like a human
    chart maker would.

    Pattern Scoring (⑧): candidates are ranked by ``score_pattern_candidate``
    against live musical features and chosen via softmax, not flat random.
    """

    # Fraction of a repeated section's patterns reused verbatim from the
    # previous occurrence (the rest are re-searched → intentional variation).
    SECTION_REUSE_RATE = 0.70

    def __init__(self, rng: random.Random, keys: int = 4):
        self.rng = rng
        self.keys = keys
        self.active: PatternDef | None = None
        self.active_json: dict | None = None
        # When a transformed variant of a pattern is chosen, its transformed
        # step sequence is stored here and consumed in place of the base steps.
        self._active_steps: list | None = None
        self.cursor: int = 0
        self.cooldown: int = 0
        # Diagnostics: how the last selection was made (for inspection/tuning).
        self.last_selection: dict | None = None

        # Section memory: section_key -> list of pattern names used
        self.section_memory: dict[str, list[str]] = {}
        self._current_section_key: str = ""
        self._section_pattern_idx: int = 0
        self._section_occurrence: dict[str, int] = {}

        # Rolling window of recently-chosen pattern names (for diversity score)
        self.recent_names: list[str] = []
        self.last_category: str = ""
        self._cat_run: int = 0

    def enter_section(self, label: str) -> None:
        """Call when the section changes to track section occurrences."""
        occ = self._section_occurrence.get(label, 0) + 1
        self._section_occurrence[label] = occ
        self._current_section_key = f"{label}_{occ}"
        self._section_pattern_idx = 0

    def _json_patterns(self) -> list[dict]:
        return get_patterns()

    def try_start(
        self, section: str, onset: float, style_name: str, difficulty: str,
        intensity: float = 0.5, ctx: dict | None = None,
    ) -> None:
        """Maybe pick a new pattern if none is active and cooldown has elapsed.

        *ctx* carries live musical features used by the ⑧ scorer (onset,
        energy, high/mid/low band, intensity, melody_dir).  When omitted a
        minimal context is built from the scalar args.
        """
        if self.active is not None or self.active_json is not None or self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - 1)
            return
        di = DIFF_IDX.get(difficulty, 2)

        if ctx is None:
            ctx = {"onset": onset, "intensity": intensity}
        ctx = dict(ctx)
        ctx["recent_names"] = self.recent_names
        ctx["difficulty_idx"] = di
        ctx["prev_category"] = self.last_category
        ctx["cat_run"] = self._cat_run

        # --- Check section memory first: reuse patterns from previous occurrence ---
        prev_key = self._find_previous_section_key(section)
        if prev_key and prev_key in self.section_memory:
            prev_patterns = self.section_memory[prev_key]
            if self._section_pattern_idx < len(prev_patterns):
                pname = prev_patterns[self._section_pattern_idx]
                self._section_pattern_idx += 1
                # Reuse the remembered pattern ~70% of the time; the other
                # ~30% falls through to a fresh candidate search (variation).
                if self.rng.random() < self.SECTION_REUSE_RATE:
                    found = self._find_pattern_by_name(pname)
                    if found is not None:
                        if isinstance(found, PatternDef):
                            self.active = found
                        else:
                            self.active_json = found
                        self.cursor = 0
                        self._active_steps = None
                        cat = _pattern_category(found)
                        self._record_pattern(pname, cat)
                        self.last_selection = {
                            "name": pname, "category": cat,
                            "transform": "reuse", "score": 0.0,
                            "pool_size": 0, "shortlist": [],
                        }
                        return

        # --- Gather every eligible candidate from BOTH sources ---
        candidates: list = [
            p for p in self._json_patterns()
            if section in p["sections"]
            and di >= p["min_diff"]
            and intensity >= p.get("intensity_min", 0.0)
            and self.keys >= p.get("min_keys", 4)
        ]
        candidates += [
            p for p in PATTERN_LIBRARY
            if section in p.sections
            and p.onset_range[0] <= onset <= p.onset_range[1]
            and (style_name in p.styles or "all" in p.styles)
            and di >= p.min_diff
        ]
        if not candidates:
            return

        # --- ⑧ Candidate search: generate many → score → top-K → final pick ---
        base, steps, cat, tx, diag = self._candidate_search(candidates, ctx)
        if isinstance(base, PatternDef):
            self.active = base
            self._record_pattern(base.name, cat)
        else:
            self.active_json = base
            self._record_pattern(base["name"], cat)
        # Only store an override when the chosen variant differs from the base.
        self._active_steps = steps if tx != "identity" else None
        self.last_selection = diag
        self.cursor = 0

    # Number of top candidates to shortlist before the final weighted pick.
    SHORTLIST_K = 3
    # Cap on the expanded candidate pool to keep scoring bounded.
    MAX_POOL = 24

    def _candidate_search(self, candidates: list, ctx: dict):
        """Generate a candidate pool, score all, shortlist top-K, pick one.

        This is the "make many candidates, keep the good ones" approach:

          1. **Generate** — each eligible base pattern spawns 1–3 variants via
             spatial transforms (identity / mirror / reverse), expanding the
             pool (e.g. 8 patterns → ~20 candidates).
          2. **Evaluate** — every variant is scored by ``_score_features``
             against the live musical context, with a diversity bonus for
             transformed variants of recently-used patterns.
          3. **Shortlist** — keep the top ``SHORTLIST_K`` by score.
          4. **Final pick** — softmax-sample among the shortlist so the best
             usually wins but the chart stays fresh.

        Returns ``(base_pattern, chosen_steps, category, transform, diag)``.
        """
        recent = ctx.get("recent_names", [])

        # (1) Generate the expanded candidate pool.
        pool: list[tuple] = []  # (base, steps, cat, name, transform, score)
        for base in candidates:
            name = _pattern_name(base)
            cat = _pattern_category(base)
            base_steps = _pattern_steps(base)

            variants: list[tuple[str, list]] = [("identity", list(base_steps))]
            # Mirror is musically valid for any pattern (flip left↔right).
            variants.append(("mirror", _transform_steps(base_steps, "mirror")))
            # Reverse only for directional/streaming shapes where it reads well.
            if cat in ("stair", "stream", "swing"):
                variants.append(("reverse", _transform_steps(base_steps, "reverse")))

            for tx, st in variants:
                score = _score_features(name, cat, st, ctx)
                # Reward a fresh variation of a pattern we've used recently.
                if tx != "identity" and name in recent:
                    score *= 1.2
                # Slightly discount transforms so identity wins ties (stability).
                elif tx != "identity":
                    score *= 0.97
                pool.append((base, st, cat, name, tx, score))

            if len(pool) >= self.MAX_POOL:
                break

        # (3) Shortlist the top-K by score.
        pool.sort(key=lambda c: c[5], reverse=True)
        shortlist = pool[:self.SHORTLIST_K]

        # (4) Softmax final pick among the shortlist.
        scores = [c[5] for c in shortlist]
        temp = 0.5
        mx = max(scores)
        exps = [math.exp((s - mx) / temp) for s in scores]
        total = sum(exps)
        r = self.rng.random() * total
        cumul = 0.0
        chosen = shortlist[-1]
        for c, e in zip(shortlist, exps):
            cumul += e
            if r <= cumul:
                chosen = c
                break

        base, steps, cat, name, tx, score = chosen
        diag = {
            "name": name,
            "category": cat,
            "transform": tx,
            "score": round(score, 3),
            "pool_size": len(pool),
            "shortlist": [(c[3], c[4], round(c[5], 2)) for c in shortlist],
        }
        return base, steps, cat, tx, diag

    def _record_pattern(self, name: str, category: str = "") -> None:
        key = self._current_section_key
        if key not in self.section_memory:
            self.section_memory[key] = []
        self.section_memory[key].append(name)
        self.recent_names.append(name)
        if len(self.recent_names) > 8:
            self.recent_names = self.recent_names[-8:]
        if category:
            if category == self.last_category:
                self._cat_run += 1
            else:
                self._cat_run = 1
            self.last_category = category

    def _find_previous_section_key(self, label: str) -> str | None:
        occ = self._section_occurrence.get(label, 0)
        if occ <= 1:
            return None
        return f"{label}_{occ - 1}"

    def _find_pattern_by_name(self, name: str) -> PatternDef | dict | None:
        for p in PATTERN_LIBRARY:
            if p.name == name:
                return p
        for p in self._json_patterns():
            if p["name"] == name:
                return p
        return None

    def next_step(self, keys: int, free_lanes: list[int]) -> list[int] | None:
        """Consume the next step from the active pattern (or its variant)."""
        steps: list | None = None
        if self._active_steps is not None:
            steps = self._active_steps
        elif self.active is not None:
            steps = self.active.steps
        elif self.active_json is not None:
            steps = self.active_json["steps"]

        if steps is None:
            return None
        if self.cursor >= len(steps):
            self._finish()
            return None

        raw = steps[self.cursor]
        self.cursor += 1

        if isinstance(raw, tuple):
            lanes = list(dict.fromkeys(_norm_to_lane(v, keys) for v in raw))
            lanes = [l for l in lanes if l in free_lanes]
        elif isinstance(raw, list):
            lanes = list(dict.fromkeys(_norm_to_lane(float(v), keys) for v in raw))
            lanes = [l for l in lanes if l in free_lanes]
        else:
            lane = _norm_to_lane(float(raw), keys)
            if lane in free_lanes:
                lanes = [lane]
            elif free_lanes:
                lanes = [min(free_lanes, key=lambda l: abs(l - lane))]
            else:
                lanes = []

        if not lanes:
            self._finish()
            return None

        if self.cursor >= len(steps):
            self._finish()

        return lanes

    def _finish(self) -> None:
        self.cooldown = self.rng.randint(2, 6)
        self.active = None
        self.active_json = None
        self._active_steps = None
        self.cursor = 0

    @property
    def is_active(self) -> bool:
        return self.active is not None or self.active_json is not None


# ---------------------------------------------------------------------------
# Phrase-level planner (③ — generate per 4-bar phrase, then vary on repeat)
#
# Human mappers think in phrases: they hear a 4-bar melodic figure, design a
# lane figure for it (도레미파 → 1234), and when the melody returns they
# *vary* it (4321, mirror, shift) instead of re-deriving from scratch.
#
# This planner:
#   * fingerprints each phrase's melody contour into a coarse signature,
#   * records the lane sequence actually played in that phrase,
#   * on a phrase whose signature matches a remembered one, replays the
#     stored lanes through a transform (reverse / mirror / shift), giving the
#     "same idea, varied" feel.
#
# It only drives *melody-following* taps; drum-anchored notes, chords and
# library patterns keep priority, so the planner layers on without fighting
# the rest of the engine.
# ---------------------------------------------------------------------------

PHRASE_TRANSFORMS = ("identity", "reverse", "mirror", "rev_mirror", "shift")

SECTION_TRANSFORMS = ("identity", "mirror", "shift_up", "shift_down")


class SectionReplay:
    """Section-level lane sequence memory with block-based 70/30 variation.

    Records the actual lane indices chosen during each section occurrence
    (e.g. chorus_1 → [1,2,3,4,1,2,3,4,...]).  When the same section label
    appears again (chorus_2), it **pre-bakes** a variation:

        Verse1 → save  →  Verse2 = 70% reused verbatim + 30% varied

    Crucially the variation is applied in **contiguous blocks**, not per-note
    coin flips.  A human mapper reuses whole lane runs from the first verse
    and only *reworks* a bar or two — so most of the section is immediately
    recognizable while a few phrases feel intentionally different.  The old
    per-note random dropout produced choppy, incoherent replays; block
    variation keeps long identical runs intact.
    """

    # Fraction of the previous section's lanes replayed verbatim.
    REUSE_RATE = 0.70
    # Transforms applied to the ~30% varied blocks.
    BLOCK_TRANSFORMS = ("mirror", "shift_up", "shift_down", "reverse")

    def __init__(self, rng: random.Random, keys: int):
        self.rng = rng
        self.keys = keys
        self._lanes: dict[str, list[int]] = {}
        self._current_label: str = ""
        self._occurrence: dict[str, int] = {}
        self._replay: list[int] | None = None
        self._replay_pos: int = 0
        self._recording: list[int] = []
        # Diagnostics: reuse stats for the section currently being replayed.
        self.last_variation: dict | None = None

    def enter_section(self, label: str) -> None:
        if self._current_label and self._recording:
            key = f"{self._current_label}_{self._occurrence.get(self._current_label, 1)}"
            self._lanes[key] = list(self._recording)

        occ = self._occurrence.get(label, 0) + 1
        self._occurrence[label] = occ
        self._current_label = label
        self._recording = []
        self._replay = None
        self._replay_pos = 0

        if occ >= 2:
            prev_key = f"{label}_{occ - 1}"
            prev = self._lanes.get(prev_key)
            if prev and len(prev) >= 3:
                self._replay = self._build_variation(prev)
                self._replay_pos = 0

    def _build_variation(self, prev: list[int]) -> list[int]:
        """Reuse ~70% of *prev* verbatim; rework ~30% in contiguous blocks."""
        n = len(prev)
        result = list(prev)
        n_vary_target = int(round(n * (1.0 - self.REUSE_RATE)))
        if n_vary_target <= 0 or n < 4:
            self.last_variation = {"reused": n, "varied": 0, "blocks": []}
            return result

        block = max(2, n // 8)
        starts = list(range(0, n, block))
        self.rng.shuffle(starts)

        varied = 0
        blocks: list[tuple[int, int, str]] = []
        for s in starts:
            if varied >= n_vary_target:
                break
            e = min(s + block, n)
            tx = self.rng.choice(self.BLOCK_TRANSFORMS)
            result[s:e] = self._transform(result[s:e], tx)
            blocks.append((s, e, tx))
            varied += (e - s)

        self.last_variation = {
            "reused": n - varied, "varied": varied,
            "reuse_pct": round((n - varied) / n * 100, 1),
            "blocks": blocks,
        }
        return result

    def _transform(self, lanes: list[int], tx: str) -> list[int]:
        k = self.keys
        if tx == "mirror":
            return [k - 1 - l for l in lanes]
        if tx == "shift_up":
            return [min(k - 1, l + 1) for l in lanes]
        if tx == "shift_down":
            return [max(0, l - 1) for l in lanes]
        if tx == "reverse":
            return list(reversed(lanes))
        return list(lanes)

    def next_lane(self, free_lanes: list[int]) -> int | None:
        """Walk the pre-baked variation deterministically (variation is baked in)."""
        if self._replay is None or self._replay_pos >= len(self._replay):
            return None
        lane = self._replay[self._replay_pos]
        self._replay_pos += 1
        if lane in free_lanes:
            return lane
        if free_lanes:
            return min(free_lanes, key=lambda l: abs(l - lane))
        return None

    def record(self, lane: int) -> None:
        self._recording.append(lane)

    @property
    def is_replaying(self) -> bool:
        return self._replay is not None and self._replay_pos < len(self._replay)

    def finalize(self) -> None:
        if self._current_label and self._recording:
            key = f"{self._current_label}_{self._occurrence.get(self._current_label, 1)}"
            self._lanes[key] = list(self._recording)


MOTIF_TRANSFORMS = ("identity", "reverse", "mirror", "rev_mirror", "shift")


@dataclass
class _MotifEntry:
    """A stored motif: contour signature → lane sequence, with usage stats."""
    contour: tuple[int, ...]
    lanes: list[int]
    hits: int = 1
    last_transform: str = "identity"


class MotifGenerator:
    """Contour-based motif detection and replay — the core of professional charting.

    What changed vs the old absolute-pitch matcher:

    1. **Contour matching**: motifs are stored as *interval sequences* (the
       direction and magnitude between successive notes), not absolute pitch
       buckets.  ``미솔라도`` = intervals (+2, +1, +2).  Later, ``레파솔시``
       = intervals (+2, +1, +2) — same contour → match, even though the
       absolute pitches differ.  This is how human ears (and chart makers)
       recognize "the same phrase" in a different key.

    2. **Variable-length detection**: tries to match at lengths 8, 6, 4
       (longest first).  Longer matches produce more satisfying replays.

    3. **Fuzzy matching**: allows up to 1 interval mismatch in a contour,
       catching slight melodic embellishments that a human mapper would
       still treat as "the same phrase".

    4. **Progressive transforms**: tracks how many times each motif has been
       replayed.  1st replay ≈ identity/shift (subtle), 2nd = mirror,
       3rd+ = reverse/rev_mirror (dramatic).  ``1234 → 1234 → 1243 → 4321``

    5. **Frequency-weighted memory**: motifs that appear more often get
       higher matching priority, preventing one-off noise from dominating.

    6. **Cross-section persistence**: the motif *memory* survives section
       changes (a chorus melody recognized in a later chorus), only the
       running input buffer resets.
    """

    MOTIF_LENGTHS = (8, 6, 4)
    N_BUCKETS = 12
    MAX_MEMORY = 64

    def __init__(self, rng: random.Random, keys: int):
        self.rng = rng
        self.keys = keys
        self.prev_melody: float = 0.5
        self._pitch_buf: list[int] = []
        self._lane_buf: list[int] = []
        self._memory: list[_MotifEntry] = []
        self._replay: list[int] | None = None
        self._replay_pos: int = 0

    def _quantize(self, melody_val: float) -> int:
        return max(0, min(self.N_BUCKETS - 1, int(melody_val * self.N_BUCKETS)))

    @staticmethod
    def _to_contour(pitches: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        """Convert absolute pitch buckets to an interval contour.

        Each interval is clamped to [-3, +3] so extreme jumps don't fragment
        the signature space.  The contour is transposition-invariant: the same
        melodic shape at any pitch level produces the same signature.
        """
        return tuple(
            max(-3, min(3, pitches[i + 1] - pitches[i]))
            for i in range(len(pitches) - 1)
        )

    @staticmethod
    def _contour_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """Hamming distance between two contours of the same length."""
        return sum(1 for x, y in zip(a, b) if x != y)

    def suggest_lane(
        self, melody_val: float, prev_lane: int, free_lanes: list[int],
    ) -> int | None:
        if not free_lanes:
            return None

        q = self._quantize(melody_val)
        self._pitch_buf.append(q)

        # --- 1. Active replay in progress ---
        if self._replay is not None and self._replay_pos < len(self._replay):
            lane = self._replay[self._replay_pos]
            self._replay_pos += 1
            self.prev_melody = melody_val
            if lane in free_lanes:
                return lane
            return min(free_lanes, key=lambda l: abs(l - lane))

        # --- 2. Try contour match at each motif length (longest first) ---
        for mlen in self.MOTIF_LENGTHS:
            if len(self._pitch_buf) < mlen:
                continue
            cur_pitches = self._pitch_buf[-mlen:]
            cur_contour = self._to_contour(cur_pitches)

            match = self._find_match(cur_contour)
            if match is not None:
                tx = self._pick_transform(match)
                transformed = self._apply_transform(match.lanes, tx)
                match.hits += 1
                match.last_transform = tx

                self._replay = transformed
                self._replay_pos = mlen
                lane = transformed[mlen - 1]
                self.prev_melody = melody_val
                if lane in free_lanes:
                    return lane
                return min(free_lanes, key=lambda l: abs(l - lane))

        # --- 3. Early match: first 3 notes of a known motif ---
        if len(self._pitch_buf) >= 3:
            partial_contour = self._to_contour(self._pitch_buf[-3:])
            best_entry: _MotifEntry | None = None
            best_hits = 0
            for entry in self._memory:
                if len(entry.contour) < 2:
                    continue
                prefix = entry.contour[:2]
                if self._contour_distance(partial_contour, prefix) <= 0:
                    if entry.hits > best_hits:
                        best_entry = entry
                        best_hits = entry.hits
            if best_entry is not None and best_hits >= 2:
                tx = self._pick_transform(best_entry)
                transformed = self._apply_transform(best_entry.lanes, tx)
                best_entry.hits += 1
                best_entry.last_transform = tx
                self._replay = transformed
                self._replay_pos = 3
                lane = transformed[2]
                self.prev_melody = melody_val
                if lane in free_lanes:
                    return lane
                return min(free_lanes, key=lambda l: abs(l - lane))

        # --- 4. No match — follow melody direction ---
        lane = self._melody_lane(melody_val, prev_lane, free_lanes)
        self.prev_melody = melody_val
        return lane

    def _find_match(self, contour: tuple[int, ...]) -> _MotifEntry | None:
        """Find the best matching motif, allowing fuzzy match (distance ≤ 1).

        Prefers: exact match > fuzzy match, higher hit count > lower,
        longer motif > shorter.
        """
        best: _MotifEntry | None = None
        best_score = -1.0

        for entry in self._memory:
            if len(entry.contour) != len(contour):
                continue
            dist = self._contour_distance(contour, entry.contour)
            if dist > 1:
                continue
            score = (
                entry.hits * 2.0
                + len(entry.contour) * 1.5
                + (3.0 if dist == 0 else 0.0)
            )
            if score > best_score:
                best = entry
                best_score = score

        return best

    def _pick_transform(self, entry: _MotifEntry) -> str:
        """Choose a transform based on how many times this motif has replayed.

        1st replay: mostly identity or subtle shift — the player recognizes it.
        2nd replay: mirror or shift — variation on the familiar theme.
        3rd+: reverse or rev_mirror — dramatic reinterpretation.
        """
        n = entry.hits
        if n <= 1:
            return self.rng.choices(
                MOTIF_TRANSFORMS,
                weights=[4.0, 0.5, 0.5, 0.0, 2.0], k=1,
            )[0]
        elif n == 2:
            return self.rng.choices(
                MOTIF_TRANSFORMS,
                weights=[1.5, 1.0, 2.5, 0.5, 2.0], k=1,
            )[0]
        elif n == 3:
            return self.rng.choices(
                MOTIF_TRANSFORMS,
                weights=[0.5, 2.5, 1.5, 2.0, 1.0], k=1,
            )[0]
        else:
            return self.rng.choices(
                MOTIF_TRANSFORMS,
                weights=[0.3, 2.0, 1.5, 3.0, 1.5], k=1,
            )[0]

    def _melody_lane(
        self, melody_val: float, prev_lane: int, free_lanes: list[int],
    ) -> int:
        """Map melody direction to lane motion with proportional step size."""
        k = self.keys
        diff = melody_val - self.prev_melody
        if abs(diff) < 0.03:
            target = prev_lane
        else:
            step = max(1, round(abs(diff) * k * 1.5))
            if diff > 0:
                target = min(k - 1, prev_lane + step)
            else:
                target = max(0, prev_lane - step)

        target = max(0, min(k - 1, target))
        if target in free_lanes:
            return target
        return min(free_lanes, key=lambda l: abs(l - target))

    def record(self, lane: int) -> None:
        self._lane_buf.append(lane)
        for mlen in self.MOTIF_LENGTHS:
            if len(self._lane_buf) < mlen or len(self._pitch_buf) < mlen:
                continue
            pitches = self._pitch_buf[-mlen:]
            contour = self._to_contour(pitches)
            lanes = list(self._lane_buf[-mlen:])

            existing = self._find_exact(contour)
            if existing is not None:
                existing.hits += 1
            else:
                self._memory.append(_MotifEntry(
                    contour=contour, lanes=lanes, hits=1,
                ))
                if len(self._memory) > self.MAX_MEMORY:
                    self._memory.sort(key=lambda e: e.hits, reverse=True)
                    self._memory = self._memory[:self.MAX_MEMORY]

    def _find_exact(self, contour: tuple[int, ...]) -> _MotifEntry | None:
        for entry in self._memory:
            if entry.contour == contour:
                return entry
        return None

    def _apply_transform(self, lanes: list[int], tx: str) -> list[int]:
        k = self.keys
        if tx == "reverse":
            return list(reversed(lanes))
        if tx == "mirror":
            return [k - 1 - l for l in lanes]
        if tx == "rev_mirror":
            return [k - 1 - l for l in reversed(lanes)]
        if tx == "shift":
            s = self.rng.choice([1, -1])
            return [max(0, min(k - 1, l + s)) for l in lanes]
        return list(lanes)

    def on_section_change(self) -> None:
        self._pitch_buf.clear()
        self._lane_buf.clear()
        self._replay = None
        self._replay_pos = 0

    @property
    def is_replaying(self) -> bool:
        return self._replay is not None and self._replay_pos < len(self._replay)


# ---------------------------------------------------------------------------
# Legacy single-note pattern helpers (used as fallback by pick_lane)
# ---------------------------------------------------------------------------

def pattern_stairs(keys: int, step: int, ascending: bool = True) -> int:
    seq = list(range(keys)) if ascending else list(range(keys - 1, -1, -1))
    return seq[step % keys]


def pattern_zigzag(keys: int, step: int) -> int:
    cycle = list(range(keys)) + list(range(keys - 2, 0, -1))
    return cycle[step % len(cycle)] if cycle else 0


def pattern_trill(keys: int, lane_a: int, lane_b: int, step: int) -> int:
    return lane_a if step % 2 == 0 else lane_b


def pattern_mirror(keys: int, step: int) -> int:
    seq = [0, keys - 1, 1, keys - 2, keys // 2, max(0, keys // 2 - 1)]
    return seq[step % len(seq)]


def pattern_roll(keys: int, step: int, direction: int = 1) -> int:
    return (step * direction) % keys


# ---------------------------------------------------------------------------
# Section-aware density multiplier (legacy fallback)
# ---------------------------------------------------------------------------

def section_density_mult(section_label: str) -> float:
    return {
        "intro": 0.5,
        "verse": 0.75,
        "pre_chorus": 0.85,
        "chorus": 1.0,
        "bridge": 0.6,
        "solo": 1.1,
        "outro": 0.4,
    }.get(section_label, 0.7)


def find_section(sections: list[Section], t: float) -> Section | None:
    for s in sections:
        if s.start <= t < s.end:
            return s
    return sections[-1] if sections else None


# ---------------------------------------------------------------------------
# Intensity curve — the "difficulty arc" a human mapper designs
#
# Instead of a flat multiplier per section label, this builds a continuous
# curve that shapes the chart like a professional mapper would:
#   Intro → easy warmup
#   Verse → moderate, building
#   Pre-chorus → rising tension
#   Chorus → explosion
#   Bridge → rest / cool-down
#   Solo → peak difficulty
#   Final chorus → even stronger than the first
#   Outro → wind down
#
# Key features:
#   * Section occurrence tracking: 2nd chorus > 1st chorus
#   * Within-section ramp: intensity rises through each section
#   * Pre-chorus buildup: smooth ramp into the chorus
#   * Global progression: later sections are naturally harder
# ---------------------------------------------------------------------------

SECTION_BASE_INTENSITY: dict[str, float] = {
    "intro":      0.25,
    "verse":      0.50,
    "pre_chorus": 0.65,
    "chorus":     0.90,
    "bridge":     0.35,
    "solo":       1.00,
    "outro":      0.30,
    "unknown":    0.50,
}

OCCURRENCE_BONUS = 0.07
GLOBAL_PROGRESSION = 0.08


@dataclass
class IntensityPoint:
    time: float
    intensity: float


def build_intensity_curve(sections: list[Section], duration: float) -> list[IntensityPoint]:
    """Build an intensity curve that follows a professional difficulty arc."""
    if not sections:
        return [IntensityPoint(0.0, 0.5), IntensityPoint(duration, 0.5)]

    occurrence: dict[str, int] = {}
    annotated: list[tuple[Section, int]] = []
    for s in sections:
        n = occurrence.get(s.label, 0) + 1
        occurrence[s.label] = n
        annotated.append((s, n))

    points: list[IntensityPoint] = []
    for i, (sec, occ) in enumerate(annotated):
        base = SECTION_BASE_INTENSITY.get(sec.label, 0.5)
        occ_bonus = min(0.15, (occ - 1) * OCCURRENCE_BONUS)
        progress = sec.start / max(duration, 1.0)
        prog_bonus = progress * GLOBAL_PROGRESSION

        sec_dur = max(0.1, sec.end - sec.start)
        # Allow values above 1.0 so repeated choruses actually get denser.
        start_val = base + occ_bonus + prog_bonus

        next_is_chorus = (i + 1 < len(annotated)
                          and annotated[i + 1][0].label in ("chorus", "solo"))

        if sec.label == "pre_chorus" or (
            next_is_chorus and sec.label in ("verse", "bridge")
        ):
            ramp_start = sec.start + sec_dur * 0.6
            next_sec = annotated[i + 1][0] if i + 1 < len(annotated) else None
            next_base = SECTION_BASE_INTENSITY.get(
                next_sec.label if next_sec else "chorus", 0.9)
            ramp_target = next_base * 0.85 + occ_bonus
            points.append(IntensityPoint(round(sec.start, 4), start_val))
            points.append(IntensityPoint(round(ramp_start, 4), start_val + 0.05))
            points.append(IntensityPoint(round(sec.end, 4), ramp_target))
        elif sec.label == "chorus":
            points.append(IntensityPoint(round(sec.start, 4), start_val))
            mid_t = sec.start + sec_dur * 0.5
            points.append(IntensityPoint(round(mid_t, 4), start_val + 0.05))
            points.append(IntensityPoint(round(sec.end, 4), start_val - 0.03))
        elif sec.label == "solo":
            points.append(IntensityPoint(round(sec.start, 4), start_val + 0.05))
            points.append(IntensityPoint(round(sec.end, 4), start_val))
        elif sec.label == "bridge":
            points.append(IntensityPoint(round(sec.start, 4), start_val))
            rest_t = sec.start + sec_dur * 0.3
            points.append(IntensityPoint(round(rest_t, 4), max(0.15, start_val - 0.10)))
            points.append(IntensityPoint(round(sec.end, 4), start_val))
        elif sec.label == "outro":
            points.append(IntensityPoint(round(sec.start, 4), start_val))
            points.append(IntensityPoint(round(sec.end, 4), max(0.10, start_val - 0.15)))
        else:
            within_ramp = 0.05 if sec.label != "intro" else 0.02
            points.append(IntensityPoint(round(sec.start, 4), start_val))
            points.append(IntensityPoint(round(sec.end, 4), start_val + within_ramp))

    if not points:
        return [IntensityPoint(0.0, 0.5)]
    return points


def intensity_at(curve: list[IntensityPoint], t: float) -> float:
    """Linearly interpolate the intensity curve at time *t*."""
    if not curve:
        return 0.5
    if t <= curve[0].time:
        return curve[0].intensity
    if t >= curve[-1].time:
        return curve[-1].intensity
    for i in range(len(curve) - 1):
        if curve[i].time <= t < curve[i + 1].time:
            span = curve[i + 1].time - curve[i].time
            if span < 0.001:
                return curve[i].intensity
            frac = (t - curve[i].time) / span
            return curve[i].intensity + frac * (curve[i + 1].intensity - curve[i].intensity)
    return curve[-1].intensity


# ---------------------------------------------------------------------------
# Density Planner — closed-loop density control
#
# The intensity curve says "the chorus should be 90% dense" but the old code
# had no feedback: if the audio was quiet the threshold was never met, or if
# everything was loud the section was saturated.
#
# The DensityPlanner adds a feedback loop:
#   1. Compute target NPS from intensity × difficulty peak NPS
#   2. Track actual NPS in a rolling window
#   3. Output a correction factor: >1 = under-dense (accept more), <1 = over-
#      dense (reject more).  The factor modulates the placement threshold.
#   4. Post-pass: trim the weakest notes in any window still >130% of target.
#
# Result: ★★ ★★★★ ★★★★★★ ★★★★★★★★ ★★★★★ ★★★ — a smooth arc.
# ---------------------------------------------------------------------------

DIFFICULTY_PEAK_NPS: dict[str, float] = {
    "easy":   1.5,
    "normal": 3.0,
    "hard":   5.0,
    "expert": 8.0,
    "master": 12.0,
}

# Minimum onset strength required before the density controller is allowed to
# *boost* placement (lower the threshold).  Below this floor the closed-loop
# correction is clamped to ≤1.0, so the difficulty-arc density target can never
# fabricate "phantom notes" where the audio has no real transient — protecting
# rhythm fit in quiet / sustained passages.  See generate_chart().
MIN_ONSET_FOR_BOOST = 0.15


class DensityPlanner:
    """Real-time density feedback controller.

    Maintains a rolling note count in a sliding window and compares it to
    the target NPS derived from the intensity curve.  Returns a correction
    factor that the placement threshold multiplies by.
    """

    WINDOW = 2.0        # seconds — rolling measurement window
    SMOOTHING = 0.7     # exponential smoothing for the correction factor
    MAX_BOOST = 1.6     # max factor when under-dense
    MAX_SUPPRESS = 0.5  # min factor when over-dense

    def __init__(self, peak_nps: float, intensity_curve: list[IntensityPoint]):
        self._peak_nps = peak_nps
        self._curve = intensity_curve
        self._note_times: list[float] = []
        self._prev_factor: float = 1.0

    def target_nps(self, t: float) -> float:
        """Target notes per second at time *t*."""
        intensity = intensity_at(self._curve, t)
        return max(0.3, self._peak_nps * intensity)

    def record(self, t: float) -> None:
        """Record that a note was placed at time *t*."""
        self._note_times.append(t)

    def actual_nps(self, t: float) -> float:
        """Actual NPS in the window ending at *t*."""
        cutoff = t - self.WINDOW
        count = sum(1 for nt in self._note_times if cutoff <= nt <= t)
        return count / self.WINDOW

    def correction(self, t: float) -> float:
        """Correction factor for the placement threshold at time *t*.

        >1.0 → under-dense, lower the threshold to accept more notes.
        <1.0 → over-dense, raise the threshold to reject notes.
        """
        target = self.target_nps(t)
        actual = self.actual_nps(t)

        if target < 0.1:
            return 1.0

        ratio = target / max(actual, 0.1)
        raw = max(self.MAX_SUPPRESS, min(self.MAX_BOOST, ratio))

        smoothed = self.SMOOTHING * self._prev_factor + (1.0 - self.SMOOTHING) * raw
        self._prev_factor = smoothed
        return smoothed


def _density_trim(
    notes: list[Note],
    planner: DensityPlanner,
    duration: float,
) -> list[Note]:
    """Post-pass: remove the weakest notes in windows that exceed 130% of target.

    Only removes non-strong-beat taps (never holds, never downbeat anchors).
    Preserves musical intent while enforcing the density curve.
    """
    if not notes:
        return notes

    window = 2.0
    step = 1.0
    t = 0.0
    remove_set: set[int] = set()

    note_indices = list(range(len(notes)))

    while t < duration:
        target = planner.target_nps(t + window / 2)
        max_count = max(1, int(target * window * 1.3))

        in_window = [
            i for i in note_indices
            if i not in remove_set and t <= notes[i].time < t + window
        ]

        if len(in_window) > max_count:
            trimmable = [
                i for i in in_window
                if notes[i].kind == "tap"
                and not _is_strong_beat(notes[i].beat)
            ]
            trimmable.sort(key=lambda i: notes[i].weight)
            excess = len(in_window) - max_count
            for i in trimmable[:excess]:
                remove_set.add(i)

        t += step

    if not remove_set:
        return notes
    return [n for i, n in enumerate(notes) if i not in remove_set]


def _is_strong_beat(beat: float) -> bool:
    """True if *beat* falls on a downbeat (whole number)."""
    return abs(beat - round(beat)) < 0.06


# ---------------------------------------------------------------------------
# Hand-aware lane picker
# ---------------------------------------------------------------------------

def pick_lane(
    step: int,
    keys: int,
    prev_lanes: list[int],
    section_label: str,
    onset: float,
    rng: random.Random,
    difficulty: str,
    melody_val: float = 0.5,
    style: dict | None = None,
    instrument: str = "other",
    hand: HandState | None = None,
) -> int:
    """Choose a lane considering instrument type, hand alternation, and flow.

    Professional mappers assign lanes based on *what* instrument is playing
    (kick → outer, snare → inner) and *which hand* should move next.  This
    creates the natural L-R alternation and intentional movement that
    distinguishes a good chart from a random scatter.
    """
    if keys <= 1:
        return 0
    if style is None:
        style = STYLES["auto"]
    if hand is None:
        hand = HandState()

    mid = keys / 2
    is_chorus = section_label == "chorus"
    is_intense = onset > 0.75
    pref_hand = hand.preferred_hand()
    hand_lanes_l = list(range(0, int(mid)))
    hand_lanes_r = list(range(int(mid), keys))
    pref_lanes = hand_lanes_l if pref_hand == 0 else hand_lanes_r

    # --- Signal 1: instrument anchoring ---
    anchors = instrument_anchors(instrument, keys)
    anchor_in_pref = [a for a in anchors if a in pref_lanes]

    # --- Signal 2: melody contour ---
    melody_lane = pitch_to_lane(melody_val, keys)

    # --- Signal 3: movement-based pattern ---
    if is_intense and rng.random() < style["trill_bias"]:
        a = rng.choice(hand_lanes_l) if hand_lanes_l else 0
        b = rng.choice(hand_lanes_r) if hand_lanes_r else keys - 1
        pattern_lane = pattern_trill(keys, a, b, step)
    elif is_chorus and is_intense and difficulty in ("expert", "master"):
        pattern_lane = pattern_zigzag(keys, step)
    elif section_label == "bridge":
        a = max(0, int(mid) - 1)
        b = int(mid)
        pattern_lane = pattern_trill(keys, a, b, step)
    elif hand.dir_steps >= keys:
        pattern_lane = pattern_stairs(keys, step, ascending=(hand.direction < 0))
    else:
        pattern_lane = pattern_stairs(keys, step, ascending=(hand.direction > 0))

    # --- Blend the three signals by style ---
    lane: int
    r = rng.random()
    drum_weight = style["drum_anchor"]
    pitch_weight = style["pitch_follow"]

    if instrument in ("kick", "snare") and r < drum_weight:
        if anchor_in_pref:
            lane = rng.choice(anchor_in_pref)
        else:
            lane = rng.choice(anchors)
    elif instrument == "vocal" and r < pitch_weight:
        lane = melody_lane
    elif r < pitch_weight:
        lane = melody_lane
    else:
        lane = pattern_lane

    # --- Hand alternation bias: nudge toward the preferred hand's side ---
    if prev_lanes and rng.random() < 0.6:
        if hand.which_hand(lane, keys) != pref_hand and pref_lanes:
            closest = min(pref_lanes, key=lambda l: abs(l - lane))
            lane = closest

    # --- Anti-jack: avoid same lane twice, prefer smooth movement ---
    if prev_lanes:
        last = prev_lanes[-1]
        if lane == last and rng.random() < 0.82:
            candidates = [l for l in range(keys) if l != lane]
            if candidates:
                scored = sorted(candidates, key=lambda c: (
                    abs(hand.which_hand(c, keys) - pref_hand) * 2.0
                    + abs(c - melody_lane) * 0.3
                    + (0.0 if abs(c - last) <= 2 else 0.5)
                ))
                lane = scored[0] if rng.random() < 0.75 else rng.choice(scored[:max(1, len(scored) // 2)])

    # Triple-jack prevention
    if len(prev_lanes) >= 2 and prev_lanes[-1] == prev_lanes[-2] == lane:
        alts = [l for l in range(keys) if l != lane]
        if alts:
            lane = min(alts, key=lambda l: abs(hand.which_hand(l, keys) - pref_hand))

    return int(max(0, min(keys - 1, lane)))


# ---------------------------------------------------------------------------
# Humanizer (Step 20)
# ---------------------------------------------------------------------------

def humanize_notes(notes: list[Note], rng: random.Random, amount: float = 0.008) -> list[Note]:
    for n in notes:
        offset = rng.gauss(0, amount)
        n.time = round(max(0.0, n.time + offset), 4)
        n.weight = round(n.weight * rng.uniform(0.93, 1.07), 4)
    return notes


# ---------------------------------------------------------------------------
# Main chart generator (Steps 11-15, 19-20)
# ---------------------------------------------------------------------------

def value_at(times, values: list[float], t: float) -> float:
    # ``times`` may be a Python list or an ndarray; ``len`` works for both and
    # avoids ndarray's ambiguous truth value.
    if len(times) == 0 or not values:
        return 0.0
    idx = int(np.searchsorted(times, t, side="left"))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def peak_near(times: list[float], onset: list[float], t: float, w: float) -> float:
    left = int(np.searchsorted(times, t - w, side="left"))
    right = int(np.searchsorted(times, t + w, side="right"))
    if right <= left:
        return 0.0
    return float(max(onset[left:right]))


def upcoming_chorus_distance(sections: list[Section], t: float, horizon: float) -> float:
    """Return 0..1 closeness to the next chorus within ``horizon`` seconds.

    Used to ramp density just before a drop, the way mappers build tension
    into a chorus. 0 = no chorus ahead, 1 = chorus starts right now.
    """
    for s in sections:
        if s.label == "chorus" and t < s.start <= t + horizon:
            return 1.0 - (s.start - t) / horizon
    return 0.0


# ---------------------------------------------------------------------------
# Chart explanation (human-readable "why this pattern here" commentary)
# ---------------------------------------------------------------------------

_CATEGORY_KO = {
    "stair": "계단", "trill": "트릴", "jack": "잭", "burst": "폭타",
    "stream": "스트림", "hold": "롱노트", "chord": "동시치기",
    "swing": "스윙", "misc": "일반",
}
_FOCUS_KO = {
    "vocal": "보컬", "drums": "드럼", "guitar": "기타", "bass": "베이스",
    "keys": "건반", "mixed": "혼합",
}
_SECTION_KO = {
    "intro": "인트로", "verse": "벌스", "pre_chorus": "프리코러스",
    "chorus": "코러스", "bridge": "브릿지", "solo": "솔로",
    "outro": "아웃트로", "unknown": "구간",
}
_TRANSFORM_KO = {
    "identity": "원형", "mirror": "좌우 대칭", "reverse": "역순",
    "rev_mirror": "역순+대칭", "shift": "이동", "reuse": "재사용",
}


def _mmss(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m}:{s:02d}"


def _musical_reason(ctx: dict) -> str:
    """Describe, in Korean, the dominant musical cue at this moment."""
    cues: list[tuple[float, str]] = [
        (ctx.get("melody_dir", 0.0), "상승하는 멜로디"),
        (-ctx.get("melody_dir", 0.0), "하강하는 멜로디"),
        (ctx.get("drum_fill", 0.0), "드럼 필"),
        (ctx.get("vocal_presence", 0.0), "이어지는 보컬 라인"),
        (ctx.get("tension", 0.0) - 0.2, "고조되는 긴장감"),
        (ctx.get("onset", 0.0) - 0.3, "강한 타격음"),
        (ctx.get("chord_change", 0.0) - 0.3, "코드 전환"),
    ]
    best_val, best_txt = max(cues, key=lambda c: c[0])
    return best_txt if best_val > 0.2 else "전반적인 리듬 흐름"


def _explain_section(t: float, label: str, focus, intensity: float,
                     replay: dict | None) -> dict:
    sec_ko = _SECTION_KO.get(label, label)
    foc_ko = _FOCUS_KO.get(focus.instrument, focus.instrument)
    text = (f"{_mmss(t)} · {sec_ko} 진입 (강도 {intensity:.2f}) — "
            f"{foc_ko} 주도")
    if replay and replay.get("varied", 0) > 0:
        pct = replay.get("reuse_pct", 70.0)
        text += f", 이전 동일 구간 레인 {pct:.0f}% 재사용 + 블록 변형"
    return {"time": round(t, 2), "time_str": _mmss(t), "kind": "section",
            "section": label, "focus": focus.instrument,
            "intensity": round(intensity, 3), "text": text}


def _explain_focus(t: float, focus) -> dict:
    foc_ko = _FOCUS_KO.get(focus.instrument, focus.instrument)
    hint = {
        "vocal": "보컬 선율을 따라 롱노트와 계단 위주",
        "drums": "킥·스네어에 맞춰 트릴·폭타 위주",
        "guitar": "리프를 따라 스트림·트릴 위주",
        "bass": "저음 그루브에 맞춘 안정적 배치",
        "keys": "건반 화음에 맞춘 동시치기",
        "mixed": "여러 악기를 균형 있게 반영",
    }.get(focus.instrument, "")
    text = f"{_mmss(t)} · 주도 악기 → {foc_ko} (신뢰도 {focus.confidence:.2f}): {hint}"
    return {"time": round(t, 2), "time_str": _mmss(t), "kind": "focus",
            "focus": focus.instrument, "text": text}


def _explain_pattern(t: float, sel: dict, label: str, ctx: dict) -> dict:
    name = sel.get("name", "?")
    tx = sel.get("transform", "identity")
    cat = sel.get("category", "misc")
    cat_ko = _CATEGORY_KO.get(cat, cat)
    reason = _musical_reason(ctx)
    parts = [f"{_mmss(t)} · '{name}' ({cat_ko}) 패턴 선택 — {reason}에 대응"]
    if tx not in ("identity", None):
        parts.append(f"{_TRANSFORM_KO.get(tx, tx)} 변형")
    if sel.get("pool_size"):
        parts.append(f"후보 {sel['pool_size']}개 중 상위 {len(sel.get('shortlist', []))}개에서 선정"
                     f" (점수 {sel.get('score', 0):.2f})")
    text = " · ".join(parts)
    return {"time": round(t, 2), "time_str": _mmss(t), "kind": "pattern",
            "pattern": name, "category": cat, "transform": tx,
            "section": label, "text": text}


def _summarize_features(analysis: AnalysisResult) -> list[dict]:
    """Average the extended feature vectors within each section.

    Returns a compact list (one entry per section) instead of the full
    per-frame arrays, so the chart JSON stays small while still exposing the
    ⑩ features for the frontend to visualize or the user to inspect.
    """
    feats = analysis.features
    ft = analysis.frame_times
    if not ft or not analysis.sections:
        return []

    ft_arr = np.asarray(ft)
    fields = {
        "spectral_flux": feats.spectral_flux,
        "instrument_change": feats.instrument_change,
        "harmonic_ratio": feats.harmonic_ratio,
        "chord_change": feats.chord_change,
        "vocal_presence": feats.vocal_presence,
        "drum_fill": feats.drum_fill,
        "tension": feats.tension,
    }

    summary: list[dict] = []
    for s in analysis.sections:
        left = int(np.searchsorted(ft_arr, s.start, side="left"))
        right = int(np.searchsorted(ft_arr, s.end, side="right"))
        entry = {"label": s.label, "start": s.start, "end": s.end}
        for name, arr in fields.items():
            if arr and right > left:
                entry[name] = round(float(np.mean(arr[left:right])), 3)
            else:
                entry[name] = 0.0
        summary.append(entry)
    return summary


def generate_chart(
    analysis: AnalysisResult,
    keys: int = 4,
    difficulty: str = "hard",
    seed: int = 42,
    humanize: bool = True,
    style: str = "auto",
) -> dict:
    rng = random.Random(seed)
    cfg = DIFFICULTIES[difficulty]
    base_sty = STYLES.get(style, STYLES["auto"])
    has_melody = bool(analysis.melody)
    # Pre-convert the frame-time axis to an ndarray ONCE.  value_at/peak_near
    # call np.searchsorted on this axis ~11× per grid step; passing a Python
    # list forces numpy to re-copy the whole 9k-element list to an array on
    # every call (O(steps × frames) — the dominant cost for long songs).  A
    # single asarray makes each lookup an O(log n) search on a real ndarray.
    frame_times_arr = np.asarray(analysis.frame_times, dtype=np.float64)
    tempo_map = analysis.tempo_map or [TempoPoint(time=0.0, bpm=analysis.bpm)]
    focus_segs = analysis.focus_segments or []

    # Build the subdivision grid following the tempo map so every beat
    # respects the local BPM — live recordings, rubato, and BPM changes
    # all produce correctly-spaced grid lines.
    subdiv = float(cfg["subdiv"])
    grid_times: list[float] = []
    t = analysis.beat_offset
    while t < analysis.duration:
        if t >= 0:
            grid_times.append(round(t, 6))
        local_bpm = bpm_at(tempo_map, t)
        t += 60.0 / local_bpm / subdiv
    total_steps = len(grid_times)

    # Build the intensity curve — the difficulty arc across the whole song.
    i_curve = build_intensity_curve(analysis.sections, analysis.duration)
    density_planner = DensityPlanner(
        peak_nps=DIFFICULTY_PEAK_NPS.get(difficulty, 5.0),
        intensity_curve=i_curve,
    )

    notes: list[Note] = []
    recent_lanes: list[int] = []
    recent_times: list[float] = []
    lane_busy_until: dict[int, float] = {}
    phrase_cache: dict[int, float] = {}
    hand = HandState()
    pstate = PatternState(rng, keys)
    motif = MotifGenerator(rng, keys)
    sreplay = SectionReplay(rng, keys)
    prev_sec_label = ""
    prev_phrase_idx = -1
    prev_melody_val = 0.5

    # Chart-explanation log: high-level "why this here" commentary.
    explanations: list[dict] = []
    expl_prev_section: str | None = None
    expl_prev_focus: str | None = None
    expl_prev_sel_id: int | None = None
    expl_prev_cat: str | None = None

    for step in range(total_steps):
        t = grid_times[step]
        if t < 0.15 or t > analysis.duration - 0.05:
            continue

        local_bpm = bpm_at(tempo_map, t)
        beat = 60.0 / local_bpm
        grid = beat / subdiv
        release_gap = max(0.05, grid * 0.5)
        window = min(grid * 0.58, 0.08)
        phrase_len = 16 * beat

        beat_pos = (t - analysis.beat_offset) / beat
        beat_frac = beat_pos - math.floor(beat_pos)
        strong = min(abs(beat_frac), abs(beat_frac - 1.0)) < 0.045
        half = abs(beat_frac - 0.5) < 0.05
        offbeat = abs(beat_frac - 0.25) < 0.05 or abs(beat_frac - 0.75) < 0.05

        section = find_section(analysis.sections, t)
        sec_label = section.label if section else "verse"

        # Track section changes for pattern memory (④ repeat patterns)
        if sec_label != prev_sec_label:
            pstate.enter_section(sec_label)
            sreplay.enter_section(sec_label)
            motif.on_section_change()
            prev_sec_label = sec_label

        # --- Focus instrument → dynamic style modulation ---
        cur_focus = focus_at(focus_segs, t) if focus_segs else FocusSegment("mixed", 0.0, 0.0, 0.3)
        sty = focus_adjusted_style(base_sty, cur_focus)

        # --- Intensity curve replaces flat section_density_mult ---
        intensity = intensity_at(i_curve, t)

        # --- Chart explanation: section & focus transitions ---
        if sec_label != expl_prev_section:
            explanations.append(_explain_section(
                t, sec_label, cur_focus, intensity, sreplay.last_variation))
            expl_prev_section = sec_label
        if cur_focus.instrument != expl_prev_focus:
            if not (cur_focus.instrument == "mixed" and expl_prev_focus is None):
                explanations.append(_explain_focus(t, cur_focus))
            expl_prev_focus = cur_focus.instrument

        phrase_idx = int(t // phrase_len)
        if phrase_idx not in phrase_cache:
            ps = phrase_idx * phrase_len
            left = int(np.searchsorted(frame_times_arr, ps, side="left"))
            right = int(np.searchsorted(frame_times_arr, ps + phrase_len, side="right"))
            if right > left:
                local = np.array(analysis.rms[left:right]) * 0.55 + np.array(analysis.onset_strength[left:right]) * 0.45
                phrase_cache[phrase_idx] = float(np.mean(local))
            else:
                phrase_cache[phrase_idx] = 0.0

        if phrase_idx != prev_phrase_idx:
            prev_phrase_idx = phrase_idx

        onset = peak_near(frame_times_arr, analysis.onset_strength, t, window)
        energy = value_at(frame_times_arr, analysis.rms, t)
        high_band = value_at(frame_times_arr, analysis.bands.high, t) if analysis.bands.high else 0.0
        mid_band = value_at(frame_times_arr, analysis.bands.mid, t) if analysis.bands.mid else 0.0
        low_band = value_at(frame_times_arr, analysis.bands.low, t) if analysis.bands.low else 0.0
        melody_val = value_at(frame_times_arr, analysis.melody, t) if has_melody else 0.5

        # --- Extended feature vector sampling (⑩) ---
        feats = analysis.features
        ft = frame_times_arr
        vocal_presence = value_at(ft, feats.vocal_presence, t) if feats.vocal_presence else 0.0
        drum_fill = value_at(ft, feats.drum_fill, t) if feats.drum_fill else 0.0
        tension = value_at(ft, feats.tension, t) if feats.tension else 0.0
        harmonic_ratio = value_at(ft, feats.harmonic_ratio, t) if feats.harmonic_ratio else 0.5
        chord_change = value_at(ft, feats.chord_change, t) if feats.chord_change else 0.0
        instrument_change = value_at(ft, feats.instrument_change, t) if feats.instrument_change else 0.0

        # Melody slope for the ⑧ scorer (-1 falling .. +1 rising).
        melody_dir = max(-1.0, min(1.0, (melody_val - prev_melody_val) * 4.0))
        prev_melody_val = melody_val

        instrument = classify_hit(low_band, mid_band, high_band)
        drum_hit = low_band * sty["drum_anchor"]
        vocal_hit = mid_band * sty["vocal_follow"]
        fill_hit = high_band

        contour = (
            0.42 * onset + 0.18 * energy + 0.12 * phrase_cache.get(phrase_idx, 0)
            + 0.10 * vocal_hit + 0.10 * fill_hit + 0.08 * drum_hit
        )

        build = upcoming_chorus_distance(analysis.sections, t, 8.0) * sty["buildup"]
        density_correction = density_planner.correction(t)
        # Phantom-note safeguard: the density controller may *suppress* freely
        # (correction < 1 raises the bar) but may only *boost* placement where a
        # real transient exists.  Below a minimum onset floor we clamp the boost
        # to ≤1.0, so the difficulty-arc target can never lower the threshold far
        # enough to fabricate notes in a silent/sustained gap.  This keeps rhythm
        # fit (notes on actual hits) from being sacrificed to density targeting.
        if onset < MIN_ONSET_FOR_BOOST:
            density_correction = min(density_correction, 1.0)
        threshold = float(cfg["accent"]) / (intensity * density_correction + build * 0.6)

        place = contour >= threshold
        if strong and (energy > 0.12 or drum_hit > 0.3) and rng.random() < cfg["base"] * intensity:
            place = True
        if half and onset > threshold * 0.92 and rng.random() < cfg["base"] * 0.85:
            place = True
        if offbeat and difficulty in ("hard", "expert", "master") and onset > threshold * 0.8 and rng.random() < (0.46 + build * 0.3) * intensity:
            place = True
        if vocal_hit > 0.5 and onset > threshold * 0.7 and rng.random() < sty["vocal_follow"]:
            place = True

        if energy < 0.08 and not strong:
            place = False
        if sec_label == "intro" and not strong and rng.random() > 0.4:
            place = False
        if sec_label == "outro" and not strong and rng.random() > 0.3:
            place = False

        free_lanes = [L for L in range(keys) if lane_busy_until.get(L, 0.0) <= t]
        busy_count = keys - len(free_lanes)

        density_window = 0.15
        recent_count = sum(1 for rt in recent_times if t - rt < density_window)
        if recent_count >= max(1, cfg["max_density"] - busy_count):
            place = False

        if len(free_lanes) == 0 or (len(free_lanes) == 1 and busy_count > 0 and not strong):
            place = False

        if not place:
            continue

        # --- Lane decision priority ---
        # 1. Section replay (repeated chorus/verse reuses previous lanes)
        # 2. Motif generator (melody contour → lane direction, PRIMARY)
        # 3. Pattern library (multi-note patterns for rhythm feel)
        # 4. pick_lane fallback (instrument anchoring, hand alternation)

        section_replay_lane = None
        if sreplay.is_replaying:
            section_replay_lane = sreplay.next_lane(free_lanes)

        if section_replay_lane is not None:
            lane = section_replay_lane
        else:
            # Pattern library: try to start / consume a phrase pattern
            score_ctx = {
                "onset": onset, "energy": energy, "high": high_band,
                "mid": mid_band, "low": low_band, "intensity": intensity,
                "melody_dir": melody_dir,
                "vocal_presence": vocal_presence, "drum_fill": drum_fill,
                "tension": tension, "harmonic_ratio": harmonic_ratio,
                "chord_change": chord_change, "instrument_change": instrument_change,
            }
            pstate.try_start(sec_label, onset, style, difficulty, intensity, score_ctx)

            # Chart explanation: log a newly-started pattern, but only when its
            # category changes, to keep the commentary high-level and readable.
            sel = pstate.last_selection
            if sel is not None and id(sel) != expl_prev_sel_id:
                expl_prev_sel_id = id(sel)
                if sel.get("category") != expl_prev_cat:
                    explanations.append(
                        _explain_pattern(t, sel, sec_label, score_ctx))
                    expl_prev_cat = sel.get("category")

            pattern_lanes = pstate.next_step(keys, free_lanes)

            if pattern_lanes is not None and len(pattern_lanes) >= 1:
                lane = pattern_lanes[0]
            elif has_melody:
                # Motif generator: melody contour → lane motion
                prev_lane = recent_lanes[-1] if recent_lanes else keys // 2
                motif_lane = motif.suggest_lane(melody_val, prev_lane, free_lanes)
                if motif_lane is not None:
                    lane = motif_lane
                else:
                    lane = pick_lane(
                        step, keys, recent_lanes, sec_label, onset, rng,
                        difficulty, melody_val, sty, instrument, hand,
                    )
                    if lane not in free_lanes:
                        if not free_lanes:
                            continue
                        lane = min(free_lanes, key=lambda L: abs(L - lane))
            else:
                lane = pick_lane(
                    step, keys, recent_lanes, sec_label, onset, rng,
                    difficulty, melody_val, sty, instrument, hand,
                )
                if lane not in free_lanes:
                    if not free_lanes:
                        continue
                    lane = min(free_lanes, key=lambda L: abs(L - lane))

        # Record for motif memory and section replay.
        motif.record(lane)
        sreplay.record(lane)

        kind = "tap"
        duration = 0.0
        # Sustained-note detection drives holds.  A note is "holdable" when the
        # audio under it is actually sustained — harmonic/tonal, carrying a
        # vocal or lead, and NOT a percussive transient.  The old logic keyed
        # off (strong beat AND low onset), but strong beats coincide with drum
        # hits (high onset), so holds almost never fired — large-scale
        # validation showed ~0% holds even for piano/vocal.  The extended
        # feature vectors (⑩) give a proper sustain signal.
        # A held note is *struck then sustained*, so we don't gate on a low
        # onset (the strike itself is an onset).  Instead we gate on the note
        # being tonal (harmonic, low drum-fill) — that's what's actually
        # holdable — via the extended feature vectors (⑩).
        sustain_signal = max(vocal_presence, harmonic_ratio * (1.0 - drum_fill))
        sustained_vocal = vocal_presence > 0.4
        holdable = (
            sustain_signal > 0.45
            and drum_fill < 0.4           # not in the middle of a drum fill
            and (energy > 0.3 or vocal_presence > 0.3)
            and (strong or half)          # anchor on beat / half-beat, not 16ths
        )
        hold_prob = cfg["long"] * sty["hold_mult"] * (0.55 + 0.9 * sustain_signal)
        if holdable and rng.random() < hold_prob:
            kind = "hold"
            long_sustain = sustained_vocal or harmonic_ratio > 0.7
            length = (rng.choice([1.0, 2.0, 4.0]) if long_sustain
                      else rng.choice([1.0, 1.5, 2.0, 3.0]))
            duration = round(beat * length, 4)
            lane_busy_until[lane] = t + duration + release_gap

        notes.append(Note(
            time=round(t, 4), lane=lane, kind=kind, duration=duration,
            beat=round(beat_pos, 4), weight=round(contour, 4),
        ))
        density_planner.record(t)

        # Pattern chords: if the pattern step was a chord, place extra notes.
        if pattern_lanes is not None and len(pattern_lanes) > 1:
            for extra_lane in pattern_lanes[1:]:
                if extra_lane != lane and lane_busy_until.get(extra_lane, 0.0) <= t:
                    notes.append(Note(
                        time=round(t, 4), lane=extra_lane, kind="tap",
                        beat=round(beat_pos, 4), weight=round(contour, 4),
                    ))
                    hand.update(extra_lane, keys)

        if recent_lanes:
            hand.update_direction(lane, recent_lanes[-1])
        hand.update(lane, keys)
        recent_lanes.append(lane)
        recent_times.append(t)
        if len(recent_lanes) > 20:
            recent_lanes = recent_lanes[-20:]
        if len(recent_times) > 50:
            recent_times = recent_times[-50:]

        # Non-pattern chords: instrument-aware hand shapes.
        if pattern_lanes is None or len(pattern_lanes) <= 1:
            chord_chance = cfg["chord"] * sty["chord_mult"] * intensity
            if (
                kind == "tap" and keys >= 4 and strong and onset > 0.78
                and (energy > 0.45 or drum_hit > 0.4)
                and rng.random() < chord_chance
                and sec_label in ("chorus", "verse")
            ):
                second = chord_partner(lane, instrument, keys)
                if second != lane and lane_busy_until.get(second, 0.0) <= t and second in free_lanes:
                    notes.append(Note(
                        time=round(t, 4), lane=second, kind="tap",
                        beat=round(beat_pos, 4), weight=round(contour, 4),
                    ))
                    hand.update(second, keys)

    sreplay.finalize()
    notes.sort(key=lambda n: (n.time, n.lane))

    notes = _density_trim(notes, density_planner, analysis.duration)

    if humanize and difficulty in ("hard", "expert", "master"):
        notes = humanize_notes(notes, rng, amount=0.006)

    taps = sum(1 for n in notes if n.kind == "tap")
    holds = sum(1 for n in notes if n.kind == "hold")

    tempo_map_out = [
        {"time": tp.time, "bpm": tp.bpm} for tp in tempo_map
    ]

    # Build density graph: target intensity curve + actual note density
    density_curve_out = [
        {"time": round(p.time, 3), "intensity": round(p.intensity, 3)}
        for p in i_curve
    ]

    # Compact per-section feature summary (⑩) — averages of the extended
    # feature vectors within each section, for inspection / visualization.
    feature_summary = _summarize_features(analysis)

    # Actual + target note density (NPS) in 2-second windows
    actual_density: list[dict] = []
    if notes and analysis.duration > 0:
        window = 2.0
        step_size = 1.0
        t_cursor = 0.0
        note_times = sorted(n.time for n in notes)
        while t_cursor < analysis.duration:
            mid_t = t_cursor + window / 2
            count = sum(1 for nt in note_times if t_cursor <= nt < t_cursor + window)
            nps = count / window
            actual_density.append({
                "time": round(mid_t, 3),
                "nps": round(nps, 2),
                "target_nps": round(density_planner.target_nps(mid_t), 2),
            })
            t_cursor += step_size

    return {
        "metadata": {
            "bpm": analysis.bpm,
            "beat_offset": analysis.beat_offset,
            "duration": analysis.duration,
            "keys": keys,
            "difficulty": difficulty,
            "style": style,
            "note_count": len(notes),
            "tap_count": taps,
            "hold_count": holds,
            "tempo_map": tempo_map_out,
            "sections": [
                {"label": s.label, "start": s.start, "end": s.end, "energy": s.energy}
                for s in analysis.sections
            ],
            "lane_colors": LANE_COLORS.get(keys, ["#ffffff"] * keys),
            "focus_segments": [
                {"instrument": fs.instrument, "start": fs.start,
                 "end": fs.end, "confidence": fs.confidence}
                for fs in focus_segs
            ],
            "density_curve": density_curve_out,
            "actual_density": actual_density,
            "feature_summary": feature_summary,
            "explanations": explanations,
        },
        "notes": [asdict(n) for n in notes],
    }
