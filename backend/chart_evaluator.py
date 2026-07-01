"""Chart quality evaluator and auto-improver.

Scores a generated chart across six dimensions that professional chart
makers consider.  When any dimension scores below threshold, the chart
is regenerated with a different seed and the best result is kept.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    rhythm_fit: float = 0.0
    hand_movement: float = 0.0
    pattern_diversity: float = 0.0
    repetition: float = 0.0
    long_note: float = 0.0
    difficulty_fit: float = 0.0
    overall: float = 0.0
    details: dict = field(default_factory=dict)


# Expected note-per-second ranges per difficulty.
DENSITY_RANGES: dict[str, tuple[float, float]] = {
    "easy":   (0.5, 2.5),
    "normal": (1.5, 5.0),
    "hard":   (3.0, 9.0),
    "expert": (5.0, 14.0),
    "master": (8.0, 22.0),
}

# Target long-note ratios per difficulty.
HOLD_RATIO_RANGES: dict[str, tuple[float, float]] = {
    "easy":   (0.10, 0.35),
    "normal": (0.08, 0.28),
    "hard":   (0.05, 0.22),
    "expert": (0.03, 0.18),
    "master": (0.02, 0.15),
}


# ---------------------------------------------------------------------------
# Individual scorers (each returns 0–100)
# ---------------------------------------------------------------------------

def _score_rhythm_fit(
    notes: list[dict], beats: list[float],
    bpm: float = 0.0, beat_offset: float = 0.0,
) -> tuple[float, dict]:
    """How well notes align with the beat *subdivision* grid.

    A note on an 8th/16th/triplet is perfectly valid, so we measure each
    note's distance to the nearest clean subdivision of a beat (1/16 and
    1/12 for triplets) rather than to whole beats only.
    """
    if not notes:
        return 50.0, {"aligned_pct": 0}

    # Prefer BPM-based subdivision check; fall back to whole-beat list.
    if bpm and bpm > 0:
        beat = 60.0 / bpm
        # Acceptable offsets within a beat: 16th-note and triplet grids.
        subs = [i / 16.0 for i in range(16)] + [i / 12.0 for i in range(12)]
        tol = 0.045  # fraction of a beat (~22 ms at 128 BPM)
        aligned = 0
        for n in notes:
            bp = (n["time"] - beat_offset) / beat
            frac = bp - math.floor(bp)
            dist = min(min(abs(frac - s), abs(frac - s - 1.0)) for s in subs)
            if dist < tol:
                aligned += 1
        pct = aligned / len(notes)
        return min(100.0, pct * 102), {"aligned_pct": round(pct * 100, 1)}

    if not beats:
        return 70.0, {"aligned_pct": 0}
    threshold = 0.04
    aligned = sum(
        1 for n in notes
        if any(abs(n["time"] - b) < threshold for b in beats if abs(n["time"] - b) < 0.5)
    )
    pct = aligned / len(notes)
    return min(100.0, pct * 105), {"aligned_pct": round(pct * 100, 1)}


def _score_hand_movement(notes: list[dict], keys: int) -> tuple[float, dict]:
    """Score hand movement quality — not too jumpy, not too static."""
    if len(notes) < 3:
        return 80.0, {"avg_dist": 0, "max_streak": 0}

    sorted_notes = sorted(notes, key=lambda n: n["time"])
    dists: list[int] = []
    same_lane_streak = 0
    max_streak = 0
    prev_lane = -1

    for n in sorted_notes:
        lane = n["lane"]
        if prev_lane >= 0:
            d = abs(lane - prev_lane)
            dists.append(d)
            if lane == prev_lane:
                same_lane_streak += 1
                max_streak = max(max_streak, same_lane_streak)
            else:
                same_lane_streak = 0
        prev_lane = lane

    if not dists:
        return 80.0, {"avg_dist": 0, "max_streak": 0}

    avg = sum(dists) / len(dists)
    ideal_low = 0.8
    ideal_high = 2.2
    if ideal_low <= avg <= ideal_high:
        dist_score = 100.0
    elif avg < ideal_low:
        dist_score = max(40.0, 100.0 - (ideal_low - avg) * 120)
    else:
        dist_score = max(40.0, 100.0 - (avg - ideal_high) * 40)

    # Penalize long same-lane streaks (jacks)
    jack_penalty = min(30.0, max_streak * 5.0) if max_streak > 3 else 0
    score = max(0, dist_score - jack_penalty)

    # Check for hand crossings (L hand going right of R hand)
    crossing_count = 0
    mid = keys / 2
    last_left = -1
    last_right = keys
    for n in sorted_notes:
        lane = n["lane"]
        if lane < mid:
            last_left = lane
        else:
            last_right = lane
        if last_left >= 0 and last_right < keys and last_left > last_right:
            crossing_count += 1

    cross_penalty = min(15.0, crossing_count * 0.5)
    score = max(0, score - cross_penalty)

    return round(score, 1), {
        "avg_dist": round(avg, 2),
        "max_jack_streak": max_streak,
        "crossings": crossing_count,
    }


def _score_pattern_diversity(notes: list[dict], keys: int) -> tuple[float, dict]:
    """Score how diverse the lane patterns are using N-gram analysis."""
    if len(notes) < 8:
        return 70.0, {"unique_3grams": 0, "total_3grams": 0}

    sorted_notes = sorted(notes, key=lambda n: n["time"])
    lanes = [n["lane"] for n in sorted_notes]

    # 3-gram analysis
    trigrams: list[tuple[int, ...]] = []
    for i in range(len(lanes) - 2):
        trigrams.append(tuple(lanes[i:i + 3]))

    if not trigrams:
        return 70.0, {"unique_3grams": 0, "total_3grams": 0}

    counter = Counter(trigrams)
    unique = len(counter)
    total = len(trigrams)
    max_possible = min(keys ** 3, total)
    diversity_ratio = unique / max(1, max_possible)

    # Also check 2-gram diversity
    bigrams = [tuple(lanes[i:i + 2]) for i in range(len(lanes) - 1)]
    bi_counter = Counter(bigrams)
    bi_unique = len(bi_counter)
    bi_max = min(keys ** 2, len(bigrams))
    bi_ratio = bi_unique / max(1, bi_max)

    score = min(100.0, (diversity_ratio * 60 + bi_ratio * 40) * 130)

    return round(score, 1), {
        "unique_3grams": unique,
        "total_3grams": total,
        "unique_2grams": bi_unique,
    }


def _score_repetition(notes: list[dict]) -> tuple[float, dict]:
    """Score repetition balance — some repetition is good, too much is boring."""
    if len(notes) < 16:
        return 75.0, {"repeat_ratio": 0}

    sorted_notes = sorted(notes, key=lambda n: n["time"])
    lanes = [n["lane"] for n in sorted_notes]

    # Check 4-gram repetition
    quads = [tuple(lanes[i:i + 4]) for i in range(len(lanes) - 3)]
    counter = Counter(quads)
    if not quads:
        return 75.0, {"repeat_ratio": 0}

    repeated = sum(1 for c in counter.values() if c > 1)
    repeat_ratio = repeated / max(1, len(counter))

    # Sweet spot: 15–40% repetition
    if 0.15 <= repeat_ratio <= 0.40:
        score = 100.0
    elif repeat_ratio < 0.15:
        score = max(50.0, 100.0 - (0.15 - repeat_ratio) * 400)
    else:
        score = max(50.0, 100.0 - (repeat_ratio - 0.40) * 200)

    return round(score, 1), {"repeat_ratio": round(repeat_ratio, 3)}


def _score_long_notes(notes: list[dict], difficulty: str) -> tuple[float, dict]:
    """Score long note usage — should match difficulty expectations."""
    if not notes:
        return 70.0, {"hold_ratio": 0}

    holds = sum(1 for n in notes if n.get("kind") == "hold")
    ratio = holds / len(notes)
    lo, hi = HOLD_RATIO_RANGES.get(difficulty, (0.05, 0.20))

    if lo <= ratio <= hi:
        score = 100.0
    elif ratio < lo:
        score = max(40.0, 100.0 - (lo - ratio) * 800)
    else:
        score = max(40.0, 100.0 - (ratio - hi) * 500)

    return round(score, 1), {"hold_ratio": round(ratio, 3), "target": f"{lo:.0%}–{hi:.0%}"}


def _score_difficulty_fit(
    notes: list[dict], difficulty: str, duration: float,
) -> tuple[float, dict]:
    """Score whether note density matches the target difficulty."""
    if not notes or duration <= 0:
        return 50.0, {"nps": 0}

    active_dur = max(1.0, duration - 1.0)
    nps = len(notes) / active_dur
    lo, hi = DENSITY_RANGES.get(difficulty, (2.0, 10.0))

    if lo <= nps <= hi:
        score = 100.0
    elif nps < lo:
        score = max(30.0, 100.0 - (lo - nps) / lo * 100)
    else:
        score = max(30.0, 100.0 - (nps - hi) / hi * 80)

    return round(score, 1), {"nps": round(nps, 2), "target": f"{lo}–{hi}"}


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_chart(
    chart: dict,
    keys: int = 4,
    difficulty: str = "hard",
    beats: list[float] | None = None,
) -> EvalResult:
    """Evaluate a chart across all quality dimensions."""
    notes = chart.get("notes", [])
    meta = chart.get("metadata", {})
    duration = meta.get("duration", 0)

    r_fit, r_det = _score_rhythm_fit(
        notes, beats or [],
        bpm=meta.get("bpm", 0.0), beat_offset=meta.get("beat_offset", 0.0),
    )
    h_mov, h_det = _score_hand_movement(notes, keys)
    p_div, p_det = _score_pattern_diversity(notes, keys)
    rep, rep_det = _score_repetition(notes)
    lng, lng_det = _score_long_notes(notes, difficulty)
    d_fit, d_det = _score_difficulty_fit(notes, difficulty, duration)

    weights = {
        "rhythm_fit": 0.20,
        "hand_movement": 0.20,
        "pattern_diversity": 0.15,
        "repetition": 0.15,
        "long_note": 0.10,
        "difficulty_fit": 0.20,
    }
    overall = (
        r_fit * weights["rhythm_fit"]
        + h_mov * weights["hand_movement"]
        + p_div * weights["pattern_diversity"]
        + rep * weights["repetition"]
        + lng * weights["long_note"]
        + d_fit * weights["difficulty_fit"]
    )

    return EvalResult(
        rhythm_fit=r_fit,
        hand_movement=h_mov,
        pattern_diversity=p_div,
        repetition=rep,
        long_note=lng,
        difficulty_fit=d_fit,
        overall=round(overall, 1),
        details={
            "rhythm": r_det,
            "hand": h_det,
            "pattern": p_det,
            "repetition": rep_det,
            "long_note": lng_det,
            "difficulty": d_det,
            "note_count": len(notes),
        },
    )


# ---------------------------------------------------------------------------
# Auto-improver: generate multiple seeds, keep the best
# ---------------------------------------------------------------------------

def auto_improve(
    analysis,
    keys: int = 4,
    difficulty: str = "hard",
    style: str = "auto",
    humanize: bool = True,
    attempts: int = 5,
    threshold: float = 75.0,
    base_seed: int = 42,
) -> tuple[dict, EvalResult]:
    """Generate multiple candidate charts and return the highest-scoring one.

    This is the "generate many → evaluate → keep the best" strategy at the
    whole-chart level: each attempt uses a different seed (so the pattern
    search, motif transforms and humanization diverge), every candidate is
    scored across all quality dimensions, and the best overall wins.  If a
    candidate clears *threshold* early we stop (no need to keep searching).

    The chosen chart carries a ``metadata.generation`` report documenting the
    candidate comparison, so the selection is explainable — you can see the
    score of every attempt and why the winner won.
    """
    from chart_engine import generate_chart

    best_chart = None
    best_eval: EvalResult | None = None
    best_seed = base_seed
    candidates: list[dict] = []

    for i in range(attempts):
        seed = base_seed + i * 7
        chart = generate_chart(analysis, keys, difficulty, seed, humanize, style)
        ev = evaluate_chart(
            chart, keys, difficulty,
            beats=getattr(analysis, "beats", None),
        )

        candidates.append({
            "seed": seed,
            "overall": ev.overall,
            "rhythm_fit": ev.rhythm_fit,
            "hand_movement": ev.hand_movement,
            "pattern_diversity": ev.pattern_diversity,
            "repetition": ev.repetition,
            "long_note": ev.long_note,
            "difficulty_fit": ev.difficulty_fit,
            "note_count": len(chart.get("notes", [])),
        })

        if best_eval is None or ev.overall > best_eval.overall:
            best_chart = chart
            best_eval = ev
            best_seed = seed

        if ev.overall >= threshold:
            break

    assert best_chart is not None and best_eval is not None

    # Attach an explainable generation report to the winning chart.
    ranked = sorted(candidates, key=lambda c: c["overall"], reverse=True)
    scores = [c["overall"] for c in candidates]
    best_chart["metadata"]["generation"] = {
        "attempts": len(candidates),
        "chosen_seed": best_seed,
        "chosen_score": best_eval.overall,
        "score_range": [round(min(scores), 1), round(max(scores), 1)],
        "threshold": threshold,
        "stopped_early": best_eval.overall >= threshold,
        "candidates": ranked,
        "summary": _generation_summary(ranked, best_seed, best_eval),
    }
    return best_chart, best_eval


def _generation_summary(ranked: list[dict], best_seed: int,
                        best_eval: EvalResult) -> str:
    """One-line Korean explanation of the multi-candidate selection."""
    n = len(ranked)
    worst = ranked[-1]["overall"]
    best = ranked[0]["overall"]
    # Identify the dimension where the winner most stood out.
    dims = {
        "리듬 정확도": best_eval.rhythm_fit,
        "손 이동": best_eval.hand_movement,
        "패턴 다양성": best_eval.pattern_diversity,
        "반복 균형": best_eval.repetition,
        "롱노트": best_eval.long_note,
        "난이도 적합도": best_eval.difficulty_fit,
    }
    top_dim = max(dims, key=dims.get)
    return (f"후보 {n}개 생성 · 시드 {best_seed}가 {best:.1f}점으로 선정 "
            f"(최저 {worst:.1f}점 대비 +{best - worst:.1f}) · "
            f"강점: {top_dim} {dims[top_dim]:.0f}점")
