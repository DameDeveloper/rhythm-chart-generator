"""Regression test runner for chart quality (⑨).

Generates synthetic test songs with known BPMs, runs the full pipeline
(analyze → generate_chart), and measures objective quality metrics.  Each
metric is checked against an acceptable range so algorithm changes that
silently degrade quality are caught immediately.

Run:  python run_quality_tests.py
Exit code 0 = all pass, 1 = one or more regressions.

Optionally writes/updates a baseline JSON for trend tracking with --baseline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running from the backend dir directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline import analyze
from chart_engine import generate_chart
from chart_evaluator import evaluate_chart, auto_improve
from tests.synth import synth_song, write_wav
from tests.metrics import collect_metrics


# Test matrix: (bpm, duration_seconds)
TEST_SONGS = [
    (90.0, 24.0),
    (120.0, 24.0),
    (128.0, 24.0),
    (140.0, 24.0),
    (174.0, 24.0),
]

DIFFICULTIES = ["easy", "normal", "hard", "expert", "master"]

# Acceptable metric ranges (min, max).  None = unbounded on that side.
THRESHOLDS = {
    "overlaps":       (0, 0),         # MUST be zero
    "bpm_error_pct":  (None, 6.0),    # detected BPM within 6%
    "diversity":      (0.25, None),   # at least 25% unique 3-grams
    "max_jack":       (None, 12),     # no absurd jack walls
    "grid_align":     (0.85, None),   # 85%+ notes on grid
}

# Per-difficulty NPS sanity ranges.  Lower bounds are loose because short
# synthetic songs are legitimately sparse; the point is to catch a chart
# that is wildly too dense or empty, not to grade musicality.
NPS_RANGES = {
    "easy":   (0.10, 4.0),
    "normal": (0.40, 8.0),
    "hard":   (0.70, 13.0),
    "expert": (1.50, 20.0),
    "master": (3.00, 30.0),
}


def check(value, lo, hi) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def run() -> int:
    failures: list[str] = []
    rows: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="chartqa_"))

    print("=" * 96)
    print(f"{'song':>10} {'diff':>7} {'bpmErr%':>8} {'notes':>6} {'nps':>6} "
          f"{'div':>5} {'rep':>5} {'hand':>5} {'jack':>5} {'burst':>6} {'grid':>5} {'score':>6} {'ovl':>4}")
    print("-" * 96)

    for bpm, dur in TEST_SONGS:
        wav = write_wav(tmp / f"song_{int(bpm)}.wav", synth_song(bpm, dur, seed=int(bpm)))
        analysis = analyze(wav)
        bpm_err = abs(analysis.bpm - bpm) / bpm * 100.0
        # The analyzer may lock onto a half/double octave — fold for fairness.
        for mult in (0.5, 2.0):
            alt = abs(analysis.bpm - bpm * mult) / (bpm * mult) * 100.0
            bpm_err = min(bpm_err, alt)

        if not check(bpm_err, *THRESHOLDS["bpm_error_pct"]):
            failures.append(f"{int(bpm)}BPM: bpm_error {bpm_err:.1f}% out of range")

        for diff in DIFFICULTIES:
            chart = generate_chart(analysis, 4, diff, seed=42, style="auto")
            m = collect_metrics(chart, 4)
            ev = evaluate_chart(chart, 4, diff, beats=analysis.beats)

            # Threshold checks
            tag = f"{int(bpm)}BPM/{diff}"
            if not check(m["overlaps"], *THRESHOLDS["overlaps"]):
                failures.append(f"{tag}: {m['overlaps']} hold/tap overlaps")
            if not check(m["diversity"], *THRESHOLDS["diversity"]):
                failures.append(f"{tag}: diversity {m['diversity']} too low")
            if not check(m["max_jack"], *THRESHOLDS["max_jack"]):
                failures.append(f"{tag}: max_jack {m['max_jack']} too high")
            if not check(m["grid_align"], *THRESHOLDS["grid_align"]):
                failures.append(f"{tag}: grid_align {m['grid_align']} too low")
            nlo, nhi = NPS_RANGES[diff]
            if not check(m["nps"], nlo, nhi):
                failures.append(f"{tag}: nps {m['nps']} outside {nlo}-{nhi}")

            print(f"{int(bpm):>8}Hz {diff:>7} {bpm_err:>8.1f} {m['note_count']:>6} "
                  f"{m['nps']:>6.1f} {m['diversity']:>5.2f} {m['repetition']:>5.2f} "
                  f"{m['avg_hand_dist']:>5.2f} {m['max_jack']:>5} {m['max_burst_nps']:>6.1f} "
                  f"{m['grid_align']:>5.2f} {ev.overall:>6.1f} {m['overlaps']:>4}")

            rows.append({"song": int(bpm), "diff": diff, "bpm_err": round(bpm_err, 2),
                         **m, "score": ev.overall})

    print("=" * 96)

    if "--baseline" in sys.argv:
        base = Path(__file__).resolve().parent / "tests" / "baseline.json"
        base.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Baseline written to {base}")

    if failures:
        print(f"\n[FAIL] {len(failures)} REGRESSION(S):")
        for f in failures:
            print(f"   - {f}")
        return 1

    avg_score = sum(r["score"] for r in rows) / max(1, len(rows))
    print(f"\n[OK] ALL {len(rows)} CHECKS PASSED   (avg quality score: {avg_score:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
