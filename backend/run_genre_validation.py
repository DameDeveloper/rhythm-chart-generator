"""Large-scale multi-genre validation harness (⑪).

Generates 100+ charts across ten genres × several tempos × all difficulties,
records the five metrics the user asked to track —

    sync (grid alignment) · density (NPS) · pattern diversity ·
    long-note ratio · hand movement —

and, crucially, scans every chart for *integrity bugs* (overlaps, duplicate
notes, out-of-range lanes/times, NaN/inf, empty charts).  Large batches like
this surface the bugs that a handful of hand-picked test songs never hit.

Run:  python run_genre_validation.py
      python run_genre_validation.py --report   # also write JSON report

Exit code 0 = no integrity bugs, 1 = one or more bugs found.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline import analyze
from chart_engine import generate_chart
from chart_evaluator import evaluate_chart
from tests.synth import synth_genre, write_wav, GENRES
from tests.metrics import collect_metrics


# ---------------------------------------------------------------------------
# Test matrix — 10 genres × 3 tempos × 5 difficulties = 150 charts
# ---------------------------------------------------------------------------

TEMPOS = [100.0, 128.0, 150.0]
DIFFICULTIES = ["easy", "normal", "hard", "expert", "master"]
KEYS = 4
DURATION = 20.0

# Genres whose audio is sustained enough that a chart with *zero* holds is a
# quality red flag (they should carry at least some long notes).
SUSTAINED_GENRES = {"piano", "orchestra", "vocal", "jazz"}


# ---------------------------------------------------------------------------
# Integrity checks — hard bugs that must never occur in any chart
# ---------------------------------------------------------------------------

def check_integrity(chart: dict, keys: int) -> list[str]:
    """Return a list of integrity-bug descriptions (empty = clean)."""
    bugs: list[str] = []
    notes = chart.get("notes", [])
    meta = chart.get("metadata", {})
    duration = meta.get("duration", 0.0)

    if not notes:
        bugs.append("empty chart (0 notes)")
        return bugs

    seen: set[tuple] = set()
    for i, n in enumerate(notes):
        t = n.get("time")
        lane = n.get("lane")
        dur = n.get("duration", 0.0)

        # NaN / inf
        for label, val in (("time", t), ("duration", dur)):
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                bugs.append(f"note {i}: invalid {label}={val}")

        # Lane range
        if lane is None or not (0 <= lane < keys):
            bugs.append(f"note {i}: lane {lane} out of range [0,{keys})")

        # Time range (allow tiny epsilon past end from humanize jitter)
        if t is not None and not (isinstance(t, float) and (math.isnan(t))):
            if t < -0.01 or t > duration + 0.5:
                bugs.append(f"note {i}: time {t} outside [0,{duration}]")

        # Negative / absurd duration
        if isinstance(dur, (int, float)) and dur < 0:
            bugs.append(f"note {i}: negative duration {dur}")

        # Exact duplicate (same time+lane)
        key = (round(float(t), 3) if isinstance(t, (int, float)) else t, lane)
        if key in seen:
            bugs.append(f"note {i}: duplicate at time={t} lane={lane}")
        seen.add(key)

    # Hold/tap overlap on the same lane (already have a metric, double-check)
    holds = [n for n in notes if n.get("kind") == "hold" and n.get("duration", 0) > 0]
    for h in holds:
        h_end = h["time"] + h["duration"]
        for m in notes:
            if m is h:
                continue
            if m["lane"] == h["lane"] and h["time"] < m["time"] < h_end - 1e-6:
                bugs.append(f"hold@{h['time']:.2f} lane{h['lane']} overlaps note@{m['time']:.2f}")
                break

    return bugs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="genreval_"))
    rows: list[dict] = []
    bugs: list[str] = []
    anomalies: list[str] = []
    t0 = time.time()

    print("=" * 104)
    print("LARGE-SCALE MULTI-GENRE VALIDATION")
    print(f"{len(GENRES)} genres × {len(TEMPOS)} tempos × {len(DIFFICULTIES)} difficulties "
          f"= {len(GENRES) * len(TEMPOS) * len(DIFFICULTIES)} charts")
    print("=" * 104)
    print(f"{'genre':>10} {'bpm':>5} {'diff':>7} {'notes':>6} {'sync':>5} {'dens':>5} "
          f"{'div':>5} {'long':>5} {'hand':>5} {'jack':>5} {'ovl':>4} {'score':>6}  bugs")
    print("-" * 104)

    for genre in GENRES:
        for bpm in TEMPOS:
            seed = int(bpm) + hash(genre) % 1000
            wav = write_wav(tmp / f"{genre}_{int(bpm)}.wav",
                            synth_genre(genre, bpm, DURATION, seed=seed))
            analysis = analyze(wav)

            per_diff_nps: dict[str, float] = {}
            for diff in DIFFICULTIES:
                chart = generate_chart(analysis, KEYS, diff, seed=42, style="auto")
                m = collect_metrics(chart, KEYS)
                ev = evaluate_chart(chart, KEYS, diff, beats=analysis.beats)
                integ = check_integrity(chart, KEYS)

                tag = f"{genre}/{int(bpm)}/{diff}"
                for b in integ:
                    bugs.append(f"{tag}: {b}")

                # Soft anomalies (quality warnings, not hard bugs).
                if m["grid_align"] < 0.80:
                    anomalies.append(f"{tag}: low sync {m['grid_align']}")
                if m["max_jack"] > 16:
                    anomalies.append(f"{tag}: jack wall {m['max_jack']}")
                if m["note_count"] > 0 and m["diversity"] < 0.20 and diff != "easy":
                    anomalies.append(f"{tag}: low diversity {m['diversity']}")
                if (genre in SUSTAINED_GENRES and diff in ("normal", "hard", "expert")
                        and m["note_count"] >= 20 and m["hold_ratio"] == 0.0):
                    anomalies.append(f"{tag}: no long notes in sustained genre")

                per_diff_nps[diff] = m["nps"]
                rows.append({"genre": genre, "bpm": int(bpm), "diff": diff,
                             **m, "score": ev.overall, "bugs": len(integ)})

                flag = f"⚠{len(integ)}" if integ else ""
                print(f"{genre:>10} {int(bpm):>5} {diff:>7} {m['note_count']:>6} "
                      f"{m['grid_align']:>5.2f} {m['nps']:>5.1f} {m['diversity']:>5.2f} "
                      f"{m['hold_ratio']:>5.2f} {m['avg_hand_dist']:>5.2f} {m['max_jack']:>5} "
                      f"{m['overlaps']:>4} {ev.overall:>6.1f}  {flag}")

            # Difficulty monotonicity: master should not be sparser than easy.
            if per_diff_nps["master"] < per_diff_nps["easy"]:
                anomalies.append(
                    f"{genre}/{int(bpm)}: density inverted "
                    f"(easy {per_diff_nps['easy']} > master {per_diff_nps['master']})")

    print("=" * 104)
    _print_summary(rows)

    elapsed = time.time() - t0
    print(f"\nGenerated {len(rows)} charts in {elapsed:.1f}s "
          f"({elapsed / max(1, len(rows)) * 1000:.0f} ms/chart)")

    if anomalies:
        print(f"\n[WARN] {len(anomalies)} quality anomaly(ies):")
        for a in anomalies[:25]:
            print(f"   - {a}")
        if len(anomalies) > 25:
            print(f"   ... and {len(anomalies) - 25} more")

    if "--report" in sys.argv:
        out = Path(__file__).resolve().parent / "tests" / "genre_validation_report.json"
        out.write_text(json.dumps(
            {"rows": rows, "bugs": bugs, "anomalies": anomalies}, indent=2),
            encoding="utf-8")
        print(f"\nReport written to {out}")

    if bugs:
        print(f"\n[FAIL] {len(bugs)} INTEGRITY BUG(S):")
        for b in bugs[:40]:
            print(f"   - {b}")
        if len(bugs) > 40:
            print(f"   ... and {len(bugs) - 40} more")
        return 1

    print(f"\n[OK] NO INTEGRITY BUGS across {len(rows)} charts.")
    return 0


def _print_summary(rows: list[dict]) -> None:
    """Per-genre and per-difficulty aggregate tables for the five metrics."""
    def agg(subset, key):
        vals = [r[key] for r in subset]
        return statistics.mean(vals) if vals else 0.0

    print("\nPER-GENRE AVERAGES (sync / density / diversity / long / hand / score)")
    print("-" * 104)
    by_genre: dict[str, list] = defaultdict(list)
    for r in rows:
        by_genre[r["genre"]].append(r)
    for genre in GENRES:
        s = by_genre.get(genre, [])
        if not s:
            continue
        print(f"{genre:>10}  sync={agg(s,'grid_align'):.2f}  dens={agg(s,'nps'):>5.1f}  "
              f"div={agg(s,'diversity'):.2f}  long={agg(s,'hold_ratio'):.2f}  "
              f"hand={agg(s,'avg_hand_dist'):.2f}  score={agg(s,'score'):>5.1f}")

    print("\nPER-DIFFICULTY AVERAGES")
    print("-" * 104)
    by_diff: dict[str, list] = defaultdict(list)
    for r in rows:
        by_diff[r["diff"]].append(r)
    for diff in DIFFICULTIES:
        s = by_diff.get(diff, [])
        if not s:
            continue
        print(f"{diff:>10}  sync={agg(s,'grid_align'):.2f}  dens={agg(s,'nps'):>5.1f}  "
              f"div={agg(s,'diversity'):.2f}  long={agg(s,'hold_ratio'):.2f}  "
              f"hand={agg(s,'avg_hand_dist'):.2f}  score={agg(s,'score'):>5.1f}")


if __name__ == "__main__":
    sys.exit(run())
