"""Load pattern definitions from JSON files in the patterns/ directory.

Each JSON file contains an array of pattern objects.  Steps can be:
  * A float  (single note, normalized 0–1)
  * A list   (chord — multiple simultaneous notes)
  * A string starting with "H" followed by a float (hold note start)

The loader converts these to the internal PatternDef format used by
chart_engine.
"""

from __future__ import annotations

import json
from pathlib import Path

PATTERNS_DIR = Path(__file__).resolve().parent / "patterns"

DIFF_IDX = {"easy": 0, "normal": 1, "hard": 2, "expert": 3, "master": 4}


def _parse_step(raw):
    """Convert a JSON step value to the internal representation.

    * float → float
    * list  → tuple of floats (chord)
    * "H0.5" → float (hold note — handled by chart engine)
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, list):
        return tuple(float(v) for v in raw)
    if isinstance(raw, str) and raw.startswith("H"):
        return float(raw[1:])
    return float(raw)


def load_patterns() -> list[dict]:
    """Load all pattern definitions from patterns/*.json.

    Returns a list of dicts compatible with PatternDef construction.
    """
    if not PATTERNS_DIR.is_dir():
        return []

    patterns: list[dict] = []
    for path in sorted(PATTERNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            steps = [_parse_step(s) for s in entry.get("steps", [])]
            if not steps:
                continue
            sections = tuple(entry.get("sections", ["verse", "chorus"]))
            min_diff = DIFF_IDX.get(entry.get("min_diff", "easy"), 0)
            intensity_min = entry.get("intensity_min", 0.0)
            patterns.append({
                "name": entry.get("name", "unknown"),
                "category": entry.get("category", "misc"),
                "steps": steps,
                "sections": sections,
                "min_diff": min_diff,
                "intensity_min": intensity_min,
                "min_keys": entry.get("min_keys", 4),
                "weight": entry.get("weight", 1.0),
            })
    return patterns


_CACHE: list[dict] | None = None


def get_patterns() -> list[dict]:
    """Cached pattern loader — loads once per process."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_patterns()
    return _CACHE


_TRANSITION_CACHE: dict[str, dict[str, float]] | None = None


def get_transition_matrix() -> dict[str, dict[str, float]]:
    """Load the category transition matrix (once per process)."""
    global _TRANSITION_CACHE
    if _TRANSITION_CACHE is None:
        path = PATTERNS_DIR / "transition_matrix.json"
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                _TRANSITION_CACHE = {
                    k: v for k, v in raw.items() if not k.startswith("_")
                }
            except (json.JSONDecodeError, OSError):
                _TRANSITION_CACHE = {}
        else:
            _TRANSITION_CACHE = {}
    return _TRANSITION_CACHE
