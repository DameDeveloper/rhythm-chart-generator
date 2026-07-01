"""Convert the internal chart representation into the `ryuthm-chart` format.

Internal notes use seconds + {kind, duration}; the ryuthm-chart format uses
milliseconds + {type, endMs} and carries song metadata, timing points, a
difficulty level, and an upper-cased difficulty name.
"""

from __future__ import annotations

# Difficulty name -> numeric level shown in song lists.
LEVELS = {
    "easy": 2,
    "normal": 5,
    "hard": 8,
    "expert": 12,
    "master": 15,
}


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def to_ryuthm_chart(internal: dict, song: dict) -> dict:
    """Build a ryuthm-chart document from an internal chart + song metadata."""
    md = internal["metadata"]
    bpm = md["bpm"]
    offset_ms = _ms(md.get("beat_offset", 0.0))
    difficulty = str(md.get("difficulty", "normal")).lower()

    notes: list[dict] = []
    for n in internal.get("notes", []):
        entry = {
            "timeMs": _ms(n["time"]),
            "lane": int(n["lane"]),
            "type": n.get("kind", "tap"),
        }
        if n.get("kind") == "hold" and n.get("duration", 0) > 0:
            entry["endMs"] = _ms(n["time"] + n["duration"])
        notes.append(entry)

    # Emit one timingPoint per tempo-map entry so BPM changes are preserved.
    raw_map = md.get("tempo_map") or []
    if raw_map:
        timing_points = [
            {"timeMs": _ms(tp["time"]), "bpm": tp["bpm"], "beatsPerBar": 4}
            for tp in raw_map
        ]
    else:
        timing_points = [{"timeMs": 0, "bpm": bpm, "beatsPerBar": 4}]

    return {
        "format": "ryuthm-chart",
        "version": 1,
        "song": {
            "id": song.get("id", ""),
            "title": song.get("title", ""),
            "artist": song.get("artist", ""),
            "youtubeId": song.get("youtubeId", ""),
            "bpm": bpm,
            "offsetMs": offset_ms,
        },
        "lanes": int(md["keys"]),
        "difficulty": difficulty.upper(),
        "level": LEVELS.get(difficulty, 5),
        "data": {
            "formatVersion": 1,
            "timingPoints": timing_points,
            "notes": notes,
        },
    }
