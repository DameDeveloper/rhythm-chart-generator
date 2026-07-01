from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import imageio_ffmpeg
os.environ["PATH"] = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent) + os.pathsep + os.environ.get("PATH", "")

from audio_pipeline import analyze, download_youtube, fetch_youtube_meta, parse_youtube_id
from chart_engine import generate_chart, DIFFICULTIES, LANE_COLORS
from chart_evaluator import auto_improve, evaluate_chart
from ryuthm_chart import to_ryuthm_chart

app = FastAPI(title="Rhythm Chart Generator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA = Path(__file__).resolve().parent / "downloads"
DATA.mkdir(parents=True, exist_ok=True)


class PipelineRequest(BaseModel):
    url: str
    keys: int = 4
    difficulty: str = "hard"
    style: str = "auto"
    seed: int = 42
    humanize: bool = True


class RegenerateRequest(BaseModel):
    keys: int = 4
    difficulty: str = "hard"
    style: str = "auto"
    seed: int = 42
    humanize: bool = True


class ChartSaveRequest(BaseModel):
    chart: dict


def cleanup_jobs(keep: str | None = None) -> None:
    """Remove all previous job folders so audio/charts don't accumulate.

    We only ever need the currently-edited song on disk (for playback and
    regeneration). Everything else is transient and is deleted as soon as a
    new song is processed.
    """
    for child in DATA.iterdir():
        if child.is_dir() and child.name != keep:
            shutil.rmtree(child, ignore_errors=True)


def finalize(job_dir: Path, job_id: str, chart: dict, song: dict) -> dict:
    """Attach metadata and persist only the tiny song.json (for regeneration).

    The chart itself is NOT written to disk — it lives in the browser and is
    only turned into a file when the user explicitly exports it.
    """
    chart["metadata"]["job_id"] = job_id
    chart["metadata"]["song"] = song
    (job_dir / "song.json").write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    return chart


def load_song(job_dir: Path, job_id: str) -> dict:
    song_path = job_dir / "song.json"
    if song_path.exists():
        return json.loads(song_path.read_text(encoding="utf-8"))
    return {"id": job_id, "title": "", "artist": "", "youtubeId": ""}


def ensure_wav(src: Path, job_dir: Path) -> Path:
    """Produce a single audio.wav for the job, converting if needed.

    The original/intermediate file is removed afterwards so only the wav the
    editor needs for playback remains.
    """
    wav_path = job_dir / "audio.wav"
    if src.suffix.lower() == ".wav":
        if src.resolve() != wav_path.resolve():
            shutil.move(str(src), str(wav_path))
        return wav_path
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다.")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-ar", "44100", "-ac", "1", str(wav_path)],
        check=True, capture_output=True, timeout=120,
    )
    src.unlink(missing_ok=True)
    return wav_path


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    keys: int = Form(4),
    difficulty: str = Form("hard"),
    style: str = Form("auto"),
    seed: int = Form(42),
    humanize: bool = Form(True),
):
    if difficulty not in DIFFICULTIES:
        raise HTTPException(400, f"난이도는 {list(DIFFICULTIES)} 중 하나여야 합니다.")
    if keys not in range(4, 9):
        raise HTTPException(400, "키 수는 4~8이어야 합니다.")

    job_id = uuid.uuid4().hex[:12]
    cleanup_jobs(keep=job_id)
    job_dir = DATA / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    uploaded = job_dir / f"source{ext}"
    with uploaded.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        wav_path = ensure_wav(uploaded, job_dir)
    except Exception as e:
        raise HTTPException(500, f"오디오 변환 실패: {e}")

    analysis = analyze(wav_path)
    chart, ev = auto_improve(analysis, keys, difficulty, style, humanize,
                             attempts=5, base_seed=seed)
    chart["metadata"]["eval"] = {
        "overall": ev.overall,
        "rhythm_fit": ev.rhythm_fit,
        "hand_movement": ev.hand_movement,
        "pattern_diversity": ev.pattern_diversity,
        "repetition": ev.repetition,
        "long_note": ev.long_note,
        "difficulty_fit": ev.difficulty_fit,
    }

    song = {
        "id": job_id,
        "title": Path(file.filename or "업로드 트랙").stem,
        "artist": "",
        "youtubeId": "",
    }
    return finalize(job_dir, job_id, chart, song)


@app.post("/pipeline")
async def pipeline(req: PipelineRequest):
    url = req.url.strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(400, "YouTube URL만 지원합니다.")
    if req.difficulty not in DIFFICULTIES:
        raise HTTPException(400, f"난이도는 {list(DIFFICULTIES)} 중 하나여야 합니다.")
    if req.keys not in range(4, 9):
        raise HTTPException(400, "키 수는 4~8이어야 합니다.")

    job_id = uuid.uuid4().hex[:12]
    cleanup_jobs(keep=job_id)
    job_dir = DATA / job_id
    try:
        wav_path = download_youtube(url, job_dir)
    except Exception as e:
        raise HTTPException(500, f"다운로드 실패: {e}")

    analysis = analyze(wav_path)
    chart, ev = auto_improve(analysis, req.keys, req.difficulty, req.style,
                             req.humanize, attempts=5, base_seed=req.seed)
    chart["metadata"]["eval"] = {
        "overall": ev.overall,
        "rhythm_fit": ev.rhythm_fit,
        "hand_movement": ev.hand_movement,
        "pattern_diversity": ev.pattern_diversity,
        "repetition": ev.repetition,
        "long_note": ev.long_note,
        "difficulty_fit": ev.difficulty_fit,
    }

    meta = fetch_youtube_meta(url)
    vid = meta.get("youtubeId") or parse_youtube_id(url)
    song = {
        "id": vid or job_id,
        "title": meta.get("title") or vid or "YouTube 트랙",
        "artist": meta.get("artist") or "",
        "youtubeId": vid,
    }
    return finalize(job_dir, job_id, chart, song)


@app.post("/regenerate/{job_id}")
async def regenerate(job_id: str, req: RegenerateRequest):
    job_dir = DATA / job_id
    wav_files = sorted(job_dir.glob("audio*.wav"))
    if not wav_files:
        raise HTTPException(404, "해당 작업을 찾을 수 없습니다.")

    analysis = analyze(wav_files[0])
    chart, ev = auto_improve(analysis, req.keys, req.difficulty, req.style,
                             req.humanize, attempts=5, base_seed=req.seed)
    chart["metadata"]["eval"] = {
        "overall": ev.overall,
        "rhythm_fit": ev.rhythm_fit,
        "hand_movement": ev.hand_movement,
        "pattern_diversity": ev.pattern_diversity,
        "repetition": ev.repetition,
        "long_note": ev.long_note,
        "difficulty_fit": ev.difficulty_fit,
    }

    song = load_song(job_dir, job_id)
    return finalize(job_dir, job_id, chart, song)


@app.get("/audio/{job_id}")
async def get_audio(job_id: str):
    job_dir = DATA / job_id
    wav_files = sorted(job_dir.glob("audio*.wav"))
    if not wav_files:
        raise HTTPException(404, "오디오 파일을 찾을 수 없습니다.")
    return FileResponse(wav_files[0], media_type="audio/wav")


@app.post("/export")
async def export_chart(req: ChartSaveRequest):
    """Convert the (possibly edited) chart to ryuthm-chart format on demand.

    Nothing is written to the server; the JSON is streamed back so the browser
    saves it to the user's Downloads folder only when they click Export.
    """
    chart = req.chart
    song = chart.get("metadata", {}).get("song") or {
        "id": "", "title": "", "artist": "", "youtubeId": "",
    }
    ryuthm = to_ryuthm_chart(chart, song)
    body = json.dumps(ryuthm, ensure_ascii=False, indent=2)
    title = (song.get("title") or "chart").strip() or "chart"
    safe = "".join(c for c in title if c.isalnum() or c in " _-").strip() or "chart"
    # RFC 5987: use filename* for non-ASCII, plain filename as ASCII fallback
    ascii_safe = "".join(c for c in safe if ord(c) < 128).strip() or "chart"
    from urllib.parse import quote
    encoded = quote(safe, safe="")
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_safe}.ryuthm.json\"; "
                f"filename*=UTF-8''{encoded}.ryuthm.json"
            ),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
