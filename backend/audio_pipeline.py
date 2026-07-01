"""Audio download, analysis, band separation, and structure detection.

Covers Steps 3, 5-10 of the roadmap:
  - YouTube download via yt-dlp
  - BPM / beat detection
  - Measure calculation
  - Frequency-band separation (lightweight Demucs alternative)
  - Onset detection
  - Song structure analysis (intro/verse/chorus/bridge/outro)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _decode_text(raw: bytes) -> str:
    """Decode subprocess output, tolerating UTF-8 / cp949 / euc-kr.

    yt-dlp on Windows may emit titles in the system code page (cp949 for
    Korean) rather than UTF-8, which otherwise turns Hangul and special
    characters into replacement boxes.
    """
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _utf8_env() -> dict:
    """Environment that nudges yt-dlp/Python child processes toward UTF-8."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def get_ffmpeg() -> str:
    """Return a usable ffmpeg path: system ffmpeg, else the bundled binary."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError("ffmpeg를 찾을 수 없습니다. 'pip install imageio-ffmpeg'를 실행하세요.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TempoPoint:
    """A point in the tempo map: BPM changes at *time* seconds."""
    time: float
    bpm: float


@dataclass
class Section:
    label: str
    start: float
    end: float
    energy: float


@dataclass
class BandEnergy:
    """Per-frame energy for low / mid / high frequency bands."""
    low: list[float] = field(default_factory=list)
    mid: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)


@dataclass
class FocusSegment:
    """A time segment where a particular instrument dominates."""
    instrument: str   # vocal, drums, guitar, bass, keys, mixed
    start: float
    end: float
    confidence: float  # 0..1 how clearly this instrument dominates


@dataclass
class FeatureVector:
    """Per-frame musical feature vectors beyond BPM/pitch/energy.

    Every list is aligned to ``AnalysisResult.frame_times``.  These richer
    descriptors let the chart engine choose patterns that match not just the
    loudness of a moment but its *musical character* — is it a sustained vocal
    line, a drum fill, a tense build-up?
    """
    spectral_flux: list[float] = field(default_factory=list)      # onset-like timbral change
    instrument_change: list[float] = field(default_factory=list)  # band-distribution change rate
    harmonic_ratio: list[float] = field(default_factory=list)     # 0=percussive .. 1=harmonic
    chord_change: list[float] = field(default_factory=list)       # chroma novelty (chord progression)
    vocal_presence: list[float] = field(default_factory=list)     # 0..1 probability a vocal is present
    drum_fill: list[float] = field(default_factory=list)          # 0..1 probability of a drum fill
    tension: list[float] = field(default_factory=list)            # 0..1 musical tension / build


@dataclass
class AnalysisResult:
    wav_path: str
    sample_rate: int
    duration: float
    bpm: float
    beat_offset: float
    beats: list[float]
    measures: list[tuple[float, float]]
    frame_hop: float
    frame_times: list[float]
    onset_strength: list[float]
    rms: list[float]
    tempo_map: list[TempoPoint] = field(default_factory=list)
    melody: list[float] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    bands: BandEnergy = field(default_factory=BandEnergy)
    focus_segments: list[FocusSegment] = field(default_factory=list)
    features: FeatureVector = field(default_factory=FeatureVector)


# ---------------------------------------------------------------------------
# YouTube download
# ---------------------------------------------------------------------------

def parse_youtube_id(url: str) -> str:
    """Extract the 11-char video id from common YouTube URL shapes."""
    import re
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else ""


def fetch_youtube_meta(url: str) -> dict:
    """Best-effort fetch of title/uploader/id without downloading media."""
    vid = parse_youtube_id(url)
    meta = {"title": "", "artist": "", "youtubeId": vid, "id": vid}
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-playlist", "--skip-download", "--encoding", "utf-8",
             "--print", "%(title)s|||%(uploader)s|||%(id)s", url],
            capture_output=True, timeout=60, env=_utf8_env(),
        )
        if out.returncode == 0:
            line = _decode_text(out.stdout).strip().splitlines()
            if line:
                parts = (line[0].split("|||") + ["", "", ""])[:3]
                title, uploader, real_id = parts
                meta.update(title=title.strip(), artist=uploader.strip())
                if real_id.strip():
                    meta["youtubeId"] = meta["id"] = real_id.strip()
    except Exception:
        pass
    return meta


def download_youtube(url: str, output_dir: Path) -> Path:
    """Download YouTube audio, then convert to WAV.

    yt-dlp downloads the raw bestaudio stream (m4a/webm) WITHOUT any
    post-processing, so it never needs to locate ffmpeg itself. We then
    convert to WAV with a known ffmpeg path (system or bundled), which
    avoids the common 'ffmpeg/ffprobe not found' failure on Windows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp", "-f", "bestaudio/best", "--no-playlist",
        "--output", template, url,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180, env=_utf8_env())
    if result.returncode != 0:
        msg = _decode_text(result.stderr).strip()
        raise RuntimeError(msg[-500:] or "yt-dlp 다운로드 실패")

    sources = sorted(output_dir.glob("source.*"))
    if not sources:
        raise RuntimeError("다운로드된 오디오 파일을 찾을 수 없습니다.")
    src = sources[0]

    wav_path = output_dir / "audio.wav"
    ffmpeg = get_ffmpeg()
    conv = subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-ar", "44100", "-ac", "1", str(wav_path)],
        capture_output=True, timeout=180,
    )
    if conv.returncode != 0:
        msg = _decode_text(conv.stderr).strip()
        raise RuntimeError("ffmpeg 변환 실패: " + msg[-400:])
    # Drop the raw download; only the wav is needed for analysis/playback.
    src.unlink(missing_ok=True)
    return wav_path


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as h:
        ch = h.getnchannels()
        sw = h.getsampwidth()
        sr = h.getframerate()
        raw = h.readframes(h.getnframes())

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif sw == 3:
        r = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        s = r[:, 0].astype(np.int32) | (r[:, 1].astype(np.int32) << 8) | (r[:, 2].astype(np.int32) << 16)
        data = np.where(s & 0x800000, s - 0x1000000, s).astype(np.float32)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0

    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return sr, data


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    sr, data = read_wav(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 0:
        data /= peak
    return data, sr


def downsample(audio: np.ndarray, sr: int, target: int = 22050) -> tuple[np.ndarray, int]:
    if sr <= target:
        return audio, sr
    old_t = np.arange(len(audio), dtype=np.float64) / sr
    new_len = int(round(len(audio) * target / sr))
    new_t = np.arange(new_len, dtype=np.float64) / target
    return np.interp(new_t, old_t, audio).astype(np.float32), target


# ---------------------------------------------------------------------------
# Core DSP helpers
# ---------------------------------------------------------------------------

def frame_audio(audio: np.ndarray, sr: int, frame_size: int = 2048, hop: int = 512):
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    count = 1 + max(0, (len(audio) - frame_size) // hop)
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(count, frame_size),
        strides=(audio.strides[0] * hop, audio.strides[0]), writeable=False,
    ).copy()
    times = (np.arange(count) * hop) / sr
    return frames, times


def normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    if hi <= lo:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def find_peaks(values: np.ndarray, height: float, distance: int) -> np.ndarray:
    out: list[int] = []
    last = -distance
    for i in range(1, len(values) - 1):
        if i - last < distance:
            continue
        if values[i] >= height and values[i] >= values[i - 1] and values[i] > values[i + 1]:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# Band separation (Step 8 lightweight alternative to Demucs)
# ---------------------------------------------------------------------------

def compute_melody(spectra: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Estimate a melodic pitch contour (0=low, 1=high) per frame.

    A good human mapper "reads" the melody: as the lead line rises in pitch,
    notes move toward higher lanes. We approximate the lead pitch with the
    spectral centroid inside the melodic band (~150-2500 Hz), mapped onto a
    log-frequency scale and lightly smoothed so the lane motion is musical
    rather than jittery.
    """
    band = (freqs >= 150) & (freqs <= 2500)
    if not band.any():
        return np.full(spectra.shape[0], 0.5)
    sub = spectra[:, band]
    f = freqs[band]
    energy = sub.sum(axis=1) + 1e-9
    centroid = (sub * f).sum(axis=1) / energy
    centroid = np.clip(centroid, f[0], f[-1])
    logc = np.log2(centroid)
    lo, hi = np.log2(f[0]), np.log2(f[-1])
    norm = (logc - lo) / (hi - lo)
    kernel = np.ones(5) / 5.0
    norm = np.convolve(norm, kernel, mode="same")
    return np.clip(norm, 0.0, 1.0)


def separate_bands(frames: np.ndarray, sr: int, frame_size: int = 2048) -> BandEnergy:
    window = np.hanning(frame_size)
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    low_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000

    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    low = normalize(np.sqrt(np.mean(spectra[:, low_mask] ** 2, axis=1)))
    mid = normalize(np.sqrt(np.mean(spectra[:, mid_mask] ** 2, axis=1)))
    high = normalize(np.sqrt(np.mean(spectra[:, high_mask] ** 2, axis=1)))
    return BandEnergy(low=low.tolist(), mid=mid.tolist(), high=high.tolist())


# ---------------------------------------------------------------------------
# BPM / beat / measure detection (Steps 5-7)
# ---------------------------------------------------------------------------

def _local_bpm(onset_seg: np.ndarray, hop: float) -> float:
    """Estimate BPM from a short onset segment via autocorrelation."""
    if len(onset_seg) < 8:
        return 120.0
    centered = onset_seg - np.mean(onset_seg)
    ac = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    min_lag = max(1, int(round(60.0 / 210.0 / hop)))
    max_lag = min(len(ac) - 1, int(round(60.0 / 70.0 / hop)))
    if max_lag <= min_lag:
        return 120.0
    lag = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
    bpm = 60.0 / (lag * hop)
    while bpm < 90:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return bpm


def estimate_tempo(times: np.ndarray, onset: np.ndarray, hop: float) -> tuple[float, float]:
    if len(times) < 4:
        return 120.0, 0.0
    bpm = _local_bpm(onset, hop)

    beat_len = 60.0 / bpm
    peaks = find_peaks(onset, height=float(np.percentile(onset, 72)), distance=max(1, int(0.12 / hop)))
    if len(peaks) == 0:
        return round(bpm, 3), 0.0
    phase = float(np.median(np.mod(times[peaks], beat_len)))
    return round(bpm, 3), round(phase, 4)


def build_tempo_map(
    times: np.ndarray,
    onset: np.ndarray,
    hop: float,
    duration: float,
    global_bpm: float,
    window_sec: float = 8.0,
    hop_sec: float = 4.0,
    change_thresh: float = 3.0,
) -> list[TempoPoint]:
    """Build a tempo map by estimating local BPM in overlapping windows.

    A new TempoPoint is emitted whenever the local BPM deviates from the
    previous segment by more than *change_thresh*.  For EDM / fixed-tempo
    songs this returns a single point (no wasted precision); for live or
    rubato material it captures the drift.
    """
    win_frames = max(8, int(window_sec / hop))
    hop_frames = max(4, int(hop_sec / hop))
    n = len(onset)
    if n < win_frames:
        return [TempoPoint(time=0.0, bpm=round(global_bpm, 3))]

    local_bpms: list[tuple[float, float]] = []
    for start in range(0, n - win_frames + 1, hop_frames):
        seg = onset[start: start + win_frames]
        t = float(times[start]) if start < len(times) else start * hop
        b = _local_bpm(seg, hop)
        local_bpms.append((t, b))

    if not local_bpms:
        return [TempoPoint(time=0.0, bpm=round(global_bpm, 3))]

    # Smooth the raw BPM curve to filter jitter.
    raw = np.array([b for _, b in local_bpms])
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(raw, kernel, mode="same")

    # Segment: only emit a new point when BPM shifts significantly.
    tempo_map: list[TempoPoint] = [
        TempoPoint(time=0.0, bpm=round(float(smoothed[0]), 3))
    ]
    for i, (t, _) in enumerate(local_bpms):
        cur = float(smoothed[i])
        if abs(cur - tempo_map[-1].bpm) >= change_thresh:
            tempo_map.append(TempoPoint(time=round(t, 4), bpm=round(cur, 3)))

    return tempo_map


def bpm_at(tempo_map: list[TempoPoint], t: float) -> float:
    """Look up the BPM at time *t* from a tempo map."""
    result = tempo_map[0].bpm
    for tp in tempo_map:
        if tp.time <= t:
            result = tp.bpm
        else:
            break
    return result


def compute_beats_from_tempo_map(
    tempo_map: list[TempoPoint], offset: float, duration: float,
) -> list[float]:
    """Generate a beat grid that follows the tempo map."""
    beats: list[float] = []
    t = offset
    while t < duration:
        if t >= 0:
            beats.append(round(t, 4))
        local_bpm = bpm_at(tempo_map, t)
        t += 60.0 / local_bpm
    return beats


def compute_beats(bpm: float, offset: float, duration: float) -> list[float]:
    beat_len = 60.0 / bpm
    beats: list[float] = []
    t = offset
    while t < duration:
        if t >= 0:
            beats.append(round(t, 4))
        t += beat_len
    return beats


def compute_measures(beats: list[float], beats_per_measure: int = 4) -> list[tuple[float, float]]:
    measures: list[tuple[float, float]] = []
    for i in range(0, len(beats) - beats_per_measure + 1, beats_per_measure):
        measures.append((beats[i], beats[min(i + beats_per_measure, len(beats) - 1)]))
    return measures


# ---------------------------------------------------------------------------
# Structure detection (Step 10)
# ---------------------------------------------------------------------------

def detect_structure(
    frame_times: list[float],
    rms: list[float],
    duration: float,
    onset: list[float] | None = None,
    high_band: list[float] | None = None,
) -> list[Section]:
    """Detect song structure with pre-chorus and solo awareness.

    Beyond basic energy thresholding, this:
    * inserts **pre_chorus** when energy is rising just before a chorus,
    * labels high-energy + high-treble passages as **solo**,
    * tracks energy gradients for buildup/breakdown detection.
    """
    if not rms:
        return [Section("unknown", 0.0, duration, 0.5)]

    window_sec = 4.0
    hop = frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.023
    window_frames = max(1, int(window_sec / hop))
    rms_arr = np.array(rms)
    n_windows = max(1, len(rms_arr) // window_frames)
    energies: list[float] = []
    hi_energies: list[float] = []
    for i in range(n_windows):
        chunk = rms_arr[i * window_frames: (i + 1) * window_frames]
        energies.append(float(np.mean(chunk)))
        if high_band is not None:
            hi_chunk = np.array(high_band[i * window_frames: (i + 1) * window_frames])
            hi_energies.append(float(np.mean(hi_chunk)) if hi_chunk.size else 0.0)

    if not energies:
        return [Section("unknown", 0.0, duration, 0.5)]

    e = np.array(energies)
    e_norm = (e - e.min()) / (e.max() - e.min() + 1e-9)

    sections: list[Section] = []
    for i, en in enumerate(e_norm):
        start = i * window_sec
        end = min((i + 1) * window_sec, duration)
        if i < len(e_norm) * 0.08:
            label = "intro"
        elif i > len(e_norm) * 0.92:
            label = "outro"
        elif en > 0.7:
            label = "chorus"
        elif en > 0.4:
            label = "verse"
        else:
            label = "bridge"
        sections.append(Section(label=label, start=round(start, 3), end=round(end, 3), energy=round(float(en), 3)))

    # --- Merge adjacent same-label sections ---
    merged: list[Section] = [sections[0]]
    for s in sections[1:]:
        if s.label == merged[-1].label:
            merged[-1].end = s.end
            merged[-1].energy = round((merged[-1].energy + s.energy) / 2, 3)
        else:
            merged.append(s)

    # --- Post-processing: insert pre_chorus before each chorus ---
    enriched: list[Section] = []
    for i, sec in enumerate(merged):
        next_is_chorus = (i + 1 < len(merged) and merged[i + 1].label == "chorus")
        if next_is_chorus and sec.label in ("verse", "bridge") and (sec.end - sec.start) > 6.0:
            split = round(sec.end - min(4.0, (sec.end - sec.start) * 0.35), 3)
            enriched.append(Section(sec.label, sec.start, split, sec.energy))
            enriched.append(Section("pre_chorus", split, sec.end,
                                    round(min(1.0, sec.energy + 0.15), 3)))
        else:
            enriched.append(sec)

    # --- Solo detection: high energy + high treble dominance ---
    if hi_energies:
        hi = np.array(hi_energies)
        hi_norm = (hi - hi.min()) / (hi.max() - hi.min() + 1e-9)
        final: list[Section] = []
        for sec in enriched:
            start_idx = max(0, int(sec.start / window_sec))
            end_idx = min(len(hi_norm), int(sec.end / window_sec) + 1)
            if end_idx > start_idx:
                avg_hi = float(np.mean(hi_norm[start_idx:end_idx]))
                if sec.label == "chorus" and avg_hi > 0.7 and sec.energy > 0.6:
                    final.append(Section("solo", sec.start, sec.end, sec.energy))
                    continue
            final.append(sec)
        return final

    return enriched


# ---------------------------------------------------------------------------
# Music feature vectors (Step 10 — rich per-frame descriptors)
# ---------------------------------------------------------------------------

def _smooth(v: np.ndarray, k: int = 5) -> np.ndarray:
    """Moving-average smoothing with a k-length kernel."""
    if v.size == 0 or k <= 1:
        return v
    kernel = np.ones(k) / k
    return np.convolve(v, kernel, mode="same")


def _rolling_count(mask: np.ndarray, win: int) -> np.ndarray:
    """Count of True values in a trailing window of *win* frames, per frame."""
    if mask.size == 0:
        return mask.astype(np.float32)
    c = np.convolve(mask.astype(np.float32), np.ones(win), mode="same")
    return c / max(1, win)


def compute_chroma(spectra: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """12-bin chroma (pitch-class energy) per frame.

    Each spectral bin above ~55 Hz is folded onto its pitch class using the
    equal-tempered log-frequency mapping; energies are summed per class and
    the vector is L1-normalized so it describes *which notes* are sounding
    regardless of loudness.  This is the basis for chord-change detection.
    """
    n_frames = spectra.shape[0]
    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    valid = freqs > 55.0
    if not valid.any():
        return chroma
    f = freqs[valid]
    sub = spectra[:, valid]
    pcs = (np.round(12 * np.log2(f / 440.0)).astype(int)) % 12
    for pc in range(12):
        m = pcs == pc
        if m.any():
            chroma[:, pc] = sub[:, m].sum(axis=1)
    norm = chroma.sum(axis=1, keepdims=True) + 1e-9
    return chroma / norm


def spectral_flatness(spectra: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness (geometric/arithmetic mean).

    Near 1.0 → broadband / noise-like (percussion, cymbals); near 0.0 →
    peaky / tonal (sustained pitched instruments, vocals).
    """
    s = spectra + 1e-9
    gm = np.exp(np.mean(np.log(s), axis=1))
    am = np.mean(s, axis=1)
    return np.clip(gm / am, 0.0, 1.0)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of *p* (already L1-normalized), 0..1."""
    q = np.clip(p, 1e-9, 1.0)
    ent = -np.sum(q * np.log2(q), axis=1)
    return ent / np.log2(p.shape[1])  # normalize by max entropy


def compute_feature_vectors(
    spectra: np.ndarray,
    freqs: np.ndarray,
    onset: np.ndarray,
    rms: np.ndarray,
    melody: np.ndarray,
    bands: BandEnergy,
    hop: float,
) -> FeatureVector:
    """Compute the seven extended feature vectors from the analysis spectra.

    All outputs are per-frame arrays in [0, 1], aligned to the frame grid.
    """
    n = spectra.shape[0]
    if n == 0:
        return FeatureVector()

    low = np.array(bands.low) if bands.low else np.zeros(n)
    mid = np.array(bands.mid) if bands.mid else np.zeros(n)
    high = np.array(bands.high) if bands.high else np.zeros(n)
    rms_arr = np.asarray(rms, dtype=np.float32)
    onset_arr = np.asarray(onset, dtype=np.float32)

    # --- 1. Spectral flux (raw timbral change, exposed separately from onset)
    raw_flux = np.maximum(0, np.diff(spectra, axis=0)).sum(axis=1)
    raw_flux = np.pad(raw_flux, (1, 0))
    spectral_flux = normalize(raw_flux)

    # --- 2. Instrument change: rate of change of the band distribution ---
    band_stack = np.stack([low, mid, high], axis=1)  # (n, 3)
    band_norm = band_stack / (band_stack.sum(axis=1, keepdims=True) + 1e-9)
    band_delta = np.abs(np.diff(band_norm, axis=0)).sum(axis=1)
    band_delta = np.pad(band_delta, (1, 0))
    instrument_change = normalize(_smooth(band_delta, 5))

    # --- 3. Harmonic / percussive ratio ---
    flat = spectral_flatness(spectra)
    percussive = np.clip(0.5 * flat + 0.5 * spectral_flux, 0.0, 1.0)
    harmonic_ratio = _smooth(np.clip(1.0 - percussive, 0.0, 1.0), 5)

    # --- 4. Chord change: chroma novelty over a beat-scale window ---
    chroma = compute_chroma(spectra, freqs)
    smooth_win = max(1, int(0.35 / hop))
    kernel = np.ones(smooth_win) / smooth_win
    chroma_s = np.apply_along_axis(
        lambda c: np.convolve(c, kernel, mode="same"), 0, chroma)
    chroma_delta = np.abs(np.diff(chroma_s, axis=0)).sum(axis=1)
    chroma_delta = np.pad(chroma_delta, (1, 0))
    chord_change = normalize(chroma_delta)

    # --- 5. Vocal presence probability ---
    # Sustained mid-band energy, harmonic (not percussive), pitch in vocal band.
    melody_arr = np.asarray(melody, dtype=np.float32) if len(melody) == n else np.full(n, 0.5)
    vocal_band = ((melody_arr > 0.15) & (melody_arr < 0.85)).astype(np.float32)
    vocal_raw = (
        normalize(mid) * (0.35 + 0.65 * harmonic_ratio)
        * (0.4 + 0.6 * (1.0 - onset_arr))
        * (0.5 + 0.5 * vocal_band)
    )
    vocal_presence = np.clip(_smooth(vocal_raw, 7), 0.0, 1.0)

    # --- 6. Drum fill probability ---
    # Dense onsets + percussive character + broadband bursts (rolling window).
    onset_hits = onset_arr > 0.45
    fill_win = max(2, int(0.7 / hop))
    onset_density = _rolling_count(onset_hits, fill_win)
    drum_raw = (
        onset_density * (0.4 + 0.6 * percussive)
        * (0.5 + 0.5 * normalize(low + high))
    )
    drum_fill = np.clip(_smooth(drum_raw, 3), 0.0, 1.0)

    # --- 7. Tension: rising energy + dissonance + density + flux ---
    energy_s = _smooth(rms_arr, 9)
    energy_grad = np.maximum(0, np.diff(energy_s, prepend=energy_s[0]))
    energy_grad = normalize(_smooth(energy_grad, 9))
    chroma_entropy = _entropy(chroma)  # dissonant / unstable harmony
    tension_raw = (
        0.35 * normalize(rms_arr)
        + 0.22 * onset_density
        + 0.20 * chroma_entropy
        + 0.13 * energy_grad
        + 0.10 * spectral_flux
    )
    tension = np.clip(_smooth(tension_raw, 11), 0.0, 1.0)

    return FeatureVector(
        spectral_flux=spectral_flux.tolist(),
        instrument_change=instrument_change.tolist(),
        harmonic_ratio=harmonic_ratio.tolist(),
        chord_change=chord_change.tolist(),
        vocal_presence=vocal_presence.tolist(),
        drum_fill=drum_fill.tolist(),
        tension=tension.tolist(),
    )


# ---------------------------------------------------------------------------
# Focus instrument detection
# ---------------------------------------------------------------------------

def detect_focus_instruments(
    frame_times: list[float],
    onset: list[float],
    rms: list[float],
    bands: BandEnergy,
    duration: float,
    window_sec: float = 3.0,
) -> list[FocusSegment]:
    """Determine the dominant instrument in each time window.

    A professional chart maker listens to a song and decides moment by moment:
    "this part is vocal-driven", "now the guitar takes over", "drum fill here".
    This function automates that judgment by analyzing spectral band ratios,
    onset density, and energy sustain within overlapping windows.

    Classification rules:
      drums  — high onset density + strong low-band (kick) or high-band (cymbals)
      vocal  — sustained mid-band energy with low onset density (legato)
      guitar — mid+high band with variable onset (melodic but articulated)
      bass   — low-band dominant with moderate, steady onset
      keys   — sharp mid-band onsets without low-band dominance
      mixed  — no single instrument clearly dominates
    """
    if not frame_times or not onset:
        return [FocusSegment("mixed", 0.0, duration, 0.3)]

    hop = frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.023
    win_frames = max(1, int(window_sec / hop))
    n = len(frame_times)
    has_bands = bool(bands.low and bands.mid and bands.high)

    onset_arr = np.array(onset)
    rms_arr = np.array(rms) if rms else np.zeros(n)
    low_arr = np.array(bands.low) if has_bands else np.zeros(n)
    mid_arr = np.array(bands.mid) if has_bands else np.zeros(n)
    high_arr = np.array(bands.high) if has_bands else np.zeros(n)

    raw_segments: list[FocusSegment] = []

    for start_i in range(0, n, win_frames):
        end_i = min(start_i + win_frames, n)
        if end_i - start_i < max(1, win_frames // 4):
            break

        t_start = float(frame_times[start_i])
        t_end = float(frame_times[min(end_i - 1, n - 1)])

        chunk_onset = onset_arr[start_i:end_i]
        chunk_rms = rms_arr[start_i:end_i]
        chunk_low = low_arr[start_i:end_i]
        chunk_mid = mid_arr[start_i:end_i]
        chunk_high = high_arr[start_i:end_i]

        avg_onset = float(np.mean(chunk_onset))
        onset_std = float(np.std(chunk_onset))
        avg_rms = float(np.mean(chunk_rms))
        avg_low = float(np.mean(chunk_low))
        avg_mid = float(np.mean(chunk_mid))
        avg_high = float(np.mean(chunk_high))
        band_total = avg_low + avg_mid + avg_high + 1e-9

        low_ratio = avg_low / band_total
        mid_ratio = avg_mid / band_total
        high_ratio = avg_high / band_total

        onset_peaks = int(np.sum(chunk_onset > 0.5))
        onset_density = onset_peaks / max(1, len(chunk_onset))

        scores: dict[str, float] = {}

        # Drums: high onset density + low-band (kick) or high-band (cymbals)
        scores["drums"] = (
            onset_density * 2.5
            + avg_onset * 1.2
            + (low_ratio * 0.8 if low_ratio > 0.35 else 0.0)
            + (high_ratio * 0.6 if high_ratio > 0.3 else 0.0)
        )

        # Vocal: sustained mid, low onset density (legato singing)
        sustain = max(0.0, avg_rms - avg_onset * 0.5)
        scores["vocal"] = (
            mid_ratio * 2.0
            + sustain * 1.5
            + max(0.0, 0.5 - onset_density) * 1.8
            + (0.3 if avg_mid > 0.4 and avg_onset < 0.45 else 0.0)
        )

        # Guitar: mid+high with variable onset (picking/strumming)
        scores["guitar"] = (
            (mid_ratio + high_ratio) * 1.2
            + onset_std * 1.5
            + (0.5 if avg_mid > 0.3 and avg_high > 0.25 else 0.0)
            + avg_onset * 0.4
        )

        # Bass: low-band dominant, steady
        scores["bass"] = (
            low_ratio * 2.5
            + max(0.0, 0.4 - onset_std) * 1.0
            + (0.6 if low_ratio > 0.42 else 0.0)
        )

        # Keys: sharp mid onsets without low dominance
        scores["keys"] = (
            mid_ratio * 1.5
            + onset_density * 1.0
            + (0.4 if avg_onset > 0.35 and low_ratio < 0.3 else 0.0)
            + avg_mid * 0.5
        )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_inst, best_score = ranked[0]
        second_score = ranked[1][1]

        margin = best_score - second_score
        confidence = min(1.0, margin / max(best_score, 0.01))

        if confidence < 0.12:
            best_inst = "mixed"
            confidence = 0.3

        raw_segments.append(FocusSegment(
            instrument=best_inst,
            start=round(t_start, 3),
            end=round(t_end, 3),
            confidence=round(confidence, 3),
        ))

    if not raw_segments:
        return [FocusSegment("mixed", 0.0, duration, 0.3)]

    # Merge adjacent segments with the same instrument
    merged: list[FocusSegment] = [raw_segments[0]]
    for seg in raw_segments[1:]:
        if seg.instrument == merged[-1].instrument:
            merged[-1].end = seg.end
            merged[-1].confidence = round(
                (merged[-1].confidence + seg.confidence) / 2, 3)
        else:
            merged.append(seg)

    merged[-1].end = round(duration, 3)
    return merged


def focus_at(segments: list[FocusSegment], t: float) -> FocusSegment:
    """Look up the focus instrument at time *t*."""
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg
    return segments[-1] if segments else FocusSegment("mixed", 0.0, 0.0, 0.3)


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze(wav_path: Path) -> AnalysisResult:
    audio, raw_sr = load_audio(wav_path)
    duration = len(audio) / raw_sr
    ds_audio, ds_sr = downsample(audio, raw_sr)

    frame_size = 2048
    hop_size = 512
    frames, frame_times = frame_audio(ds_audio, ds_sr, frame_size, hop_size)
    window = np.hanning(frame_size)

    rms = np.sqrt(np.mean((frames * window) ** 2, axis=1))
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    flux = np.maximum(0, np.diff(spectra, axis=0)).sum(axis=1)
    flux = np.pad(flux, (1, 0))
    onset = normalize(flux) * 0.75 + normalize(np.maximum(0, np.diff(rms, prepend=rms[0]))) * 0.25

    hop = float(frame_times[1] - frame_times[0]) if len(frame_times) > 1 else 0.023
    bpm, beat_offset = estimate_tempo(frame_times, onset, hop)

    tempo_map = build_tempo_map(frame_times, onset, hop, duration, bpm)
    beats = compute_beats_from_tempo_map(tempo_map, beat_offset, duration)
    measures = compute_measures(beats)
    rms_norm = normalize(rms)

    freqs = np.fft.rfftfreq(frame_size, 1.0 / ds_sr)
    melody = compute_melody(spectra, freqs)
    bands = separate_bands(frames, ds_sr, frame_size)
    sections = detect_structure(
        frame_times.tolist(), rms_norm.tolist(), duration,
        onset=onset.tolist(), high_band=bands.high,
    )
    focus_segments = detect_focus_instruments(
        frame_times.tolist(), onset.tolist(), rms_norm.tolist(),
        bands, duration,
    )
    features = compute_feature_vectors(
        spectra, freqs, onset, rms_norm, melody, bands, hop,
    )

    return AnalysisResult(
        wav_path=str(wav_path),
        sample_rate=raw_sr,
        duration=round(duration, 3),
        bpm=bpm,
        beat_offset=beat_offset,
        beats=beats,
        measures=measures,
        frame_hop=hop,
        frame_times=frame_times.tolist(),
        onset_strength=onset.tolist(),
        rms=rms_norm.tolist(),
        tempo_map=tempo_map,
        melody=melody.tolist(),
        sections=sections,
        bands=bands,
        focus_segments=focus_segments,
        features=features,
    )
