import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { AdSlot } from './Ads'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NoteData {
  time: number
  lane: number
  kind: string
  duration: number
  beat: number
  weight: number
}

interface SectionData {
  label: string
  start: number
  end: number
  energy: number
}

interface TempoPointData {
  time: number
  bpm: number
}

interface EvalScores {
  overall: number
  rhythm_fit: number
  hand_movement: number
  pattern_diversity: number
  repetition: number
  long_note: number
  difficulty_fit: number
}

interface DensityPoint {
  time: number
  intensity: number
}

interface ActualDensityPoint {
  time: number
  nps: number
}

interface ChartMeta {
  bpm: number
  beat_offset: number
  duration: number
  keys: number
  difficulty: string
  note_count: number
  tap_count: number
  hold_count: number
  tempo_map?: TempoPointData[]
  sections: SectionData[]
  lane_colors: string[]
  job_id: string
  eval?: EvalScores
  density_curve?: DensityPoint[]
  actual_density?: ActualDensityPoint[]
  song?: { id: string; title: string; artist: string; youtubeId: string }
}

interface ChartData {
  metadata: ChartMeta
  notes: NoteData[]
}

type AppView = 'input' | 'loading' | 'editor'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LANE_W = 56
const NOTE_H = 14
const HEADER_H = 40
const PX_PER_SEC = 200
const JUDGE_Y = 80
const DENSITY_GRAPH_W = 100
const DENSITY_MARGIN = 40

function bpmAt(tempoMap: TempoPointData[] | undefined, t: number, fallback: number): number {
  if (!tempoMap || tempoMap.length === 0) return fallback
  let bpm = tempoMap[0].bpm
  for (const tp of tempoMap) {
    if (tp.time <= t) bpm = tp.bpm
    else break
  }
  return bpm
}

function bpmLabel(tempoMap: TempoPointData[] | undefined, fallback: number): string {
  if (!tempoMap || tempoMap.length <= 1) return `${fallback} BPM`
  const bpms = tempoMap.map((tp) => tp.bpm)
  const lo = Math.min(...bpms)
  const hi = Math.max(...bpms)
  if (Math.abs(hi - lo) < 1) return `${fallback} BPM`
  return `${lo.toFixed(0)}–${hi.toFixed(0)} BPM`
}

const SECTION_COLORS: Record<string, string> = {
  intro: 'rgba(100,200,255,0.08)',
  verse: 'rgba(100,255,100,0.08)',
  chorus: 'rgba(255,100,100,0.10)',
  bridge: 'rgba(255,200,100,0.08)',
  outro: 'rgba(180,100,255,0.08)',
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [view, setView] = useState<AppView>('input')
  const [url, setUrl] = useState('')
  const [keys, setKeys] = useState(4)
  const [difficulty, setDifficulty] = useState('hard')
  const [style, setStyle] = useState('auto')
  const [loading, setLoading] = useState('')
  const [error, setError] = useState('')
  const [chart, setChart] = useState<ChartData | null>(null)
  const [jobId, setJobId] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleResult(res: Response) {
    if (!res.ok) {
      const b = await res.json().catch(() => ({ detail: '서버 오류' }))
      throw new Error(b.detail)
    }
    const data: ChartData = await res.json()
    setChart(data)
    setJobId(data.metadata.job_id)
    setView('editor')
  }

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault()
    if (!/https?:\/\/.*(youtube\.com|youtu\.be)/.test(url.trim())) {
      setError('유효한 YouTube URL을 입력해주세요.')
      return
    }
    setView('loading')
    setLoading('YouTube에서 다운로드 중...')
    setError('')
    try {
      const res = await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), keys, difficulty, style, seed: Date.now() % 100000 }),
      })
      await handleResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류')
      setView('input')
    }
  }

  async function uploadFile(file: File) {
    setView('loading')
    setLoading(`"${file.name}" 분석 중...`)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('keys', String(keys))
      form.append('difficulty', difficulty)
      form.append('style', style)
      form.append('seed', String(Date.now() % 100000))
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      await handleResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류')
      setView('input')
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  async function handleRegenerate() {
    if (!jobId) return
    setLoading('채보 재생성 중...')
    try {
      const res = await fetch(`/api/regenerate/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys, difficulty, style, seed: Date.now() % 100000 }),
      })
      if (!res.ok) throw new Error('재생성 실패')
      const data: ChartData = await res.json()
      setChart(data)
      setLoading('')
    } catch {
      setLoading('')
    }
  }

  async function handleExport() {
    if (!chart) return
    // Convert the current (edited) chart and download it to the user's
    // Downloads folder. Nothing is stored on the server.
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chart }),
    })
    if (!res.ok) return
    const blob = await res.blob()
    const title = (chart.metadata.song?.title || 'chart').replace(/[^\w \-]/g, '').trim() || 'chart'
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${title}.ryuthm.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  if (view === 'loading') {
    return (
      <div className="app center-col">
        <div className="spinner" />
        <p className="loading-text">{loading}</p>
      </div>
    )
  }

  if (view === 'editor' && chart) {
    return (
      <ChartEditor
        chart={chart}
        jobId={jobId}
        keys={keys}
        difficulty={difficulty}
        style={style}
        onKeysChange={(k) => { setKeys(k) }}
        onDifficultyChange={(d) => { setDifficulty(d) }}
        onStyleChange={(s) => { setStyle(s) }}
        onRegenerate={handleRegenerate}
        onExport={handleExport}
        onChartUpdate={setChart}
        onBack={() => setView('input')}
      />
    )
  }

  return (
    <div
      className={`app center-col ${dragOver ? 'drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <header className="header">
        <h1>Rhythm Chart Generator</h1>
        <p className="subtitle">오디오 파일을 업로드하거나 YouTube 링크를 입력하세요</p>
      </header>

      <div className="upload-area" onClick={() => fileRef.current?.click()}>
        <input
          ref={fileRef} type="file" hidden
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
          onChange={handleFileSelect}
        />
        <p className="upload-icon">🎵</p>
        <p className="upload-text">클릭하거나 파일을 드래그하세요</p>
        <p className="upload-hint">WAV, MP3, OGG, FLAC, M4A</p>
      </div>

      <div className="divider"><span>또는</span></div>

      <form className="url-form" onSubmit={handleGenerate}>
        <input
          className="url-input"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => { setUrl(e.target.value); setError('') }}
        />
        <button type="submit" className="submit-btn">URL 생성</button>
      </form>

      <div className="form-options">
        <label>
          키
          <select value={keys} onChange={(e) => setKeys(+e.target.value)}>
            {[4, 5, 6, 7, 8].map((k) => <option key={k} value={k}>{k}K</option>)}
          </select>
        </label>
        <label>
          난이도
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            {['easy', 'normal', 'hard', 'expert', 'master'].map((d) => (
              <option key={d} value={d}>{d.toUpperCase()}</option>
            ))}
          </select>
        </label>
        <label>
          스타일
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            <option value="auto">AUTO (밸런스)</option>
            <option value="djmax">DJMAX (악기·드럼)</option>
            <option value="sekai">프로젝트 세카이 (보컬)</option>
          </select>
        </label>
      </div>

      {error && <p className="error">{error}</p>}

      <AdSlot className="ad-landing" style={{ marginTop: 24, minHeight: 90 }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Chart Editor (Steps 16-18)
// ---------------------------------------------------------------------------

interface EditorProps {
  chart: ChartData
  jobId: string
  keys: number
  difficulty: string
  style: string
  onKeysChange: (k: number) => void
  onDifficultyChange: (d: string) => void
  onStyleChange: (s: string) => void
  onRegenerate: () => void
  onExport: () => void
  onChartUpdate: (c: ChartData) => void
  onBack: () => void
}

function ChartEditor({
  chart, jobId, keys, difficulty, style,
  onKeysChange, onDifficultyChange, onStyleChange, onRegenerate, onExport, onChartUpdate, onBack,
}: EditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrent] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [tool, setTool] = useState<'select' | 'add' | 'delete'>('select')
  const [dragNote, setDragNote] = useState<number | null>(null)
  const animRef = useRef(0)

  const pxPerSec = PX_PER_SEC * zoom
  const laneW = LANE_W
  const totalW = keys * laneW
  const totalH = Math.max(800, chart.metadata.duration * pxPerSec + 200)
  const colors = chart.metadata.lane_colors

  const timeToY = useCallback(
    (t: number) => totalH - JUDGE_Y - t * pxPerSec,
    [totalH, pxPerSec],
  )

  // Audio setup
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.src = `/api/audio/${jobId}`
    audio.load()
  }, [jobId])

  // Animation loop
  useEffect(() => {
    const tick = () => {
      const audio = audioRef.current
      if (audio && !audio.paused) {
        setCurrent(audio.currentTime)
      }
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [])

  // Auto-scroll to follow playhead
  useEffect(() => {
    const container = containerRef.current
    if (!container || !playing) return
    const y = timeToY(currentTime)
    const viewH = container.clientHeight
    const targetScroll = y - viewH / 2
    container.scrollTop = Math.max(0, targetScroll)
  }, [currentTime, playing, timeToY])

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = totalW + DENSITY_MARGIN + DENSITY_GRAPH_W + 20
    canvas.height = totalH

    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Section backgrounds
    for (const sec of chart.metadata.sections) {
      const y1 = timeToY(sec.end)
      const y2 = timeToY(sec.start)
      ctx.fillStyle = SECTION_COLORS[sec.label] || 'rgba(128,128,128,0.05)'
      ctx.fillRect(0, y1, totalW, y2 - y1)
      ctx.fillStyle = 'rgba(255,255,255,0.3)'
      ctx.font = '11px sans-serif'
      ctx.fillText(sec.label.toUpperCase(), 4, y1 + 14)
    }

    // Lane lines
    for (let i = 0; i <= keys; i++) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'
      ctx.beginPath()
      ctx.moveTo(i * laneW, 0)
      ctx.lineTo(i * laneW, totalH)
      ctx.stroke()
    }

    // Beat lines (tempo-map-aware)
    const tmap = chart.metadata.tempo_map
    let bt = chart.metadata.beat_offset
    let beatIdx = 0
    while (bt < chart.metadata.duration) {
      const y = timeToY(bt)
      const isMeasure = beatIdx % 4 === 0
      ctx.strokeStyle = isMeasure ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.07)'
      ctx.lineWidth = isMeasure ? 1.5 : 0.5
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(totalW, y)
      ctx.stroke()
      if (isMeasure) {
        ctx.fillStyle = 'rgba(255,255,255,0.25)'
        ctx.font = '10px monospace'
        ctx.fillText(`${beatIdx / 4 + 1}`, totalW + 4, y + 4)
      }
      const localBpm = bpmAt(tmap, bt, chart.metadata.bpm)
      bt += 60 / localBpm
      beatIdx++
    }
    ctx.lineWidth = 1

    // Notes
    for (const note of chart.notes) {
      const x = note.lane * laneW + 2
      const y = timeToY(note.time) - NOTE_H / 2
      const w = laneW - 4
      const color = colors[note.lane] || '#fff'

      if (note.kind === 'hold' && note.duration > 0) {
        const holdH = note.duration * pxPerSec
        ctx.fillStyle = color + '44'
        ctx.fillRect(x, y - holdH + NOTE_H / 2, w, holdH)
        ctx.fillStyle = color + 'aa'
        ctx.fillRect(x, y - holdH + NOTE_H / 2, w, 4)
      }

      ctx.fillStyle = color
      ctx.shadowColor = color
      ctx.shadowBlur = 6
      ctx.beginPath()
      ctx.roundRect(x, y, w, NOTE_H, 3)
      ctx.fill()
      ctx.shadowBlur = 0
    }

    // Playhead
    const playY = timeToY(currentTime)
    ctx.strokeStyle = '#ff3366'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, playY)
    ctx.lineTo(totalW, playY)
    ctx.stroke()
    ctx.lineWidth = 1

    // Time label
    ctx.fillStyle = '#ff3366'
    ctx.font = 'bold 12px monospace'
    const min = Math.floor(currentTime / 60)
    const sec = Math.floor(currentTime % 60)
    ctx.fillText(`${min}:${sec.toString().padStart(2, '0')}`, totalW + 4, playY - 4)

    // --- Density graph (right side) ---
    const graphX = totalW + DENSITY_MARGIN
    const graphW = DENSITY_GRAPH_W
    const dc = chart.metadata.density_curve
    const ad = chart.metadata.actual_density

    if (dc && dc.length >= 2) {
      // Background
      ctx.fillStyle = 'rgba(0,0,0,0.3)'
      ctx.fillRect(graphX - 2, 0, graphW + 4, totalH)

      // Separator line
      ctx.strokeStyle = 'rgba(255,255,255,0.15)'
      ctx.beginPath()
      ctx.moveTo(graphX - 2, 0)
      ctx.lineTo(graphX - 2, totalH)
      ctx.stroke()

      // Label
      ctx.fillStyle = 'rgba(255,255,255,0.5)'
      ctx.font = '10px sans-serif'
      ctx.fillText('DENSITY', graphX + 2, 14)

      // Find max intensity for scaling
      const maxIntensity = Math.max(1.0, ...dc.map(p => p.intensity))

      // Draw target intensity curve (filled area)
      ctx.beginPath()
      ctx.moveTo(graphX, timeToY(dc[0].time))
      for (const p of dc) {
        const px = graphX + (p.intensity / maxIntensity) * graphW
        const py = timeToY(p.time)
        ctx.lineTo(px, py)
      }
      ctx.lineTo(graphX, timeToY(dc[dc.length - 1].time))
      ctx.closePath()
      ctx.fillStyle = 'rgba(74,158,255,0.15)'
      ctx.fill()

      // Draw target curve line
      ctx.beginPath()
      for (let i = 0; i < dc.length; i++) {
        const px = graphX + (dc[i].intensity / maxIntensity) * graphW
        const py = timeToY(dc[i].time)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.strokeStyle = 'rgba(74,158,255,0.6)'
      ctx.lineWidth = 1.5
      ctx.stroke()
      ctx.lineWidth = 1

      // Draw actual note density as bars
      if (ad && ad.length > 0) {
        const maxNps = Math.max(1, ...ad.map(p => p.nps))
        for (const p of ad) {
          const py = timeToY(p.time)
          const barW = (p.nps / maxNps) * graphW
          const barH = Math.max(2, pxPerSec * 0.8)
          const intensity = p.nps / maxNps
          const r = Math.round(74 + intensity * 181)
          const g = Math.round(158 - intensity * 100)
          const b = Math.round(255 - intensity * 200)
          ctx.fillStyle = `rgba(${r},${g},${b},0.5)`
          ctx.fillRect(graphX, py - barH / 2, barW, barH)
        }
      }

      // Star markers at section boundaries
      ctx.font = '9px sans-serif'
      for (const sec of chart.metadata.sections) {
        const sy = timeToY(sec.start)
        // Find intensity at section start
        let intAtStart = 0.5
        for (let i = 0; i < dc.length - 1; i++) {
          if (dc[i].time <= sec.start && dc[i + 1].time >= sec.start) {
            const frac = (sec.start - dc[i].time) / Math.max(0.001, dc[i + 1].time - dc[i].time)
            intAtStart = dc[i].intensity + frac * (dc[i + 1].intensity - dc[i].intensity)
            break
          }
        }
        const stars = Math.max(1, Math.min(10, Math.round(intAtStart * 10)))
        const starStr = '★'.repeat(stars)
        ctx.fillStyle = 'rgba(255,204,0,0.7)'
        ctx.fillText(starStr, graphX + 2, sy + 4)
      }

      // Playhead on density graph
      ctx.strokeStyle = '#ff3366'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(graphX - 2, playY)
      ctx.lineTo(graphX + graphW + 2, playY)
      ctx.stroke()
      ctx.lineWidth = 1
    }
  }, [chart, currentTime, keys, totalW, totalH, pxPerSec, timeToY, colors, laneW])

  function togglePlay() {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      audio.play()
      setPlaying(true)
    } else {
      audio.pause()
      setPlaying(false)
    }
  }

  function handleCanvasClick(e: React.MouseEvent) {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const lane = Math.floor(x / laneW)
    if (lane < 0 || lane >= keys) return
    const time = (totalH - JUDGE_Y - y) / pxPerSec

    if (tool === 'add') {
      const localBpm = bpmAt(chart.metadata.tempo_map, time, chart.metadata.bpm)
      const beatLen = 60 / localBpm
      const snapped = Math.round((time - chart.metadata.beat_offset) / (beatLen / 4)) * (beatLen / 4) + chart.metadata.beat_offset
      const newNote: NoteData = { time: +snapped.toFixed(4), lane, kind: 'tap', duration: 0, beat: 0, weight: 0.5 }
      const newNotes = [...chart.notes, newNote].sort((a, b) => a.time - b.time || a.lane - b.lane)
      onChartUpdate({ ...chart, notes: newNotes, metadata: { ...chart.metadata, note_count: newNotes.length } })
    } else if (tool === 'delete') {
      const idx = chart.notes.findIndex((n) => {
        const ny = timeToY(n.time)
        const nx = n.lane * laneW
        return Math.abs(y - ny) < NOTE_H && x >= nx && x < nx + laneW
      })
      if (idx >= 0) {
        const newNotes = chart.notes.filter((_, i) => i !== idx)
        onChartUpdate({ ...chart, notes: newNotes, metadata: { ...chart.metadata, note_count: newNotes.length } })
      }
    } else {
      // Select mode: seek audio
      const audio = audioRef.current
      if (audio && time >= 0 && time <= chart.metadata.duration) {
        audio.currentTime = time
        setCurrent(time)
      }
    }
  }

  // --- Sync adjustment ----------------------------------------------------

  // Shift every note + the grid by a constant offset (first-beat alignment).
  function shiftOffset(deltaMs: number) {
    const d = deltaMs / 1000
    const notes = chart.notes.map((n) => ({ ...n, time: Math.max(0, +(n.time + d).toFixed(4)) }))
    onChartUpdate({
      ...chart,
      notes,
      metadata: { ...chart.metadata, beat_offset: +(chart.metadata.beat_offset + d).toFixed(4) },
    })
  }

  // Fine BPM change: rescale note spacing around the offset anchor. Fixes the
  // late-song drift that a slightly-wrong tempo causes.
  function nudgeBpm(delta: number) {
    const oldBpm = chart.metadata.bpm
    const newBpm = +(oldBpm + delta).toFixed(3)
    if (newBpm <= 0) return
    const off = chart.metadata.beat_offset
    const factor = oldBpm / newBpm
    const notes = chart.notes.map((n) => ({
      ...n,
      time: +(off + (n.time - off) * factor).toFixed(4),
      duration: +(n.duration * factor).toFixed(4),
    }))
    onChartUpdate({ ...chart, notes, metadata: { ...chart.metadata, bpm: newBpm } })
  }

  // Half/double tempo: detector octave error. Notes sit on real onsets, so we
  // only relabel the BPM (and thus the grid), leaving note times untouched.
  function octaveBpm(mult: number) {
    const newBpm = +(chart.metadata.bpm * mult).toFixed(3)
    if (newBpm < 40 || newBpm > 400) return
    onChartUpdate({ ...chart, metadata: { ...chart.metadata, bpm: newBpm } })
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.code === 'Space') { e.preventDefault(); togglePlay() }
    if (e.code === 'KeyA') setTool('add')
    if (e.code === 'KeyD') setTool('delete')
    if (e.code === 'KeyS') setTool('select')
    if (e.code === 'Equal') setZoom((z) => Math.min(z + 0.25, 4))
    if (e.code === 'Minus') setZoom((z) => Math.max(z - 0.25, 0.25))
    if (e.code === 'BracketLeft') shiftOffset(-5)
    if (e.code === 'BracketRight') shiftOffset(5)
  }

  return (
    <div className="editor" tabIndex={0} onKeyDown={handleKeyDown}>
      <audio ref={audioRef} preload="auto" />

      <div className="editor-toolbar">
        <button className="back-btn" onClick={onBack}>← 뒤로</button>
        <div className="toolbar-group">
          <span className="info">{bpmLabel(chart.metadata.tempo_map, chart.metadata.bpm)}</span>
          <span className="info">{chart.metadata.note_count}노트</span>
          <span className="info">{chart.metadata.duration.toFixed(1)}초</span>
          {chart.metadata.eval && (
            <span className="info" title={`리듬=${chart.metadata.eval.rhythm_fit?.toFixed(0)} 손이동=${chart.metadata.eval.hand_movement?.toFixed(0)} 패턴=${chart.metadata.eval.pattern_diversity?.toFixed(0)} 반복=${chart.metadata.eval.repetition?.toFixed(0)} 롱노트=${chart.metadata.eval.long_note?.toFixed(0)} 난이도=${chart.metadata.eval.difficulty_fit?.toFixed(0)}`}
              style={{ color: chart.metadata.eval.overall >= 80 ? '#4caf50' : chart.metadata.eval.overall >= 65 ? '#ff9800' : '#f44336', fontWeight: 'bold' }}>
              품질 {chart.metadata.eval.overall?.toFixed(0)}점
            </span>
          )}
        </div>
        <div className="toolbar-group">
          <label>
            <select value={keys} onChange={(e) => onKeysChange(+e.target.value)}>
              {[4, 5, 6, 7, 8].map((k) => <option key={k} value={k}>{k}K</option>)}
            </select>
          </label>
          <label>
            <select value={difficulty} onChange={(e) => onDifficultyChange(e.target.value)}>
              {['easy', 'normal', 'hard', 'expert', 'master'].map((d) => (
                <option key={d} value={d}>{d.toUpperCase()}</option>
              ))}
            </select>
          </label>
          <label>
            <select value={style} onChange={(e) => onStyleChange(e.target.value)}>
              <option value="auto">AUTO</option>
              <option value="djmax">DJMAX</option>
              <option value="sekai">SEKAI</option>
            </select>
          </label>
          <button onClick={onRegenerate}>재생성</button>
          <button onClick={onExport}>내보내기</button>
        </div>
        <div className="toolbar-group sync-group" title="첫 박이 안 맞으면 오프셋, 후반부가 밀리면 BPM을 조정하세요">
          <span className="sync-label">싱크</span>
          <button onClick={() => shiftOffset(-5)} title="노트를 5ms 앞으로 ([)">◀5</button>
          <button onClick={() => shiftOffset(-1)} title="1ms 앞으로">◀1</button>
          <span className="info">{Math.round(chart.metadata.beat_offset * 1000)}ms</span>
          <button onClick={() => shiftOffset(1)} title="1ms 뒤로">1▶</button>
          <button onClick={() => shiftOffset(5)} title="노트를 5ms 뒤로 (])">5▶</button>
          <span className="sync-sep">|</span>
          <button onClick={() => octaveBpm(0.5)} title="절반 템포로 (오검출 보정)">½</button>
          <button onClick={() => nudgeBpm(-0.1)} title="BPM −0.1 (드리프트 보정)">−</button>
          <button onClick={() => octaveBpm(2)} title="2배 템포로 (오검출 보정)">×2</button>
          <button onClick={() => nudgeBpm(0.1)} title="BPM +0.1 (드리프트 보정)">+</button>
        </div>
        <div className="toolbar-group">
          <button className={tool === 'select' ? 'active' : ''} onClick={() => setTool('select')}>선택 (S)</button>
          <button className={tool === 'add' ? 'active' : ''} onClick={() => setTool('add')}>추가 (A)</button>
          <button className={tool === 'delete' ? 'active' : ''} onClick={() => setTool('delete')}>삭제 (D)</button>
        </div>
        <div className="toolbar-group">
          <button onClick={() => setZoom((z) => Math.max(z - 0.25, 0.25))}>−</button>
          <span className="info">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}>+</button>
          <button className="play-btn" onClick={togglePlay}>{playing ? '⏸' : '▶'}</button>
        </div>
      </div>

      <div className="editor-body" ref={containerRef}>
        <canvas
          ref={canvasRef}
          className="chart-canvas"
          style={{ width: totalW + DENSITY_MARGIN + DENSITY_GRAPH_W + 20, height: totalH }}
          onClick={handleCanvasClick}
        />
      </div>
    </div>
  )
}

export default App
