import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, Check, ChevronRight,
  Clock3, Eye, FileVideo, Layers3, LockKeyhole, Pause, Play,
  Plus, RotateCcw, ShieldCheck, Sparkles, Trash2, UploadCloud, X,
} from 'lucide-react'
import { demoCases } from './demoData'
import type { AnalysisResult, AnalysisStatus, Probability, VideoInput } from './types'

const MAX_BYTES = 80 * 1024 * 1024
const MAX_SECONDS = 5.15
const ACCEPTED = ['video/mp4', 'video/quicktime', 'video/webm']
const CLOUD_MODE = import.meta.env.VITE_INFERENCE_MODE === 'cloud'
const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')
const apiUrl = (path: string) => `${API_BASE}${path}`

async function sha256(file: File) {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')
}

async function responsePayload(response: Response) {
  const payload = await response.json().catch(() => ({ error: `Request failed (${response.status}).` }))
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`)
  return payload
}

async function cloudInference(videos: VideoInput[], accessCode: string, signal: AbortSignal): Promise<AnalysisResult> {
  const files = await Promise.all(videos.map(async ({ file }) => ({
    name: file.name, size: file.size, type: file.type, sha256: await sha256(file),
  })))
  const session = await responsePayload(await fetch(apiUrl('/api/session'), {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ files, demoCode: accessCode }), signal,
  })) as { id: string; secret: string }
  for (let index = 0; index < videos.length; index += 1) {
    await responsePayload(await fetch(apiUrl(`/api/upload/${session.id}/${index}`), {
      method: 'PUT', headers: { 'content-type': videos[index].file.type, 'X-Upload-Secret': session.secret },
      body: videos[index].file, signal,
    }))
  }
  return responsePayload(await fetch(apiUrl(`/api/infer/${session.id}`), {
    method: 'POST', headers: { 'X-Upload-Secret': session.secret }, signal,
  })) as Promise<AnalysisResult>
}

function Brand() {
  return <div className="brand"><span className="brand-mark"><span /></span><span>REFEREE<span className="brand-accent">AI</span></span></div>
}

function Header() {
  return <header className="header"><Brand /><nav><a href="#workspace">Analyse</a><a href="#method">Method</a><a href="#limitations">Limitations</a><a href="mailto:MikeCharlotteMao@outlook.com">Contact</a></nav><div className="research-pill"><span /> Research prototype</div></header>
}

function Hero() {
  return <section className="hero">
    <div className="eyebrow"><Sparkles size={14} /> Multi-view decision support</div>
    <h1>See the incident.<br /><em>From every angle.</em></h1>
    <p>Upload synchronized match clips and explore how a research model interprets foul type, severity and visual evidence across time and camera views.</p>
    <div className="hero-byline"><span>A research prototype by</span><strong>Lecheng Mao</strong><a href="mailto:MikeCharlotteMao@outlook.com">Get in touch</a></div>
    <a className="hero-cta" href="#workspace">Start an analysis <ArrowRight size={17} /></a>
    <div className="hero-stats"><div><strong>1–4</strong><span>camera views</span></div><div><strong>≤ 5s</strong><span>per clip</span></div><div><strong>5</strong><span>temporal windows</span></div></div>
  </section>
}

function UploadSlot({ input, index, onRemove, onPick }: { input?: VideoInput; index: number; onRemove: () => void; onPick: (files: FileList | null) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  if (input) return <article className={`video-slot filled ${input.status}`}>
    <video src={input.url} muted playsInline />
    <div className="video-shade" />
    <span className="view-label">VIEW {index + 1}</span>
    <button className="remove-button" onClick={onRemove} aria-label={`Remove view ${index + 1}`}><X size={15} /></button>
    <div className="video-meta"><FileVideo size={15} /><div><strong>{input.file.name}</strong><span>{input.duration ? `${input.duration.toFixed(1)} sec` : 'Checking clip…'} · {(input.file.size / 1024 / 1024).toFixed(1)} MB</span></div></div>
    {input.error && <div className="video-error">{input.error}</div>}
  </article>
  return <button className="video-slot empty" onClick={() => ref.current?.click()}>
    <input ref={ref} type="file" hidden accept="video/mp4,video/quicktime,video/webm" multiple onChange={e => onPick(e.target.files)} />
    <span className="plus"><Plus size={19} /></span><strong>{index === 0 ? 'Add first view' : 'Add another view'}</strong><small>MP4, MOV or WebM</small>
  </button>
}

function ProbabilityList({ values, tone }: { values: Probability[]; tone: 'green' | 'amber' }) {
  return <div className="prob-list">{values.map((item, i) => <div className="prob-row" key={item.label}>
    <div className="prob-label"><span>{i + 1}</span>{item.label}<strong>{Math.round(item.value * 100)}%</strong></div>
    <div className="prob-track"><i className={tone} style={{ width: `${item.value * 100}%` }} /></div>
  </div>)}</div>
}

function PredictionCard({ title, icon, values, tone }: { title: string; icon: React.ReactNode; values: Probability[]; tone: 'green' | 'amber' }) {
  const top = values[0]
  return <article className="prediction-card">
    <div className="card-heading"><span className={`icon-box ${tone}`}>{icon}</span><div><small>{title}</small><h3>{top.label}</h3></div><div className={`confidence ${tone}`}>{Math.round(top.value * 100)}<small>%</small></div></div>
    <ProbabilityList values={values} tone={tone} />
  </article>
}

function Evidence({ result }: { result: AnalysisResult }) {
  return <div className="evidence-grid view-only">
    <article className="panel view-panel"><div className="panel-title"><div><small>CAMERA CONTRIBUTION</small><h3>Evidence by view</h3></div><Eye size={18} /></div>
      <div className="view-bars">{result.viewWeights.map((w, i) => <div key={i}><span>View {i + 1}</span><div><i style={{ width: `${w * 100}%` }} /></div><strong>{Math.round(w * 100)}%</strong></div>)}</div>
    </article>
  </div>
}

function Results({ result, onReset }: { result: AnalysisResult; onReset: () => void }) {
  return <section className="results">
    <div className="section-heading"><div><span className="section-number">02</span><div><small>ANALYSIS COMPLETE</small><h2>{result.title}</h2><p>{result.subtitle}</p></div></div><button className="secondary-button" onClick={onReset}><RotateCcw size={15} /> New analysis</button></div>
    <div className="confidence-note"><span className={`status-dot ${result.confidence}`} /><strong>{result.confidence[0].toUpperCase() + result.confidence.slice(1)} confidence</strong><p>{result.note}</p></div>
    <div className="prediction-grid"><PredictionCard title="MOST LIKELY FOUL TYPE" icon={<Activity size={18} />} values={result.foul} tone="green" /><PredictionCard title="MOST LIKELY SEVERITY" icon={<ShieldCheck size={18} />} values={result.severity} tone="amber" /></div>
    <Evidence result={result} />
    <div className="red-warning"><AlertTriangle size={18} /><div><strong>Limited support for red-card decisions</strong><p>Only 27 red-card incidents were available during training. Red-card output must not be treated as a reliable officiating recommendation.</p></div></div>
  </section>
}

function DemoCases({ selected, onSelect }: { selected: AnalysisResult | null; onSelect: (item: AnalysisResult) => void }) {
  return <section className="demo-section"><div className="demo-intro"><small>NO UPLOAD NEEDED</small><h2>Choose a real incident</h2><p>Open a prepared multi-camera case, review every angle, then ask the model to analyse it.</p></div><div className="demo-cards">{demoCases.map((item, i) => <button key={item.id} onClick={() => onSelect(item)} className={`demo-card ${selected?.id === item.id ? 'selected' : ''}`}>
    <video src={item.videos?.[0]} muted playsInline preload="metadata" />
    <span className="demo-index">0{i + 1}</span><span className="demo-view-count"><Eye size={13} /> {item.videos?.length} views</span>
    <div className="demo-card-copy"><small>PREPARED MULTI-VIEW CASE</small><strong>{item.title}</strong><span>Open incident <ChevronRight size={15} /></span></div>
  </button>)}</div></section>
}

function PreparedIncident({ item, analyzing, onAnalyse, onClose }: { item: AnalysisResult; analyzing: boolean; onAnalyse: () => void; onClose: () => void }) {
  const videosRef = useRef<HTMLDivElement>(null)
  const controlAll = (action: 'play' | 'pause') => {
    videosRef.current?.querySelectorAll('video').forEach(video => action === 'play' ? void video.play() : video.pause())
  }
  return <section className="prepared-review" id="prepared-review">
    <div className="prepared-header"><div><small>PREPARED INCIDENT · NO RESULT SHOWN YET</small><h2>{item.title}</h2><p>{item.subtitle}. Review the footage before requesting the stored model analysis.</p></div><button className="close-review" onClick={onClose}><X size={16} /> Close</button></div>
    <div ref={videosRef} className={`prepared-videos views-${item.videos?.length}`}>{item.videos?.map((src, i) => <article key={src}><div className="prepared-video-label">VIEW {i + 1}</div><video src={src} controls muted playsInline preload="metadata" /></article>)}</div>
    <div className="prepared-controls"><div className="sync-controls"><button onClick={() => controlAll('play')}><Play size={14} /> Play all</button><button onClick={() => controlAll('pause')}><Pause size={14} /> Pause all</button><span>Clips show the same incident from synchronized cameras.</span></div><button className="analyse-button" onClick={onAnalyse} disabled={analyzing}>{analyzing ? <><span className="spinner" /> Analysing prepared case…</> : <>Analyse this incident <ArrowRight size={17} /></>}</button></div>
  </section>
}

function Workspace() {
  const [videos, setVideos] = useState<VideoInput[]>([])
  const [status, setStatus] = useState<AnalysisStatus>('idle')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [selectedDemo, setSelectedDemo] = useState<AnalysisResult | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [accessCode, setAccessCode] = useState('')
  const allValid = videos.length > 0 && videos.every(v => v.status === 'valid')
  useEffect(() => () => videos.forEach(v => URL.revokeObjectURL(v.url)), [])

  const validateFile = (file: File, id: string, url: string) => {
    if (!ACCEPTED.includes(file.type)) return setVideos(old => old.map(v => v.id === id ? { ...v, status: 'invalid', error: 'Unsupported format' } : v))
    if (file.size > MAX_BYTES) return setVideos(old => old.map(v => v.id === id ? { ...v, status: 'invalid', error: 'File exceeds 80 MB' } : v))
    const media = document.createElement('video')
    media.preload = 'metadata'; media.src = url
    media.onloadedmetadata = () => setVideos(old => old.map(v => v.id === id ? media.duration <= MAX_SECONDS ? { ...v, duration: media.duration, status: 'valid' } : { ...v, duration: media.duration, status: 'invalid', error: 'Clip must be 5 seconds or shorter' } : v))
    media.onerror = () => setVideos(old => old.map(v => v.id === id ? { ...v, status: 'invalid', error: 'Could not read video metadata' } : v))
  }
  const addFiles = (files: FileList | null) => {
    if (!files) return
    const room = 4 - videos.length
    Array.from(files).slice(0, room).forEach(file => {
      const id = crypto.randomUUID(), url = URL.createObjectURL(file)
      setVideos(old => [...old, { id, file, url, status: 'checking' }])
      validateFile(file, id, url)
    })
    setResult(null); setApiError(null); setStatus('idle')
  }
  const remove = (index: number) => setVideos(old => { URL.revokeObjectURL(old[index].url); return old.filter((_, i) => i !== index) })
  const analyse = async () => {
    if (!allValid) return
    setApiError(null)
    setStatus('analyzing')
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 5 * 60 * 1000)
    try {
      let payload: AnalysisResult
      if (CLOUD_MODE) {
        if (!accessCode.trim()) throw new Error('Enter the demo access code to use live GPU analysis.')
        payload = await cloudInference(videos, accessCode.trim(), controller.signal)
      } else {
        const body = new FormData()
        videos.forEach(video => body.append('videos', video.file))
        payload = await responsePayload(await fetch('/api/infer', { method: 'POST', body, signal: controller.signal })) as AnalysisResult
      }
      setResult(payload)
      setStatus('completed')
      window.setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 80)
    } catch (error) {
      setStatus('failed')
      setApiError(error instanceof DOMException && error.name === 'AbortError'
        ? 'Inference timed out. Please try again.'
        : error instanceof Error ? error.message : 'Inference failed.')
    } finally {
      window.clearTimeout(timeout)
    }
  }
  const reset = () => { videos.forEach(v => URL.revokeObjectURL(v.url)); setVideos([]); setSelectedDemo(null); setResult(null); setApiError(null); setStatus('idle'); document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' }) }
  const selectDemo = (item: AnalysisResult) => { setSelectedDemo(item); setResult(null); setStatus('idle'); window.setTimeout(() => document.getElementById('prepared-review')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50) }
  const analyseDemo = () => {
    if (!selectedDemo) return
    setStatus('analyzing')
    window.setTimeout(() => {
      setResult(selectedDemo); setStatus('completed')
      window.setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 80)
    }, 1400)
  }

  return <>
    <section className="workspace" id="workspace">
      <div className="section-heading"><div><span className="section-number">01</span><div><small>INCIDENT INPUT</small><h2>Add your camera views</h2><p>Use synchronized clips showing the same incident.</p></div></div><div className="privacy"><LockKeyhole size={15} /><span><strong>Private by design</strong>Videos are not stored in this frontend prototype.</span></div></div>
      <div className="video-grid">{[0, 1, 2, 3].map(i => <UploadSlot key={i} index={i} input={videos[i]} onRemove={() => remove(i)} onPick={addFiles} />)}</div>
      <div className="requirements"><span className={videos.length ? 'ok' : ''}><Check size={14} /> 1–4 views</span><span className={videos.length && videos.every(v => !v.duration || v.duration <= MAX_SECONDS) ? 'ok' : ''}><Clock3 size={14} /> Maximum 5 seconds</span><span className={videos.length && videos.every(v => ACCEPTED.includes(v.file.type)) ? 'ok' : ''}><FileVideo size={14} /> MP4, MOV or WebM</span></div>
      {CLOUD_MODE && <div className="access-code"><div className="access-copy"><span><LockKeyhole size={15} /><strong>Live-analysis access code</strong></span><p>Live upload is access-controlled to limit GPU costs. Prepared cases remain open to everyone.</p><a href="mailto:MikeCharlotteMao@outlook.com?subject=RefereeAI%20live%20demo%20access">Request access <ArrowRight size={13} /></a></div><label><span className="sr-only">Demo access code</span><input type="password" value={accessCode} onChange={event => setAccessCode(event.target.value)} placeholder="Enter demo code" autoComplete="off" /></label></div>}
      <div className="action-row"><div><strong>{videos.length ? `${videos.length} view${videos.length > 1 ? 's' : ''} selected` : 'No clips selected'}</strong><span>{videos.some(v => v.status === 'invalid') ? 'Resolve invalid clips before continuing.' : allValid ? 'Input checks passed.' : 'Add a clip to begin.'}</span></div><button className="analyse-button" disabled={!allValid || status === 'analyzing'} onClick={analyse}>{status === 'analyzing' ? <><span className="spinner" /> Analysing evidence…</> : <>Analyse incident <ArrowRight size={17} /></>}</button></div>
      {apiError && <div className="api-error"><AlertTriangle size={15} /><span><strong>Analysis failed.</strong> {apiError}</span></div>}
      <div className="cold-start"><Clock3 size={15} /><span><strong>{CLOUD_MODE ? 'On-demand GPU inference.' : 'Live local inference enabled.'}</strong> {CLOUD_MODE ? 'The first live analysis may take 30–90 seconds while the GPU starts. Temporary uploads are deleted after the request.' : 'The first analysis loads MViT and may take longer. Uploaded clips are processed in a temporary directory and deleted after the request.'}</span></div>
    </section>
    <DemoCases selected={selectedDemo} onSelect={selectDemo} />
    {selectedDemo && <PreparedIncident item={selectedDemo} analyzing={status === 'analyzing'} onAnalyse={analyseDemo} onClose={() => { setSelectedDemo(null); setResult(null); setStatus('idle') }} />}
    {result && <div id="results"><Results result={result} onReset={reset} /></div>}
  </>
}

function Method() {
  const steps = [{ n: '01', icon: <FileVideo />, title: 'Five temporal windows', text: 'Each short clip is divided into one-second evidence windows.' }, { n: '02', icon: <Layers3 />, title: 'Multi-view representation', text: 'MViT encodes every available camera angle and time window.' }, { n: '03', icon: <BarChart3 />, title: 'Task-specific fusion', text: 'DT-GCVA predicts foul type and severity with separate evidence weights.' }]
  return <><section className="method" id="method"><div className="method-copy"><small>HOW IT WORKS</small><h2>One incident.<br />Multiple perspectives.</h2><p>The prototype turns a short multi-camera incident into structured, inspectable model evidence.</p></div><div className="method-steps">{steps.map(s => <article key={s.n}><span>{s.n}</span><div className="method-icon">{s.icon}</div><h3>{s.title}</h3><p>{s.text}</p></article>)}</div></section>
    <section className="future-direction"><div className="future-icon"><Activity size={28} /></div><div><small>FUTURE DIRECTION</small><h2>From incident analysis to real-time referee assistance.</h2><p>The long-term vision is to monitor synchronized match feeds, flag potentially important incidents and present multi-view evidence to officials in real time.</p></div><div className="future-principle"><strong>Human in command</strong><span>Designed to support faster, more informed review—not replace the referee’s final judgement.</span></div></section>
    <section className="limitations" id="limitations"><ShieldCheck size={25} /><div><small>RESPONSIBLE USE</small><h2>Decision support, not a referee.</h2><p>This research prototype can surface patterns and uncertainty. It cannot confirm a foul, replace an official or provide reliable red-card screening. Always interpret outputs with the original footage and human judgement.</p></div></section></>
}

export default function App() {
  return <><Header /><main><Hero /><Workspace /><Method /></main><footer><Brand /><div className="footer-credit"><span>Designed &amp; built by Lecheng Mao</span><a href="mailto:MikeCharlotteMao@outlook.com">MikeCharlotteMao@outlook.com</a></div><p>AI football incident analysis · Research prototype</p><span>Phase 3 · 2026</span></footer></>
}
