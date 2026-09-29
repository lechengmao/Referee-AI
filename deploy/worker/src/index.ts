export interface Env {
  UPLOADS: R2Bucket
  GATE: DurableObjectNamespace
  DEMO_CODE: string
  IP_HASH_SECRET: string
  INFERENCE_SIGNING_SECRET: string
  MODAL_INFERENCE_URL: string
  MODAL_PROXY_KEY: string
  MODAL_PROXY_SECRET: string
  MAX_FILE_BYTES: string
  MAX_VIEWS: string
  IP_DAILY_LIMIT: string
  GLOBAL_DAILY_LIMIT: string
  MONTHLY_LIMIT: string
  SESSION_TTL_SECONDS: string
  ALLOWED_ORIGIN: string
}

type FileMeta = { name: string; size: number; type: string; sha256: string }
type Session = {
  id: string
  secret: string
  ipHash: string
  files: FileMeta[]
  uploaded: boolean[]
  status: 'uploading' | 'running' | 'complete' | 'failed'
  expiresAt: number
  fingerprint: string
  limits: { ip: number; global: number; month: number }
}

const ALLOWED_TYPES = new Set(['video/mp4', 'video/quicktime', 'video/webm'])
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), {
  status,
  headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

const bytesToHex = (bytes: ArrayBuffer) => [...new Uint8Array(bytes)].map(v => v.toString(16).padStart(2, '0')).join('')
const base64url = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes)).replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_')

async function hmac(secret: string, value: string): Promise<ArrayBuffer> {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'])
  return crypto.subtle.sign('HMAC', key, new TextEncoder().encode(value))
}

async function hash(value: string): Promise<string> {
  return bytesToHex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))
}

async function signedToken(secret: string, claims: Record<string, unknown>): Promise<string> {
  const encoded = base64url(new TextEncoder().encode(JSON.stringify(claims)))
  return `${encoded}.${bytesToHex(await hmac(secret, encoded))}`
}

async function gate(env: Env, path: string, body: unknown) {
  const stub = env.GATE.get(env.GATE.idFromName('global'))
  return stub.fetch(`https://gate.internal${path}`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
}

function sessionCredentials(request: Request) {
  return request.headers.get('X-Upload-Secret') || ''
}

async function createSession(request: Request, env: Env): Promise<Response> {
  const body = await request.json() as { files?: FileMeta[]; demoCode?: string }
  if (!env.DEMO_CODE || body.demoCode !== env.DEMO_CODE) return json({ error: 'Invalid demo access code.' }, 403)
  const files = body.files || []
  if (files.length < 1 || files.length > Number(env.MAX_VIEWS)) return json({ error: 'Upload 1–4 video views.' }, 400)
  if (files.some(file => !ALLOWED_TYPES.has(file.type) || file.size < 1 || file.size > Number(env.MAX_FILE_BYTES) || !/^[a-f0-9]{64}$/i.test(file.sha256))) {
    return json({ error: 'A file failed format, size, or checksum validation.' }, 400)
  }
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown'
  const ipHash = bytesToHex(await hmac(env.IP_HASH_SECRET, ip))
  const fingerprint = await hash(files.map(f => `${f.sha256}:${f.size}`).join('|'))
  const response = await gate(env, '/create', {
    ipHash, fingerprint, files,
    limits: { ip: Number(env.IP_DAILY_LIMIT), global: Number(env.GLOBAL_DAILY_LIMIT), month: Number(env.MONTHLY_LIMIT) },
    ttl: Number(env.SESSION_TTL_SECONDS),
  })
  return new Response(response.body, response)
}

async function uploadView(request: Request, env: Env, sessionId: string, index: number): Promise<Response> {
  const metaResponse = await gate(env, '/get', { id: sessionId, secret: sessionCredentials(request) })
  if (!metaResponse.ok) return new Response(metaResponse.body, metaResponse)
  const session = await metaResponse.json() as Session
  const expected = session.files[index]
  if (!expected) return json({ error: 'Invalid view index.' }, 400)
  const length = Number(request.headers.get('content-length') || 0)
  if (length !== expected.size || request.headers.get('content-type') !== expected.type) return json({ error: 'Upload metadata mismatch.' }, 400)
  await env.UPLOADS.put(`${session.id}/${index}`, request.body, {
    httpMetadata: { contentType: expected.type }, customMetadata: { expiresAt: String(session.expiresAt), sha256: expected.sha256 },
  })
  return gate(env, '/uploaded', { id: session.id, secret: session.secret, index })
}

async function internalObject(request: Request, env: Env, sessionId: string, index: number): Promise<Response> {
  const expires = Number(new URL(request.url).searchParams.get('expires') || 0)
  const signature = new URL(request.url).searchParams.get('signature') || ''
  const signed = `${sessionId}:${index}:${expires}`
  if (expires < Math.floor(Date.now() / 1000) || signature !== bytesToHex(await hmac(env.INFERENCE_SIGNING_SECRET, signed))) return json({ error: 'Expired object URL.' }, 403)
  const object = await env.UPLOADS.get(`${sessionId}/${index}`)
  if (!object) return json({ error: 'Object not found.' }, 404)
  return new Response(object.body, { headers: { 'content-type': object.httpMetadata?.contentType || 'application/octet-stream', 'content-length': String(object.size), 'cache-control': 'private, no-store' } })
}

async function runInference(request: Request, env: Env, sessionId: string): Promise<Response> {
  const reserve = await gate(env, '/reserve', { id: sessionId, secret: sessionCredentials(request) })
  if (!reserve.ok) return new Response(reserve.body, reserve)
  const session = await reserve.json() as Session
  const expires = Math.floor(Date.now() / 1000) + 300
  const origin = new URL(request.url).origin
  const videos = await Promise.all(session.files.map(async (file, index) => {
    const signature = bytesToHex(await hmac(env.INFERENCE_SIGNING_SECRET, `${session.id}:${index}:${expires}`))
    return { ...file, url: `${origin}/api/internal/object/${session.id}/${index}?expires=${expires}&signature=${signature}` }
  }))
  const payload = JSON.stringify({ requestId: session.id, videos })
  const bodySha256 = await hash(payload)
  const token = await signedToken(env.INFERENCE_SIGNING_SECRET, { jti: session.id, exp: Math.floor(Date.now() / 1000) + 300, body_sha256: bodySha256 })
  try {
    const modal = await fetch(env.MODAL_INFERENCE_URL, {
      method: 'POST',
      headers: {
        'content-type': 'application/json', 'X-Inference-Token': token,
        'Modal-Key': env.MODAL_PROXY_KEY, 'Modal-Secret': env.MODAL_PROXY_SECRET,
      },
      body: payload,
    })
    const text = await modal.text()
    if (!modal.ok) throw new Error(`Inference service returned ${modal.status}: ${text.slice(0, 180)}`)
    await gate(env, '/finish', { id: session.id, ok: true })
    return new Response(text, { headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } })
  } catch (error) {
    await gate(env, '/finish', { id: session.id, ok: false })
    return json({ error: error instanceof Error ? error.message : 'Inference failed.' }, 502)
  } finally {
    await Promise.all(session.files.map((_, index) => env.UPLOADS.delete(`${session.id}/${index}`)))
  }
}

async function cleanup(env: Env) {
  let cursor: string | undefined
  do {
    const listed = await env.UPLOADS.list({ cursor, limit: 1000, include: ['customMetadata'] })
    const now = Date.now()
    await Promise.all(listed.objects.filter(object => Number(object.customMetadata?.expiresAt || 0) < now).map(object => env.UPLOADS.delete(object.key)))
    cursor = listed.truncated ? listed.cursor : undefined
  } while (cursor)
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    const origin = request.headers.get('Origin')
    const cors: Record<string, string> = origin === env.ALLOWED_ORIGIN ? {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-Upload-Secret',
      'Access-Control-Max-Age': '86400',
      'Vary': 'Origin',
    } : {}
    if (request.method === 'OPTIONS') return new Response(null, { status: origin === env.ALLOWED_ORIGIN ? 204 : 403, headers: cors })
    let response: Response
    if (url.pathname === '/api/health') response = json({ status: 'ok', inference: 'scale-to-zero' })
    else if (url.pathname === '/api/session' && request.method === 'POST') response = await createSession(request, env)
    else {
    let match = url.pathname.match(/^\/api\/upload\/([^/]+)\/(\d+)$/)
      if (match && request.method === 'PUT') response = await uploadView(request, env, match[1], Number(match[2]))
      else {
        match = url.pathname.match(/^\/api\/infer\/([^/]+)$/)
        if (match && request.method === 'POST') response = await runInference(request, env, match[1])
        else {
          match = url.pathname.match(/^\/api\/internal\/object\/([^/]+)\/(\d+)$/)
          response = match && request.method === 'GET' ? await internalObject(request, env, match[1], Number(match[2])) : json({ error: 'Not found.' }, 404)
        }
      }
    }
    const headers = new Headers(response.headers)
    Object.entries(cors).forEach(([key, value]) => headers.set(key, value))
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers })
  },
  async scheduled(_event: ScheduledEvent, env: Env): Promise<void> { await cleanup(env) },
}

export class InferenceGate implements DurableObject {
  constructor(private state: DurableObjectState) {}

  async fetch(request: Request): Promise<Response> {
    const path = new URL(request.url).pathname
    const body = await request.json() as Record<string, any>
    const now = Date.now()
    if (path === '/create') {
      const day = new Date(now).toISOString().slice(0, 10)
      const month = day.slice(0, 7)
      const ipKey = `count:ip:${day}:${body.ipHash}`
      const globalKey = `count:global:${day}`
      const monthKey = `count:month:${month}`
      const keys = [ipKey, globalKey, monthKey, `dedupe:${body.ipHash}:${body.fingerprint}`]
      const counts = await this.state.storage.get<number | string>(keys)
      const ipCount = Number(counts.get(ipKey) || 0)
      const globalCount = Number(counts.get(globalKey) || 0)
      const monthCount = Number(counts.get(monthKey) || 0)
      const duplicate = counts.get(keys[3])
      if (duplicate) return json({ error: 'This incident was already submitted recently.' }, 409)
      if (ipCount >= body.limits.ip) return json({ error: 'This IP has reached today’s 2-analysis limit.' }, 429)
      if (globalCount >= body.limits.global) return json({ error: 'Today’s live-analysis quota has been reached. Prepared cases remain available.' }, 429)
      if (monthCount >= body.limits.month) return json({ error: 'The monthly inference budget has been reached. Prepared cases remain available.' }, 429)
      const id = crypto.randomUUID(), secret = crypto.randomUUID() + crypto.randomUUID()
      const session: Session = { id, secret, ipHash: body.ipHash, files: body.files, uploaded: body.files.map(() => false), status: 'uploading', expiresAt: now + body.ttl * 1000, fingerprint: body.fingerprint, limits: body.limits }
      await this.state.storage.put(`session:${id}`, session)
      return json({ id, secret, expiresAt: session.expiresAt })
    }
    const session = await this.state.storage.get<Session>(`session:${body.id}`)
    if (!session || session.expiresAt < now || (body.secret && body.secret !== session.secret)) return json({ error: 'Upload session is invalid or expired.' }, 403)
    if (path === '/get') return json(session)
    if (path === '/uploaded') {
      if (session.status !== 'uploading' || !Number.isInteger(body.index) || !session.files[body.index]) return json({ error: 'Invalid upload state.' }, 409)
      session.uploaded[body.index] = true
      await this.state.storage.put(`session:${session.id}`, session)
      return json({ ok: true })
    }
    if (path === '/reserve') {
      if (session.status !== 'uploading' || session.uploaded.some(value => !value)) return json({ error: 'Session is incomplete or already used.' }, 409)
      const day = new Date(now).toISOString().slice(0, 10), month = day.slice(0, 7)
      const ipKey = `count:ip:${day}:${session.ipHash}`, globalKey = `count:global:${day}`, monthKey = `count:month:${month}`
      const counts = await this.state.storage.get<number>([ipKey, globalKey, monthKey])
      const ipCount = counts.get(ipKey) || 0
      const globalCount = counts.get(globalKey) || 0
      const monthCount = counts.get(monthKey) || 0
      if (ipCount >= session.limits.ip || globalCount >= session.limits.global || monthCount >= session.limits.month) return json({ error: 'The live-analysis quota was reached while this upload was pending.' }, 429)
      session.status = 'running'
      await this.state.storage.put({ [`session:${session.id}`]: session, [ipKey]: ipCount + 1, [globalKey]: globalCount + 1, [monthKey]: monthCount + 1, [`dedupe:${session.ipHash}:${session.fingerprint}`]: session.id })
      return json(session)
    }
    if (path === '/finish') {
      session.status = body.ok ? 'complete' : 'failed'
      await this.state.storage.put(`session:${session.id}`, session)
      return json({ ok: true })
    }
    return json({ error: 'Unknown gate operation.' }, 404)
  }
}
