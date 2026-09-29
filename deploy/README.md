# Production deployment

The production path is deliberately split into three trust zones:

1. `frontend/` is a static Vite build. Prepared incidents and their stored
   results work without any backend.
2. `deploy/worker/` is the only public live-inference gateway. It validates the
   access code and file manifest, streams temporary uploads to R2, applies
   atomic quotas in a Durable Object, signs one-use inference requests, and
   removes uploaded objects after success or failure.
3. `deploy/modal_app.py` owns the frozen MViT + DT-GCVA model. It scales from
   zero to one T4 GPU and requires both Modal proxy authentication and a
   short-lived request signature from the Worker.

No Modal URL, proxy credential, or signing secret is compiled into the browser.

## Limits encoded in the gateway

- 1–4 MP4, MOV, or WebM views
- 80 MB per view and five seconds per clip (browser and GPU both check duration)
- two analyses per IP per UTC day
- 20 analyses site-wide per UTC day
- 200 analyses per UTC month (change `MONTHLY_LIMIT` to the desired budget cap)
- one active GPU container with one input at a time
- five-minute GPU request token and 15-minute upload session
- no automatic GPU retry; duplicate submissions are rejected
- hourly cleanup of stale R2 objects, plus immediate deletion after inference

## 1. Modal

Install and authenticate the CLI, create a secret named
`referee-ai-gateway` containing a random `INFERENCE_SIGNING_SECRET`, then create
a Modal Proxy Token. Deploy from the repository root:

```sh
modal deploy deploy/modal_app.py
```

Record the generated endpoint URL and proxy token only as Worker secrets. The
signing secret must be identical in Modal and Cloudflare.

## 2. Cloudflare gateway

Create an R2 bucket called `referee-ai-uploads`, install the Worker dependencies,
and add these six secrets with `npx wrangler secret put NAME`:

```text
DEMO_CODE
IP_HASH_SECRET
INFERENCE_SIGNING_SECRET
MODAL_INFERENCE_URL
MODAL_PROXY_KEY
MODAL_PROXY_SECRET
```

Then deploy:

```sh
cd deploy/worker
npm install
npm run typecheck
npm run deploy
```

Route `/api/*` on the public site hostname to this Worker. The object download
URLs are signed, private, and expire after five minutes.

## 3. Static frontend

Copy `frontend/.env.production.example` to `.env.production`, build `frontend/`,
and publish `frontend/dist/` with Cloudflare Pages or Vercel:

```sh
cd frontend
npm install
npm run build
```

The production environment sets `VITE_INFERENCE_MODE=cloud`; local development
continues using the Flask API proxy. Do not add any cloud secret to a `VITE_*`
variable because Vite exposes those variables to visitors.

## Privacy and failure behavior

The Worker does not log video bodies. Modal downloads each view into a Python
temporary directory that is destroyed after the request. The Worker deletes the
R2 objects in a `finally` block and the hourly cleanup is a safety net for
abandoned uploads. If daily or monthly quota is exhausted, `/api/session`
returns 429 while the static site and prepared examples remain available.
