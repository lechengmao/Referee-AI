# RefereeAI frontend

Hosted prototype: **https://app.referee-ai.workers.dev/**

Phase 3 frontend prototype for the MViT + DT-GCVA inference service. It supports
both a local Flask API during development and the protected cloud gateway used
by the hosted site.

## Run locally

Start the API from the repository root:

```bash
TORCH_HOME="$PWD/.cache/torch" .venv/bin/python -m scripts.inference_api
```

Then start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Production build:

```bash
npm run build
```

The generated static site is written to `dist/`.

## Current scope

- Responsive single-page analysis workspace.
- One to four local video previews with client-side type, size and duration checks.
- Real five-window MViT feature extraction and DT-GCVA classification.
- Camera-contribution visualization.
- Three instant precomputed demonstration cases.
- Explicit uncertainty and responsible-use messaging.
- Uploaded-video analysis returns output from the frozen local checkpoint.

In local mode, uploaded videos are sent only to the local API and removed when
the request completes. In cloud mode, the browser uploads through the protected
Cloudflare gateway and the private GPU service described in
[`../deploy/README.md`](../deploy/README.md).

## API boundary

The production frontend submits validated files to a protected gateway, never
directly to the GPU service. Configure cloud mode from
`.env.production.example`; do not put secrets in `VITE_*` variables because Vite
embeds them in public browser assets.

The repository includes only the small prepared demo clips whose redistribution
rights have been confirmed. Do not add other SoccerNet footage unless its
redistribution terms permit it.
