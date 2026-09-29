"""Scale-to-zero Modal GPU endpoint for frozen MViT + DT-GCVA inference.

Deploy with: modal deploy deploy/modal_app.py
The endpoint is additionally protected by Modal proxy authentication. Only the
Cloudflare Worker receives those proxy credentials.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import modal
from fastapi import Request


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "experiments/spatiotemporal-balanced-full-20260928/best.pt"
MVIT_WEIGHTS = ROOT / ".cache/torch/hub/checkpoints/mvit_v2_s-ae3be167.pth"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]==0.116.1",
        "opencv-python-headless==4.12.0.88",
        "requests==2.32.5",
        "torch==2.8.0",
        "torchvision==0.23.0",
    )
    .add_local_dir(ROOT / "src", remote_path="/app/src")
    .add_local_file(CHECKPOINT, remote_path="/app/model/best.pt")
    .add_local_file(
        MVIT_WEIGHTS,
        remote_path="/root/.cache/torch/hub/checkpoints/mvit_v2_s-ae3be167.pth",
    )
)

app = modal.App("referee-ai-inference")
FOUL_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive/Simulation",
]
SEVERITY_CLASSES = [
    "No offence", "Offence + No card", "Offence + Yellow card",
    "Offence + Red card",
]
MAX_FILE_BYTES = 80 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * MAX_FILE_BYTES
MAX_DURATION_SECONDS = 5.15
ALLOWED_TYPES = {"video/mp4", "video/quicktime", "video/webm"}


def _decode_segment(value: str) -> dict:
    padding = "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(value + padding))


def _verify_gateway_token(token: str, body: bytes) -> dict:
    try:
        encoded, signature = token.split(".", 1)
        expected = hmac.new(
            os.environ["INFERENCE_SIGNING_SECRET"].encode(),
            encoded.encode(), hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        claims = _decode_segment(encoded)
    except Exception as exc:
        raise ValueError("Invalid gateway token") from exc
    if claims.get("exp", 0) < int(time.time()):
        raise ValueError("Expired gateway token")
    digest = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(claims.get("body_sha256", ""), digest):
        raise ValueError("Request body does not match token")
    return claims


@app.cls(
    image=image,
    gpu="T4",
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=300,
    retries=0,
    secrets=[modal.Secret.from_name("referee-ai-gateway")],
)
@modal.concurrent(max_inputs=1)
class RefereeModel:
    @modal.enter()
    def load(self):
        import sys
        import torch
        from torchvision.models.video import MViT_V2_S_Weights

        sys.path.insert(0, "/app")
        from src.models.mvit_encoder import MViTEncoder
        from src.models.spatiotemporal_model import SpatiotemporalDTGCVAModel

        checkpoint = torch.load("/app/model/best.pt", map_location="cpu", weights_only=False)
        self.device = "cuda"
        self.model = SpatiotemporalDTGCVAModel(**checkpoint["config"]["model"])
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval().to(self.device)
        self.encoder = MViTEncoder(pretrained=True, representation="embedding")
        self.encoder.eval().to(self.device)
        self.preprocess = MViT_V2_S_Weights.KINETICS400_V1.transforms()

    def _duration(self, path: Path) -> float:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("A video could not be decoded")
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or frames <= 0:
                raise ValueError("A video has invalid metadata")
            return frames / fps
        finally:
            cap.release()

    def _download(self, video: dict, directory: Path, index: int) -> Path:
        import requests
        if video.get("type") not in ALLOWED_TYPES:
            raise ValueError("Unsupported video type")
        size = int(video.get("size", 0))
        if size <= 0 or size > MAX_FILE_BYTES:
            raise ValueError("Invalid video size")
        parsed = urlparse(video.get("url", ""))
        if parsed.scheme != "https":
            raise ValueError("Video download URL must use HTTPS")
        suffix = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}[video["type"]]
        path = directory / f"view-{index}{suffix}"
        total = 0
        with requests.get(video["url"], stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with path.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise ValueError("Downloaded video exceeds size limit")
                    output.write(chunk)
        if total != size:
            raise ValueError("Downloaded video size mismatch")
        if self._duration(path) > MAX_DURATION_SECONDS:
            raise ValueError("Each video must be 5 seconds or shorter")
        return path

    def _top(self, values, labels):
        order = sorted(range(len(labels)), key=values.__getitem__, reverse=True)[:3]
        return [{"label": labels[i], "value": values[i]} for i in order]

    def predict(self, paths: list[Path]) -> dict:
        import torch
        from src.datasets.mvfoul import read_temporal_windows

        with torch.inference_mode():
            raw_views = [read_temporal_windows(path)[0] for path in paths]
            windows_count = min(len(windows) for windows in raw_views)
            encoded = []
            for raw in raw_views:
                windows = torch.stack([self.preprocess(window) for window in raw[:windows_count]])
                parts = [self.encoder(chunk.to(self.device)).cpu() for chunk in windows.split(2)]
                encoded.append(torch.cat(parts))
            features = torch.stack(encoded).unsqueeze(0).float().to(self.device)
            mask = torch.ones(features.shape[:3], dtype=torch.bool, device=self.device)
            output = self.model(features, mask)
            foul = output["foul_logits"][0].softmax(-1).cpu().tolist()
            severity = output["severity_logits"][0].softmax(-1).cpu().tolist()
            views = ((output["foul_view_attention"][0] + output["severity_view_attention"][0]) / 2).cpu().tolist()
            total = sum(views)
            views = [value / total for value in views]
        confidence = min(max(foul), max(severity))
        return {
            "id": "live-upload",
            "title": "Uploaded incident",
            "subtitle": f"{len(paths)} uploaded view{'s' if len(paths) != 1 else ''} · {windows_count} temporal windows · cloud inference",
            "foul": self._top(foul, FOUL_CLASSES),
            "severity": self._top(severity, SEVERITY_CLASSES),
            "temporalWeights": [],
            "viewWeights": views,
            "confidence": "high" if confidence >= .70 else "moderate" if confidence >= .50 else "low",
            "note": "Generated by the frozen MViT + DT-GCVA checkpoint. Interpret as research evidence, not an officiating decision.",
        }

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True, docs=False)
    async def infer(self, request: Request):
        from fastapi import HTTPException

        raw = await request.body()
        auth = request.headers.get("X-Inference-Token", "")
        try:
            claims = _verify_gateway_token(auth, raw)
            payload = json.loads(raw)
            videos = payload.get("videos", [])
            if not 1 <= len(videos) <= 4 or claims.get("jti") != payload.get("requestId"):
                raise ValueError("Invalid inference request")
            if sum(int(item.get("size", 0)) for item in videos) > MAX_TOTAL_BYTES:
                raise ValueError("Total upload size exceeds limit")
            with tempfile.TemporaryDirectory(prefix="referee-ai-") as directory:
                paths = [self._download(item, Path(directory), i) for i, item in enumerate(videos)]
                return self.predict(paths)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
