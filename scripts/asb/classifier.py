#!/usr/bin/env python3

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PYTHON = ROOT / ".venv/bin/python"


class ClassifierSidecar:
    def __init__(self, model: str, revision: str, *, retries: int = 2, timeout: float = 600.0) -> None:
        self.retries = retries
        self.timeout = timeout
        log = open(Path(tempfile.gettempdir()) / f"asb-classifier-{os.getpid()}.log", "w")
        self.process = subprocess.Popen(
            [str(CLASSIFIER_PYTHON), "-m", "scripts.asb.classifier", "--model", model, "--revision", revision],
            stdout=subprocess.PIPE, stderr=log, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        atexit.register(self.stop)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"classifier sidecar exited before announcing its port (see {log.name})")
        self.base_url = f"http://127.0.0.1:{json.loads(line)['port']}"

    def classify(self, text: str) -> dict[str, Any]:
        body = json.dumps({"text": text}).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(f"{self.base_url}/v1/classify", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                if not isinstance(payload.get("label"), str) or not isinstance(payload.get("score"), (int, float)):
                    raise RuntimeError(f"classifier returned {payload!r}")
                return payload
            except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
                last = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"classifier unavailable: {last}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Classifier:
    def __init__(self, model: str, revision: str) -> None:
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        snapshot = snapshot_download(repo_id=model, revision=revision, local_files_only=True)
        self.pipeline = pipeline("text-classification", model=snapshot, device="cpu", trust_remote_code=True)
        self.lock = threading.Lock()

    def classify(self, text: str) -> dict[str, Any]:
        with self.lock:
            result = self.pipeline(text)[0]
        return {"label": str(result["label"]), "score": float(result["score"])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            self._send(200, self.server.classifier.classify(text))
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {str(exc)[:500]}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a Hugging Face text classifier on a loopback port.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.classifier = Classifier(args.model, args.revision)
    print(json.dumps({"port": server.server_address[1]}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
