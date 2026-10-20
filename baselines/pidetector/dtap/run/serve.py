from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.pidetector.core import MODEL_ID, MODEL_REVISION, TRAINED_CONTEXT_TOKENS

MAX_INPUT_TOKENS = 512
ONNX_FILES = [
    "onnx/model.onnx",
    "onnx/config.json",
    "onnx/tokenizer.json",
    "onnx/tokenizer_config.json",
    "onnx/special_tokens_map.json",
    "onnx/added_tokens.json",
    "onnx/spm.model",
]


class OnnxClassifier:
    def __init__(self, model_dir: str | None = None) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        self._np = np
        root = Path(model_dir).resolve() if model_dir else Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, allow_patterns=ONNX_FILES))
        onnx_dir = root / "onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.tokenizer.model_input_names = ["input_ids", "attention_mask"]
        self.tokenizer.truncation_side = "right"
        config = json.loads((onnx_dir / "config.json").read_text())
        self.id2label = {int(key): value for key, value in config["id2label"].items()}
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(os.cpu_count() or 1, 4))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = [item.name for item in self.session.get_inputs()]
        self.lock = threading.Lock()

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        peak = max(logits)
        exp = [math.exp(value - peak) for value in logits]
        total = sum(exp)
        return [value / total for value in exp]

    @lru_cache(maxsize=4096)
    def classify(self, text: str) -> dict[str, Any]:
        input_tokens = len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_tensors="np",
        )
        classified_tokens = int(encoded["input_ids"].shape[-1])
        feeds = {name: self._np.asarray(encoded[name], dtype=self._np.int64) for name in self.inputs}
        with self.lock:
            logits = self.session.run(None, feeds)[0][0].tolist()
        probabilities = self._softmax([float(value) for value in logits])
        label_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "label": self.id2label[label_id],
            "score": probabilities[label_id],
            "input_tokens": input_tokens,
            "classified_tokens": classified_tokens,
            "truncated": input_tokens > classified_tokens,
            "over_context": input_tokens > TRAINED_CONTEXT_TOKENS,
        }


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

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"ready": True})

    def do_POST(self) -> None:
        if self.path != "/v1/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                raise ValueError("request body must be 1..20MiB")
            text = json.loads(self.rfile.read(length)).get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            self._send(200, self.server.classifier.classify(text))
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {str(exc)[:500]}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the PI Detector classifier for DTap.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8908)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    classifier = OnnxClassifier(args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.classifier = classifier
    print(f"PIDETECTOR_SIDECAR_READY http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
