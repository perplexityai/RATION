from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.promptguard.agentdojo.config import CONTEXT_WINDOW_TOKENS, LABEL_MEANING, MALICIOUS_LABEL

SEGMENT_TOKENS = 512
BATCH_SIZE = 16
REENCODE_MAX_LENGTH = 512


def segment_ids(token_ids: list[int], size: int = SEGMENT_TOKENS) -> list[list[int]]:
    if not token_ids:
        return [[]]
    return [token_ids[i:i + size] for i in range(0, len(token_ids), size)]


def malicious_probability(label: str, score: float) -> float:
    return float(score) if str(label) == MALICIOUS_LABEL else 1.0 - float(score)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [malicious_probability(r["label"], r["score"]) for r in results]
    argmax = max(range(len(scores)), key=scores.__getitem__)
    winner = results[argmax]
    return {
        "label": str(winner["label"]),
        "score": float(winner["score"]),
        "label_meaning": LABEL_MEANING.get(str(winner["label"]), "UNKNOWN"),
        "n_segments": len(results),
        "segment_argmax": argmax,
        "segment_max_malicious": scores[argmax],
    }


class Classifier:
    def __init__(self, snapshot: Path, workers: int, threads: int) -> None:
        import torch
        from transformers import AutoTokenizer, pipeline

        torch.set_num_threads(threads)
        device = torch.device("cpu")
        first = pipeline("text-classification", model=str(snapshot), device=device, trust_remote_code=True)
        self.pool: "queue.Queue[Any]" = queue.Queue()
        self.pool.put(first)
        for _ in range(workers - 1):
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            self.pool.put(pipeline("text-classification", model=first.model, tokenizer=tokenizer,
                                   device=device, trust_remote_code=True))

    def classify(self, text: str) -> dict[str, Any]:
        pipe = self.pool.get()
        try:
            tokenizer = pipe.tokenizer
            token_ids = tokenizer(text, truncation=False)["input_ids"]
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in segment_ids(token_ids)]
            results = pipe(texts, batch_size=BATCH_SIZE, truncation=True, max_length=REENCODE_MAX_LENGTH)
            if isinstance(results, dict):
                results = [results]
            if len(results) != len(texts):
                raise RuntimeError(f"pipeline returned {len(results)} results for {len(texts)} segments")
            guard_truncated = sum(
                1 for t in texts
                if len(tokenizer(t, truncation=False)["input_ids"]) > REENCODE_MAX_LENGTH
            )
        finally:
            self.pool.put(pipe)
        return {
            **aggregate(results),
            "input_tokens": len(token_ids),
            "over_context": len(token_ids) > CONTEXT_WINDOW_TOKENS,
            "guard_truncated_segments": guard_truncated,
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


def default_threads_and_workers() -> tuple[int, int]:
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    threads = max(1, min(8, cpus // 4))
    workers = max(1, min(3, cpus // threads))
    return threads, workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Prompt Guard 2 classifier for DTap.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8909)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    threads, workers = default_threads_and_workers()
    classifier = Classifier(Path(args.model_dir), workers=workers, threads=threads)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.classifier = classifier
    print(f"PROMPTGUARD_SIDECAR_READY http://{args.host}:{args.port} workers={workers} threads={threads}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
