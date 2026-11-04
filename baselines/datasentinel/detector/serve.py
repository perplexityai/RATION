#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from baselines.datasentinel import core


def _libc():
    import ctypes
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def tame_host_memory() -> None:
    libc = _libc()
    if libc is not None:
        libc.mallopt(-8, 2)


def trim_host_memory() -> None:
    libc = _libc()
    if libc is not None:
        libc.malloc_trim(0)


class Detector:

    def __init__(self, base_path: Path, checkpoint_path: Path, device_index: int = 0) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device_map = {"": f"cuda:{device_index}"} if torch.cuda.is_available() else "auto"
        self.torch = torch
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            str(base_path), quantization_config=bnb, device_map=device_map, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_path), add_bos_token=True, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, str(checkpoint_path))
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.lock = threading.Lock()

    def _fit(self, text: str, prompt: str, tokens: int, budget: int) -> tuple[str, int, bool]:
        if tokens <= budget:
            return prompt, tokens, False
        data_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        template_tokens = len(self.tokenizer(core.build_prompt("x"))["input_ids"])
        keep = max(1, min(len(data_ids), budget - template_tokens))
        while keep >= 1:
            cut = self.tokenizer.decode(data_ids[:keep], skip_special_tokens=True)
            if not cut.strip():
                break
            prompt = core.build_prompt(cut)
            tokens = len(self.tokenizer(prompt)["input_ids"])
            if tokens <= budget:
                return prompt, tokens, True
            keep -= max(1, (tokens - budget))
        raise ValueError(f"prompt budget {budget} leaves no data tokens beside the fixed template")

    def detect(self, text: str, max_prompt_tokens: int | None) -> dict[str, Any]:
        t0 = time.time()
        full_prompt = core.build_prompt(text)
        input_tokens = len(self.tokenizer(full_prompt)["input_ids"])
        if max_prompt_tokens is None:
            prompt, classified_tokens, truncated = full_prompt, input_tokens, False
        else:
            prompt, classified_tokens, truncated = self._fit(text, full_prompt, input_tokens, max_prompt_tokens)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.lock:
            with self.torch.no_grad():
                output = self.model.generate(
                    **encoded, max_new_tokens=core.MAX_NEW_TOKENS, repetition_penalty=core.REPETITION_PENALTY
                )[0]
        decoded = self.tokenizer.decode(output, skip_special_tokens=True)
        response = decoded.replace(prompt, "")
        return {
            "contaminated": core.is_contaminated(response),
            "response": response,
            "input_tokens": input_tokens,
            "classified_tokens": classified_tokens,
            "truncated": truncated,
            "latency_s": round(time.time() - t0, 3),
        }


class ReplicaPool:
    ATTEMPTS = 3

    def __init__(self, base_path: Path, checkpoint_path: Path, replicas: int) -> None:
        import multiprocessing as mp

        import torch

        ctx = mp.get_context("spawn")
        gpu_count = max(torch.cuda.device_count(), 1)
        self.workers: list[_Worker] = []
        for index in range(replicas):
            requests, replies = ctx.Queue(), ctx.Queue()
            proc = ctx.Process(
                target=_worker_main,
                args=(index % gpu_count, str(base_path), str(checkpoint_path), requests, replies),
                daemon=True,
            )
            proc.start()
            self.workers.append(_Worker(index, proc, requests, replies))
        for worker in self.workers:
            kind, payload = worker.replies.get()
            if kind != "ready":
                raise SystemExit(f"replica {worker.index + 1}: {payload}")
        self.free: "queue.Queue[_Worker]" = queue.Queue()
        for worker in self.workers:
            self.free.put(worker)

    def detect(self, text: str, max_prompt_tokens: int | None) -> dict[str, Any]:
        for attempt in range(1, self.ATTEMPTS + 1):
            worker = self.free.get()
            try:
                result = worker.call(("detect", text, max_prompt_tokens))
            except RuntimeError:
                if worker.alive:
                    self.free.put(worker)
                if attempt == self.ATTEMPTS:
                    raise
                continue
            self.free.put(worker)
            return result


class _Worker:
    CALL_TIMEOUT_S = 600.0

    def __init__(self, index: int, proc: Any, requests: Any, replies: Any) -> None:
        self.index, self.proc, self.requests, self.replies = index, proc, requests, replies
        self.alive = True

    def call(self, message: tuple) -> dict[str, Any]:
        self.requests.put(message)
        try:
            kind, payload = self.replies.get(timeout=self.CALL_TIMEOUT_S)
        except queue.Empty:
            self.alive = False
            raise RuntimeError(f"replica {self.index + 1} did not answer within {self.CALL_TIMEOUT_S:.0f}s; retired")
        if kind == "ok":
            return payload
        raise RuntimeError(f"replica {self.index + 1}: {payload}")


def _worker_main(device_index: int, base_path: str, checkpoint_path: str, requests: Any, replies: Any) -> None:
    tame_host_memory()
    try:
        detector = Detector(Path(base_path), Path(checkpoint_path), device_index=device_index)
        trim_host_memory()
        replies.put(("ready", None))
    except Exception as exc:
        replies.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return
    served = 0
    while True:
        _, text, max_prompt_tokens = requests.get()
        try:
            replies.put(("ok", detector.detect(text, max_prompt_tokens)))
        except Exception as exc:
            replies.put(("error", f"{type(exc).__name__}: {exc}"))
        served += 1
        if served % 200 == 0:
            trim_host_memory()


def make_handler(detector: ReplicaPool):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/v1/detect":
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                result = detector.detect(payload["text"], payload.get("max_prompt_tokens"))
            except Exception as exc:
                print(f"[detect] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, result)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the DataSentinel detector.")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=1)
    args = parser.parse_args()
    pool = ReplicaPool(args.base_path, args.checkpoint_path, args.replicas)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pool))
    server.daemon_threads = True
    print(f"[serve] listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
