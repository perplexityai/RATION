#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.llm_client import (
    openai_model_id,
    usage_from_openai,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s proxy: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("openai_proxy")

_TASK_TOKENS: dict[str, int] = {}
_TASK_TOKENS_LOCK = threading.Lock()
_BUCKET_SEGMENT = "/experiment-task/"
_UPSTREAM_ATTEMPTS = 6


def experiment_bucket(path: str, headers: Any) -> str:
    header = headers.get("X-EXPERIMENT-TASK") if headers is not None else None
    if header:
        return str(header)
    clean = path.split("?", 1)[0]
    start = clean.find(_BUCKET_SEGMENT)
    if start < 0:
        return ""
    rest = clean[start + len(_BUCKET_SEGMENT):]
    return unquote(rest.split("/", 1)[0])


def strip_bucket_segment(path: str) -> str:
    start = path.find(_BUCKET_SEGMENT)
    if start < 0:
        return path
    rest = path[start + len(_BUCKET_SEGMENT):]
    tail = rest.split("/", 1)
    return path[:start] + ("/" + tail[1] if len(tail) > 1 else "/")


def record_usage(bucket: str, charged: int) -> None:
    with _TASK_TOKENS_LOCK:
        _TASK_TOKENS[bucket] = _TASK_TOKENS.get(bucket, 0) + max(0, charged)


def usage_payload(bucket: str) -> dict[str, Any]:
    with _TASK_TOKENS_LOCK:
        return {"task": bucket, "tokens": _TASK_TOKENS.get(bucket, 0)}


def _prompt_chars(messages: list[dict]) -> int:
    total = 0
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(
                len(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text", ""), str)
            )
    return total


def _model_aliases() -> dict[str, str]:
    raw = os.environ.get("PROXY_MODEL_ALIASES_JSON", "").strip()
    return {request.strip().lower(): target for request, target in json.loads(raw).items()} if raw else {}


def resolve_model(model: str) -> str:
    forced = os.environ.get("PROXY_FORCE_MODEL", "").strip()
    if forced:
        return openai_model_id(forced)
    if not model:
        raise ValueError("request names no model; the proxy never substitutes one")
    return openai_model_id(_model_aliases().get(model.strip().lower(), model))


def _upstream_url() -> str:
    return (os.environ.get("OPENAI_UPSTREAM_URL") or "https://api.openai.com/v1").rstrip("/")


def _retry_after_s(headers: Any, attempt: int) -> float:
    try:
        value = headers.get("retry-after") if headers is not None else None
        if value:
            return max(0.5, float(value))
    except (TypeError, ValueError):
        pass
    return min(2.0 * 2**attempt, 60.0)


def forward_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT") or 600.0)
    url = _upstream_url() + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(_UPSTREAM_ATTEMPTS):
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=timeout)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last = exc
            delay = _retry_after_s(None, attempt)
        else:
            if response.status_code < 400:
                return response.json()
            if response.status_code != 429 and response.status_code < 500:
                raise UpstreamRequestError(response.status_code, response.text)
            last = UpstreamRequestError(response.status_code, response.text)
            delay = _retry_after_s(response.headers, attempt)
        if attempt + 1 < _UPSTREAM_ATTEMPTS:
            log.warning("upstream transient failure (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, _UPSTREAM_ATTEMPTS, last, delay)
            time.sleep(delay)
    raise last


class UpstreamRequestError(RuntimeError):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"upstream HTTP {status}: {text[:500]}")
        self.status = status
        self.text = text


def openai_message_to_stream_chunks(
    message: dict,
    finish: str,
    *,
    completion_id: str,
    created: int,
    model: str,
    usage: dict | None = None,
) -> list[dict]:

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}
            ],
        }

    chunks = [chunk({"role": "assistant", "content": ""})]
    if message.get("content"):
        chunks.append(chunk({"content": message["content"]}))
    for index, call in enumerate(message.get("tool_calls") or []):
        chunks.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    ]
                }
            )
        )
    chunks.append(chunk({}, finish))
    if usage is not None:
        chunks.append(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
        )
    return chunks


def render_event_stream(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_event_stream(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/experiment-usage"):
            tid = (parse_qs(urlparse(self.path).query).get("task") or [""])[0]
            self._send_json(200, usage_payload(tid))
        elif path.endswith("/models"):
            self._send_json(200, {"object": "list", "data": []})
        else:
            self._send_json(200, {"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        bucket = experiment_bucket(self.path, self.headers)
        self.path = strip_bucket_segment(self.path)
        try:
            req = json.loads(raw or b"{}")
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"bad JSON: {exc}"}})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unhandled path {self.path}"}})
            return
        try:
            requested_model = req.get("model")
            if not isinstance(requested_model, str):
                raise ValueError("request names no model; the proxy never substitutes one")
            request_model = resolve_model(requested_model)
            excluded_prompt_chars = int(
                self.headers.get("X-EXPERIMENT-EXCLUDED-PROMPT-CHARS", "0") or 0
            )
            if excluded_prompt_chars < 0:
                raise ValueError("X-EXPERIMENT-EXCLUDED-PROMPT-CHARS must be non-negative")
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        messages = req.get("messages") or []
        streaming = bool(req.get("stream"))
        stream_options = req.get("stream_options")
        include_usage = streaming and isinstance(stream_options, dict) and bool(
            stream_options.get("include_usage")
        )
        body = {k: v for k, v in req.items() if k not in ("stream", "stream_options")}
        body["model"] = request_model
        log.info("chat/completions model=%s messages=%d tools=%d stream=%s",
                 request_model, len(messages), len(req.get("tools") or []), streaming)
        try:
            completion = forward_chat_completion(body)
        except UpstreamRequestError as exc:
            log.error("upstream rejected the request: %s", exc)
            try:
                payload = json.loads(exc.text)
            except json.JSONDecodeError:
                payload = {"error": {"message": exc.text}}
            self._send_json(exc.status, payload)
            return
        except Exception as exc:
            log.exception("upstream call failed")
            self._send_json(502, {"error": {"message": f"upstream error: {type(exc).__name__}: {exc}"}})
            return

        usage = usage_from_openai(completion.get("usage"))
        if bucket:
            total = usage.total_input_tokens + usage.output_tokens
            prompt_chars = _prompt_chars(messages)
            excluded_prompt_chars = min(excluded_prompt_chars, prompt_chars)
            excluded_tokens = 0
            if excluded_prompt_chars and prompt_chars:
                excluded_tokens = min(
                    usage.total_input_tokens,
                    round(usage.total_input_tokens * excluded_prompt_chars / prompt_chars),
                )
            record_usage(bucket, total - excluded_tokens)

        if not streaming:
            self._send_json(200, completion)
            return
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {"role": "assistant", "content": ""}
        finish = choice.get("finish_reason") or ("tool_calls" if message.get("tool_calls") else "stop")
        chunks = openai_message_to_stream_chunks(
            message,
            finish,
            completion_id=completion.get("id") or "chatcmpl-" + uuid.uuid4().hex[:24],
            created=int(completion.get("created") or time.time()),
            model=completion.get("model") or request_model,
            usage=completion.get("usage") if include_usage else None,
        )
        self._send_event_stream(render_event_stream(chunks))


class LocalProxy:
    def __init__(self, *, port: int = 0, aliases: dict[str, str] | None = None) -> None:
        self.requested_port = port
        self.aliases = dict(aliases or {})
        self.server: ThreadingHTTPServer | None = None
        self._previous_aliases: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "LocalProxy":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._previous_aliases = os.environ.get("PROXY_MODEL_ALIASES_JSON")
        if self.aliases:
            os.environ["PROXY_MODEL_ALIASES_JSON"] = json.dumps(self.aliases, sort_keys=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", self.requested_port), Handler)
        threading.Thread(target=self.server.serve_forever, name="openai-proxy", daemon=True).start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._previous_aliases is None:
            os.environ.pop("PROXY_MODEL_ALIASES_JSON", None)
        else:
            os.environ["PROXY_MODEL_ALIASES_JSON"] = self._previous_aliases


def main() -> None:
    ap = argparse.ArgumentParser(description='Serve the local OpenAI accounting proxy.')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8903)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("OpenAI proxy listening on http://%s:%d (POST .../chat/completions -> %s)",
             args.host, args.port, _upstream_url())
    server.serve_forever()


if __name__ == "__main__":
    main()
