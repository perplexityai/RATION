from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer

from scripts.common.openai_proxy import Handler


class StatusRecordingHandler(Handler):

    def _send_json(self, code: int, payload: dict) -> None:
        if self.command == "POST" and code != 200:
            self.server.failed_statuses.append(code)
        super()._send_json(code, payload)


class StatusRecordingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StatusRecordingHandler)
        self.failed_statuses: list[int] = []


def start_proxy(model: str) -> tuple[StatusRecordingServer, threading.Thread]:
    server = StatusRecordingServer()
    thread = threading.Thread(target=server.serve_forever, name="progent-openai-proxy", daemon=True)
    thread.start()
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "proxy")
    os.environ["PROXY_FORCE_MODEL"] = model
    return server, thread
