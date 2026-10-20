from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

import openai

from scripts.common.openai_proxy import Handler


@dataclass
class FilterTransport:

    server: ThreadingHTTPServer
    thread: threading.Thread
    client: openai.OpenAI

    @classmethod
    def start(cls) -> "FilterTransport":
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, name="agentdojo-toolfilter-proxy", daemon=True)
        thread.start()
        host, port = server.server_address
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{host}:{port}/v1",
            timeout=float(os.environ.get("LLM_REQUEST_TIMEOUT", "600")),
        )
        return cls(server, thread, client)

    def close(self) -> None:
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
