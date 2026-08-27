"""A minimal stand-in for `ollama serve` used by the integration test.

Implements just enough of the Ollama API (GET /api/tags, POST /api/generate)
for Prompt Enhancer's lifecycle to run end-to-end without the real daemon.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11434
STARTUP_DELAY = 0.3  # force the readiness-poll loop to actually spin


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request logging
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            self._send_json({"models": [{"name": "qwen3.5:9b"}]})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        if self.path == "/api/generate":
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.end_headers()  # HTTP/1.0: no content-length -> client reads to EOF
            for event in (
                {"response": "# Role\n", "done": False},
                {"response": "You are a fake-daemon persona.", "done": False},
                {"done": True},
            ):
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        else:
            self._send_json({"error": "not found"}, status=404)


if __name__ == "__main__":
    time.sleep(STARTUP_DELAY)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
