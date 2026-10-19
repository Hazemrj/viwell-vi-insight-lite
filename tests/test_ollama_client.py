"""Checks that OllamaClient speaks Ollama's /api/chat protocol, using a tiny fake Ollama server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.vi.llm import OllamaClient

GOOD = {"member_id": "M0001", "insight": "Your steps dipped a little this week, so let's ease back in.",
        "evidence": ["steps_7d: 100 vs baseline 120"], "actions": ["Walk after lunch.", "Keep a regular bedtime."],
        "tone": "supportive", "safety_flag": False}
seen = {}


class FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.update(path=self.path, body=body)
        payload = json.dumps({"message": {"role": "assistant", "content": json.dumps(GOOD)}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def test_ollama_request_and_response_shape():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = OllamaClient(url=f"http://127.0.0.1:{server.server_port}", model="qwen2.5:3b")
    text = client.chat([{"role": "user", "content": "hi"}])
    server.shutdown()
    assert json.loads(text)["member_id"] == "M0001"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:3b" and seen["body"]["stream"] is False and seen["body"]["format"] == "json"
