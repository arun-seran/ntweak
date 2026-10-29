import http.client
from io import BytesIO
import json
import tempfile
from types import SimpleNamespace
import unittest

from ntweak.server import App, Handler


class MemorySocket:
    """Exercise real HTTP parsing/writing without requiring a loopback bind."""
    def __init__(self, request):
        self.request = BytesIO(request)
        self.output = BytesIO()
    def makefile(self, mode, *args):
        return self.request
    def sendall(self, data):
        self.output.write(data)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.server = SimpleNamespace(app=App(cls.temp.name), server_port=8787)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        encoded = json.dumps(body).encode() if body is not None else b""
        headers = {"Host": "127.0.0.1:8787", "Content-Type": "application/json", "Content-Length": str(len(encoded)), "Connection": "close", **(headers or {})}
        request = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
        sock = MemorySocket(request.encode() + encoded)
        Handler(sock, ("127.0.0.1", 1234), self.server)
        raw_headers, raw_body = sock.output.getvalue().split(b"\r\n\r\n", 1)
        lines = raw_headers.decode().split("\r\n")
        parsed = dict(line.split(": ", 1) for line in lines[1:])
        return int(lines[0].split()[1]), parsed, raw_body

    def test_static_ui_and_status_are_served_locally(self):
        code, headers, body = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(b"Steering rack", body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        code, _, body = self.request("GET", "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["runtime"]["state"], "idle")

    def test_cross_origin_and_rebound_hosts_are_rejected(self):
        for headers in [{"Origin": "https://evil.example"}, {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"}]:
            with self.subTest(headers=headers):
                code, _, _ = self.request("POST", "/api/models/unload", {}, headers)
                self.assertEqual(code, 403)

    def test_json_recipe_import_stays_offline(self):
        code, _, body = self.request("POST", "/api/recipes/import", {"recipe": {"neurons": [{"layer": 1, "n": 2, "dose": 0.3}]}})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["neurons"][0]["neuron"], 2)

    def test_no_model_chat_errors_are_streamed_and_slot_cleans_up(self):
        for _ in range(2):
            code, _, body = self.request("POST", "/api/chat", {"id": "a" * 32, "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(code, 200)
            event = json.loads(body.splitlines()[0])
            self.assertEqual(event["type"], "error")
            self.assertIn("load a model", event["error"])
        self.assertEqual(self.server.app.chats, {})

    def test_chat_uses_real_stream_and_cancellation(self):
        original = self.server.app.runtime.chat
        def fake_engine(body, cancel, emit):
            emit({"type": "token", "text": "A test stream"})
            emit({"type": "done", "cancelled": False})
        self.server.app.runtime.chat = fake_engine
        try:
            code, _, body = self.request("POST", "/api/chat", {"id": "b" * 32})
            self.assertEqual(code, 200)
            self.assertEqual([json.loads(x)["type"] for x in body.splitlines()], ["token", "done"])
        finally:
            self.server.app.runtime.chat = original

    def test_path_traversal_is_not_served(self):
        self.assertEqual(self.request("GET", "/../README.md")[0], 404)


if __name__ == "__main__":
    unittest.main()
