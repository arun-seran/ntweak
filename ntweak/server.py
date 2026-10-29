"""Loopback-only web app. The UI and model downloads need only Python's stdlib."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # inference never reaches out; downloader uses explicit urllib requests
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .engine import Runtime, dependencies
from .models import Cancelled, ModelLibrary, inspect_model
from .recipes import fetch_recipe, normalize_recipe

STATIC = Path(__file__).parent / "static"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / ".data"


def error_message(exc):
    # Never return signed CDN links, bearer credentials, or complete HTTP request data.
    if isinstance(exc, (ValueError, Cancelled)):
        return str(exc) or "Cancelled."
    if isinstance(exc, OSError):
        return f"Local file or network operation failed: {getattr(exc, 'reason', None) or getattr(exc, 'strerror', None) or type(exc).__name__}. Check your connection, free disk space and folder permissions."
    return f"{type(exc).__name__}: {str(exc)[:1200]}"


class App:
    def __init__(self, data_dir):
        self.library = ModelLibrary(data_dir)
        self.runtime = Runtime()
        self.jobs, self.chats = {}, {}
        self.lock = threading.Lock()

    def start_job(self, kind, work):
        with self.lock:
            if any(j["kind"] == kind and j["status"] not in ("done", "error", "cancelled") for j in self.jobs.values()):
                raise ValueError(f"A {kind} operation is already running.")
            jid = str(uuid.uuid4())
            event = threading.Event()
            job = {"id": jid, "kind": kind, "status": "starting", "started": time.time(), "cancel": event}
            # Bound history while retaining active operations.
            for old in list(self.jobs):
                if len(self.jobs) < 50:
                    break
                if self.jobs[old]["status"] in ("done", "error", "cancelled"):
                    del self.jobs[old]
            self.jobs[jid] = job
        def emit(update):
            with self.lock:
                job.update(update)
        def worker():
            try:
                result = work(event, emit)
                emit({"status": "done", "result": result})
            except Cancelled:
                emit({"status": "cancelled", "message": "Cancelled. Partial files are kept for resume."})
            except Exception as exc:
                emit({"status": "error", "error": error_message(exc)})
        threading.Thread(target=worker, daemon=True).start()
        return {"id": jid}

    def status(self):
        with self.lock:
            jobs = [{k: v for k, v in x.items() if k != "cancel"} for x in self.jobs.values()]
        return {"runtime": self.runtime.status(), "dependencies": dependencies(),
                "models": self.library.list(), "jobs": jobs,
                "data_dir": str(self.library.root.parent.resolve())}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        # Request bodies (chat text and tokens) are never logged.
        if self.path != "/api/status":
            super().log_message(format, *args)

    def allowed(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        origin = self.headers.get("Origin")
        return (self.headers.get("Host") in hosts
                and (not origin or (urlparse(origin).scheme == "http" and urlparse(origin).netloc in hosts))
                and self.headers.get("Sec-Fetch-Site") not in ("cross-site", "same-site"))

    def headers_for(self, status, mime, length=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def json(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.headers_for(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def body(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Expected an application/json request.")
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > 2 * 1024 * 1024:
            raise ValueError("Request must be between 1 byte and 2 MB.")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object.")
        return body

    def do_GET(self):
        if not self.allowed():
            self.close_connection = True
            return self.json({"error": "Only this app's local pages can access the server."}, 403)
        try:
            path = urlparse(self.path).path
            if path == "/api/status":
                return self.json(self.app.status())
            files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                     "/style.css": ("style.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            if path not in files:
                return self.json({"error": "Not found"}, 404)
            filename, mime = files[path]
            body = (STATIC / filename).read_bytes()
            self.headers_for(200, mime + "; charset=utf-8", len(body))
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.json({"error": error_message(exc)}, 500)

    def do_POST(self):
        if not self.allowed():
            self.close_connection = True
            return self.json({"error": "Only this app's local pages can access the server."}, 403)
        try:
            path, body = urlparse(self.path).path, self.body()
            if path == "/api/models/inspect":
                return self.json(inspect_model(body.get("repo"), body.get("revision", "main"), body.get("token", "") or os.environ.get("HF_TOKEN", "")))
            if path == "/api/models/download":
                # Snapshot token into the worker closure; never save in a job/status/manifest.
                repo, revision, filenames = body.get("repo"), body.get("revision", "main"), body.get("files")
                token = body.get("token", "") or os.environ.get("HF_TOKEN", "")
                return self.json(self.app.start_job("download", lambda cancel, emit: self.app.library.download(repo, revision, filenames, token, cancel, emit)), 202)
            if path == "/api/models/register":
                return self.json(self.app.library.register(body.get("path", "")))
            if path == "/api/models/load":
                info = self.app.library.get(body.get("id"))
                device = body.get("device", "auto")
                return self.json(self.app.start_job("load", lambda cancel, emit: self.app.runtime.load(info, device, emit)), 202)
            if path == "/api/models/unload":
                self.app.runtime.unload()
                return self.json({"ok": True})
            if path == "/api/jobs/cancel":
                with self.app.lock:
                    job = self.app.jobs.get(body.get("id"))
                    if not job or job["kind"] != "download":
                        raise ValueError("No cancellable download found.")
                    job["cancel"].set()
                return self.json({"ok": True})
            if path == "/api/recipes/import":
                return self.json(fetch_recipe(body["url"]) if "url" in body else normalize_recipe(body.get("recipe")))
            if path == "/api/chat/cancel":
                with self.app.lock:
                    event = self.app.chats.get(body.get("id"))
                    if event:
                        event.set()
                return self.json({"ok": True})
            if path == "/api/chat":
                return self.chat(body)
            return self.json({"error": "Not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError, TypeError) as exc:
            self.json({"error": error_message(exc)}, 400)
        except Exception as exc:
            self.json({"error": error_message(exc)}, 500)

    def chat(self, body):
        cid = body.get("id", "")
        if not isinstance(cid, str) or not re.fullmatch(r"[a-f0-9-]{16,64}", cid):
            raise ValueError("Provide a unique chat request ID.")
        event, events = threading.Event(), queue.Queue()
        with self.app.lock:
            if self.app.chats:
                raise ValueError("A response is already running. Stop it or wait for it to finish.")
            self.app.chats[cid] = event
        def work():
            try:
                self.app.runtime.chat(body, event, events.put)
            except Cancelled:
                events.put({"type": "done", "cancelled": True})
            except Exception as exc:
                events.put({"type": "error", "error": error_message(exc)})
            finally:
                events.put(None)
        threading.Thread(target=work, daemon=True).start()
        try:
            self.headers_for(200, "application/x-ndjson; charset=utf-8")
            while True:
                try:
                    item = events.get(timeout=5)
                except queue.Empty:
                    item = {"type": "heartbeat"}
                if item is None:
                    break
                self.wfile.write((json.dumps(item, allow_nan=False) + "\n").encode())
                self.wfile.flush()
        finally:
            event.set()
            # Keep the slot occupied until hooks have been removed by the model thread.
            if 'item' in locals() and item is None:
                with self.app.lock:
                    self.app.chats.pop(cid, None)
            else:
                def wait_done():
                    while events.get() is not None:
                        pass
                    with self.app.lock:
                        self.app.chats.pop(cid, None)
                threading.Thread(target=wait_done, daemon=True).start()


def create_server(port=8787, data_dir=DEFAULT_DATA):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = App(data_dir)
    return server


def main():
    parser = argparse.ArgumentParser(description="Ntweak — local neuron steering and chat")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("NTWEAK_DATA_DIR", DEFAULT_DATA)))
    parser.add_argument("--open", action="store_true", help="Open the app in your default browser after the server is ready")
    args = parser.parse_args()
    try:
        server = create_server(args.port, args.data_dir)
    except OSError as exc:
        parser.exit(1, f"Could not start the local server on port {args.port}: {exc}\nTry a different port with --port 8788.\n")
    print(f"\n  Ntweak is running at http://127.0.0.1:{server.server_port}\n  Models: {server.app.library.root.resolve()}\n  Press Ctrl+C to stop.\n", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(f"http://127.0.0.1:{server.server_port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
