import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ntweak.models import Cancelled, ModelLibrary, download_file, inspect_model, safe_filename, validate_repo

CONTENT = b"sample-weights-0123456789" * 1024


class FileResponse(BytesIO):
    def __init__(self, offset):
        super().__init__(CONTENT[offset:])
        self.status = 206 if offset else 200
        self.headers = {"Content-Length": str(len(CONTENT) - offset)}
        if offset:
            self.headers["Content-Range"] = f"bytes {offset}-{len(CONTENT)-1}/{len(CONTENT)}"


class MemoryOpener:
    def __init__(self):
        self.requests = []
    def open(self, request, timeout=None):
        self.requests.append(dict(request.headers))
        offset = int(request.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0]) if not request.full_url.endswith("/ignore-range") else 0
        return FileResponse(offset)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.info = {"name": "weights.bin", "size": len(CONTENT), "sha256": hashlib.sha256(CONTENT).hexdigest()}
        self.opener = MemoryOpener()
        self.url = "https://huggingface.co/test/repo"
        self.patch = patch("urllib.request.build_opener", return_value=self.opener)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_download_resume_and_verified_cache(self):
        path = self.root / "weights.bin"
        path.with_name(path.name + ".ntweak-partial").write_bytes(CONTENT[:100])
        progress = []
        download_file(self.url + "/file", path, self.info, "", threading.Event(), progress.append)
        self.assertEqual(path.read_bytes(), CONTENT)
        self.assertEqual(progress[0], 100)
        self.assertEqual(self.opener.requests[-1]["Range"], "bytes=100-")
        before = len(self.opener.requests)
        download_file(self.url + "/file", path, self.info, "", threading.Event(), progress.append)
        self.assertEqual(len(self.opener.requests), before)

    def test_server_ignoring_range_restarts_cleanly(self):
        path = self.root / "weights.bin"
        path.with_name(path.name + ".ntweak-partial").write_bytes(CONTENT[:100])
        download_file(self.url + "/ignore-range", path, self.info, "", threading.Event(), lambda _: None)
        self.assertEqual(path.read_bytes(), CONTENT)

    def test_integrity_failure_is_not_registered_as_complete(self):
        path = self.root / "weights.bin"
        with self.assertRaisesRegex(ValueError, "Integrity"):
            download_file(self.url + "/file", path, {**self.info, "sha256": "0" * 64}, "", threading.Event(), lambda _: None)
        self.assertFalse(path.exists())

    def test_cancel_keeps_partial(self):
        path = self.root / "weights.bin"
        event = threading.Event()
        def cancel_after_first_chunk(count):
            if count:
                event.set()
        with self.assertRaises(Cancelled):
            download_file(self.url + "/file", path, self.info, "", event, cancel_after_first_chunk)
        self.assertFalse(path.exists())
        self.assertTrue(path.with_name(path.name + ".ntweak-partial").exists())

    def test_reject_traversal_and_untrusted_repos(self):
        for name in ["../secrets", "/etc/passwd", "foo/../../bar", "a\\b", ".ntweak-download.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_filename(name)
        for repo in ["https://evil/model", "../../tmp", "one", "a/b/c"]:
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                validate_repo(repo)

    def test_inspect_prefers_safetensors_and_pins_revision(self):
        upstream = {"sha": "a" * 40, "siblings": [{"rfilename": "config.json", "size": 100}, {"rfilename": "model.safetensors", "size": 200}, {"rfilename": "pytorch_model.bin", "size": 200}]}
        with patch("ntweak.models.request_json", return_value=upstream):
            result = inspect_model("Qwen/Qwen3.5-4B")
        self.assertEqual(result["commit"], "a" * 40)
        self.assertFalse(result["files"][2]["selected"])

    def test_local_registry_round_trip_and_shard_completeness(self):
        library = ModelLibrary(self.root / "data")
        model = self.root / "model"; model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}))
        (model / "shard-1.safetensors").write_bytes(b"x")
        self.assertFalse(library.complete_weights(model))
        (model / "shard-2.safetensors").write_bytes(b"x")
        self.assertTrue(library.complete_weights(model))
        item = library.register(model)
        self.assertEqual(library.get(item["id"])["path"], str(model.resolve()))


if __name__ == "__main__":
    unittest.main()
