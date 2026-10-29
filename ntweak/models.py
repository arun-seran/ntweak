"""Hugging Face downloads with pinned revisions, resumable files and byte progress."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class Cancelled(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def validate_repo(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise ValueError("Enter a Hugging Face model ID, such as Qwen/Qwen3.5-4B.")
    return value


def safe_filename(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in ("..", ".") for x in value.split("/")) or "\\" in value or value.startswith(".ntweak"):
        raise ValueError("Invalid model filename.")
    return value


def request_json(url, token=""):
    headers = {"User-Agent": "Ntweak/0.1", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("This repository needs access. Accept its terms on Hugging Face and provide an HF_TOKEN with read access.") from None
        if exc.code == 404:
            raise ValueError("Model or revision not found on Hugging Face.") from None
        raise


def inspect_model(repo, revision="main", token=""):
    validate_repo(repo)
    if not isinstance(revision, str) or not revision or len(revision) > 200:
        raise ValueError("Invalid revision.")
    data = request_json(f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true", token)
    commit = data.get("sha", "")
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", commit):
        raise ValueError("Hugging Face did not return a pinned commit.")
    siblings = data.get("siblings", [])
    has_safe = any(x["rfilename"].endswith(".safetensors") for x in siblings)
    files = []
    for x in siblings:
        name = safe_filename(x["rfilename"])
        lfs = x.get("lfs") or {}
        size = x.get("size", lfs.get("size", 0)) or 0
        default = (name.endswith((".json", ".safetensors", ".model", ".tiktoken", ".txt", ".jinja"))
                   or (name.endswith(".bin") and not has_safe))
        files.append({"name": name, "size": size, "sha256": lfs.get("sha256"),
                      "blob_id": x.get("blobId"), "selected": default})
    return {"repo": repo, "revision": revision, "commit": commit, "files": files,
            "bytes": sum(x["size"] for x in files), "pipeline": data.get("pipeline_tag", "unknown"),
            "gated": bool(data.get("gated", False))}


def verify_file(path, info):
    if info["size"] and path.stat().st_size != info["size"]:
        return False
    digest = info.get("sha256")
    if digest:
        h = hashlib.sha256()
    elif info.get("blob_id"):
        digest = info["blob_id"]
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
    else:
        return True
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == digest


def download_file(url, path, info, token, cancel, progress):
    """Resume only at a confirmed Content-Range; keep partials on cancellation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and verify_file(path, info):
        progress(path.stat().st_size)
        return
    partial = path.with_name(path.name + ".ntweak-partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if info["size"] and offset >= info["size"]:
        if offset == info["size"] and verify_file(partial, info):
            partial.replace(path)
            progress(offset)
            return
        offset = 0
    headers = {"User-Agent": "Ntweak/0.1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if cancel.is_set():
        raise Cancelled()
    # urllib removes neither arbitrary headers nor Bearer credentials on redirect.
    # HF uses signed CDN URLs; send tokens only to HF, never the CDN.
    from urllib.request import HTTPRedirectHandler, build_opener
    class Redirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            from urllib.parse import urlparse
            redirected = super().redirect_request(req, fp, code, msg, hdrs, newurl)
            if redirected and urlparse(newurl).hostname != "huggingface.co":
                redirected.remove_header("Authorization")
            return redirected
    with build_opener(Redirect()).open(Request(url, headers=headers), timeout=30) as response:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("The server returned an invalid resume range. Retry the download.")
        elif response.status == 206:
            if not response.headers.get("Content-Range", "").startswith("bytes 0-"):
                raise ValueError("Unexpected partial response.")
        else:
            offset = 0
        with partial.open("ab" if offset else "wb") as stream:
            progress(offset)
            while True:
                if cancel.is_set():
                    raise Cancelled()
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                offset += len(chunk)
                progress(offset)
    if not verify_file(partial, info):
        partial.unlink(missing_ok=True)
        raise ValueError(f"Integrity check failed for {info['name']}; retry to download it again.")
    partial.replace(path)


class ModelLibrary:
    def __init__(self, data_dir):
        self.root = Path(data_dir) / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = Path(data_dir) / "models.json"
        self.lock = threading.Lock()

    def _registered(self):
        if self.registry.exists():
            return json.loads(self.registry.read_text())
        return []

    def list(self):
        with self.lock:
            result = self._registered()
        known = {x["path"] for x in result}
        # Reuse standard HF snapshots without copying multi-GB weights.
        roots = {Path.home() / ".cache/huggingface/hub"}
        if os.environ.get("HF_HOME"):
            roots.add(Path(os.environ["HF_HOME"]) / "hub")
        if os.environ.get("HF_HUB_CACHE"):
            roots.add(Path(os.environ["HF_HUB_CACHE"]))
        for root in roots:
            if not root.is_dir():
                continue
            for config in root.glob("models--*/snapshots/*/config.json"):
                p = config.parent
                if str(p) in known or not any(p.glob("*.safetensors")):
                    continue
                # An interrupted HF snapshot can contain only some shards.
                if not self.complete_weights(p):
                    continue
                repo = p.parent.parent.name.removeprefix("models--").replace("--", "/", 1)
                result.append({"id": self.key(repo, p.name), "repo": repo, "revision": p.name,
                               "path": str(p), "source": "Hugging Face cache"})
                known.add(str(p))
        return [dict(x, available=Path(x["path"]).is_dir()) for x in result]

    @staticmethod
    def complete_weights(path):
        indexes = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model.bin.index.json"))
        if indexes:
            try:
                return all((path / name).is_file() for index in indexes
                           for name in set(json.loads(index.read_text())["weight_map"].values()))
            except (OSError, ValueError, KeyError):
                return False
        return any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file()

    @staticmethod
    def key(repo, revision):
        return hashlib.sha256(f"{repo}@{revision}".encode()).hexdigest()[:20]

    def register(self, path, repo=None, revision="local", source="Local folder"):
        path = Path(path).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("Choose a local model directory containing config.json.")
        return self._save({"id": self.key(str(path), revision), "repo": repo or path.name,
                           "revision": revision, "path": str(path), "source": source})

    def _save(self, item):
        with self.lock:
            rows = [x for x in self._registered() if x["path"] != item["path"]]
            atomic_json(self.registry, rows + [item])
        return item

    def get(self, model_id):
        item = next((x for x in self.list() if x["id"] == model_id), None)
        if not item or not item["available"]:
            raise ValueError("Select an available local model first.")
        return item

    def download(self, repo, revision, filenames, token, cancel, emit):
        info = inspect_model(repo, revision, token)
        allowed = {x["name"]: x for x in info["files"]}
        if not isinstance(filenames, list) or not filenames or any(x not in allowed for x in filenames):
            raise ValueError("Select at least one valid file from this repository.")
        selected = [allowed[x] for x in dict.fromkeys(filenames)]
        total = sum(x["size"] for x in selected)
        folder = self.root / self.key(repo, info["commit"])
        folder.mkdir(exist_ok=True)
        remaining = sum(max(0, x["size"] - ((folder / x["name"]).stat().st_size if (folder / x["name"]).exists() else 0)) for x in selected)
        if shutil.disk_usage(folder).free < remaining + 128 * 1024**2:
            raise ValueError("Not enough free disk space for the selected files.")
        done = 0
        for index, file in enumerate(selected):
            if cancel.is_set():
                raise Cancelled()
            def progress(count):
                emit({"status": "downloading", "file": file["name"], "completed_bytes": done + count,
                      "total_bytes": total, "files_done": index, "files_total": len(selected)})
            url = f"https://huggingface.co/{repo}/resolve/{info['commit']}/{quote(file['name'], safe='/')}"
            download_file(url, folder / file["name"], file, token, cancel, progress)
            done += (folder / file["name"]).stat().st_size
        item = {"id": self.key(str(folder.resolve()), info["commit"]), "repo": repo,
                "revision": info["commit"], "path": str(folder.resolve()), "source": "Downloaded in Ntweak"}
        self._save(item)
        atomic_json(folder / ".ntweak-download.json", {"repo": repo, "commit": info["commit"], "files": filenames})
        return item
