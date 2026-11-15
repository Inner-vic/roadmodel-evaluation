from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .common import digest

REFERENCE_SHA256 = "185acb2c5c020e2de7ed73498bb2348854676ce9c178e3e306e6e45f6ab638ba"


def prepare_bundle(archive: str | Path, destination: str | Path,
                   expected_sha256: str = REFERENCE_SHA256) -> dict:
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    actual = digest(archive)
    if actual != expected_sha256.lower():
        raise ValueError("Offline bundle SHA256 does not match its declared source")
    if destination.exists():
        raise ValueError("Bundle destination already exists; choose a new directory")
    with zipfile.ZipFile(archive) as zipped:
        manifests = [n for n in zipped.namelist() if n.endswith("/MANIFEST.json")]
        if len(manifests) != 1:
            raise ValueError("Expected one bundle MANIFEST.json")
        prefix = str(PurePosixPath(manifests[0]).parent) + "/"
        manifest = json.loads(zipped.read(manifests[0]).decode("utf-8-sig"))
        files = {entry["path"]: entry for entry in manifest["files"]}
        for info in zipped.infolist():
            if info.is_dir():
                continue
            rel = PurePosixPath(info.filename.removeprefix(prefix))
            if (not info.filename.startswith(prefix) or rel.is_absolute() or ".." in rel.parts
                    or any(":" in p or "\\" in p for p in rel.parts)):
                raise ValueError("Unsafe bundle member")
            if rel.as_posix() != "MANIFEST.json" and rel.as_posix() not in files:
                raise ValueError(f"Unlisted bundle member: {rel}")
        destination.mkdir(parents=True)
        for info in zipped.infolist():
            if info.is_dir():
                continue
            relative = info.filename[len(prefix):]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
        for relative, entry in files.items():
            path = destination / relative
            if not path.is_file() or path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Bundle payload verification failed: {relative}")
    return {"bundle_sha256": actual, "directory": str(destination), "payload_files": len(files),
            "models": [str(destination / p) for p in files if p.startswith("examples/assets/")]}
