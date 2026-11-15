from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    path = Path(path).resolve()
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous evidence unless explicitly requested.
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return path


class Result:
    def __init__(self, kind: str, version_name: str | None, run_dir: Path,
                 expected_frames: int | None = None):
        if expected_frames is not None and (isinstance(expected_frames, bool) or expected_frames < 1):
            raise ValueError("expected_frames must be a positive integer")
        self.value: dict[str, Any] = {
            "schema": "pcm.single-eval/v1",
            "collector": {"name": "pcm-single-eval", "version": __version__},
            "evaluation_type": kind,
            "version_name": version_name,
            "run_dir": str(run_dir.resolve()),
            "collected_at": utc_now(),
            "execution": {"state": "unknown"},
            "collection": {"status": "empty", "issues": []},
            "coverage": {"expected_frames": expected_frames, "observed": {}},
            "metrics": [], "artifacts": [], "provenance": {},
            "quality_assessment": "not_assessed",
        }
        self._metrics: dict[tuple, dict] = {}
        self._artifacts: set[str] = set()

    def issue(self, code: str, message: str, severity: str = "error") -> None:
        self.value["collection"]["issues"].append({
            "code": code, "severity": severity, "message": message,
        })

    def artifact(self, path: Path, role: str) -> None:
        path = path.resolve()
        if str(path) in self._artifacts:
            return
        self._artifacts.add(str(path))
        self.value["artifacts"].append({"path": str(path), "role": role,
                                        "size_bytes": path.stat().st_size,
                                        "sha256": digest(path)})

    def metric(self, name: str, value: Any, unit: str, source: Path,
               locator: str, **dimensions: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            return
        if not math.isfinite(value):
            self.issue("non_finite_metric", f"{source.name}:{locator}: {name} is not finite")
            return
        key = (name, tuple(sorted(dimensions.items())))
        self._metrics[key] = {"name": name, "value": value, "unit": unit,
                              "dimensions": dimensions,
                              "source": {"path": str(source.resolve()), "locator": locator}}

    def observe(self, name: str, count: Any) -> None:
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self.value["coverage"]["observed"][name] = count
        elif count is not None:
            self.issue("invalid_frame_count", f"{name}: expected nonnegative integer, got {count!r}")

    def read(self, path: Path, role: str, *, required: bool = False) -> Any:
        if not path.is_file():
            if required:
                self.issue("missing_artifact", f"Missing {path.name}")
            return None
        self.artifact(path, role)
        try:
            value = load_json(path)
            if not isinstance(value, dict):
                self.issue("invalid_json_shape", f"{path.name}: expected a JSON object")
                return None
            return value
        except (ValueError, OSError) as error:
            self.issue("invalid_json", f"{path.name}: {error}")
            return None

    def finish(self) -> dict:
        self.value["metrics"] = list(self._metrics.values())
        counts = self.value["coverage"]["observed"]
        expected = self.value["coverage"]["expected_frames"]
        if expected is not None:
            for name, count in counts.items():
                if name in ("board_metrics", "onnx_metrics", "tensor_comparison", "model_samples") and count != expected:
                    self.issue("coverage_mismatch", f"{name}: {count} frames, requested {expected}")
        issues = self.value["collection"]["issues"]
        self.value["collection"]["status"] = (
            "empty" if not self.value["metrics"] else
            "partial" if any(x["severity"] == "error" for x in issues) else "complete"
        )
        return self.value
