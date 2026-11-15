"""LPAI read-only collection through the installed lpai-agent CLI."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import save_json, utc_now
from .parsers import collect_model

# Include class rows, section headers, final summaries and coverage lines.
METRIC_FILTER = (
    r"per-class stats|iou=|mIoU:|iou2:|\[core\]|Classifier Core|macro_recall=|"
    r"class_[0-9]|acc=|Total samples|[Ee]val.*samples|[Nn]umber of.*samples|"
    r"[Ee]val dataset|[Vv]al dataset|\[EvalSubsample\]"
)


def configured() -> bool:
    if os.environ.get("LPAI_API_KEY"):
        return True
    path = Path.home() / ".config/lpai-agent/config"
    return path.is_file() and bool(re.search(r"^\s*api_key\s*=\s*\S+", path.read_text(), re.M))


def request(args: list[str], *, cli: str = "lpai-agent", preserve_envelope: bool = False) -> Any:
    if not configured():
        raise ValueError("LPAI authentication is not configured in its standard environment/config")
    proc = subprocess.run([cli, *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=150, shell=False,
                          **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
    if proc.returncode:
        # Do not persist an unfiltered platform response: it may contain configuration.
        raise ValueError(f"lpai-agent {' '.join(args[:2])} failed (exit {proc.returncode})")
    try:
        value = json.loads(proc.stdout)
    except ValueError as error:
        raise ValueError("lpai-agent returned non-JSON output") from error
    if isinstance(value, dict) and value.get("status") in ("error", "failed"):
        raise ValueError(f"lpai-agent {' '.join(args[:2])} returned an error")
    if preserve_envelope:
        return value
    return value.get("data", value) if isinstance(value, dict) else value


def timestamp(value: Any) -> datetime:
    if isinstance(value, dict):
        value = value.get("time")
    if not isinstance(value, str) or not value:
        raise ValueError("Task/pod does not provide a usable log timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def select_pod(pods: Any, name: str | None = None) -> dict:
    if isinstance(pods, dict):
        pods = pods.get("items", pods.get("pods", []))
    if not isinstance(pods, list):
        raise ValueError("Unrecognized tasks pods response")
    candidates = [p for p in pods if isinstance(p, dict) and (
        p.get("name") == name if name else
        p.get("step") == "task" and p.get("pod_type") == "user" and
        re.search(r"(?:master|worker)-0$", p.get("name", ""))
    )]
    masters = [p for p in candidates if re.search(r"master-0$", p.get("name", ""))]
    if not name and masters:
        candidates = masters
    if len(candidates) != 1:
        raise ValueError("A unique evaluation master pod could not be selected; specify --pod")
    return candidates[0]


def extract_lines(data: Any) -> list[tuple[int, str]]:
    if not isinstance(data, dict) or data.get("resultType") != "streams":
        raise ValueError("Unrecognized LPAI log response (expected Loki streams)")
    rows = []
    for stream in data.get("result", []):
        for ts, line in stream.get("values", []):
            rows.append((int(ts), str(line)))
    return sorted(set(rows))


def collect_task(task_uid: str, output_dir: str | Path, *, version_name: str | None = None,
                 pod_name: str | None = None, limit: int = 5000, cli: str = "lpai-agent") -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9-]+", task_uid):
        raise ValueError("Invalid task UID")
    if not 1 <= limit <= 10000:
        raise ValueError("Log limit must be between 1 and 10000")
    root = Path(output_dir).resolve()
    if root.exists():
        raise ValueError("Choose a new collection directory; prior evidence is preserved")
    task = request(["tasks", "get", "--uid", task_uid], cli=cli)
    if not isinstance(task, dict):
        raise ValueError("Unrecognized tasks get response")
    pod = select_pod(request(["tasks", "pods", "--uid", task_uid], cli=cli), pod_name)
    start = timestamp(pod.get("create_timestamp") or task.get("start_timestamp"))
    end = timestamp(pod.get("finish_timestamp") or task.get("update_timestamp")) + timedelta(seconds=1)
    ns, cluster = task.get("namespace"), task.get("cluster")
    if not ns or not cluster:
        raise ValueError("Task namespace/cluster unavailable")
    logs = request(["logs", "pod", "--namespace", ns, "--cluster", cluster,
                    "--pod", pod["name"], "--start", start.isoformat(), "--end", end.isoformat(),
                    "--grep", METRIC_FILTER, "--direction", "backward", "--limit", str(limit)], cli=cli)
    lines = extract_lines(logs)
    root.mkdir(parents=True)
    (root / "model.log").write_text("\n".join(line for _, line in lines) + "\n", encoding="utf-8")
    status = str(task.get("status", "unknown")).lower()
    state = "succeeded" if status == "succeeded" else "failed" if status in ("failed", "stopped") else status
    # Only retain selected nonsecret task fields; do not serialize full task/env/input configuration.
    evidence = {"source": "lpai_task", "task_uid": task_uid, "state": state,
                "task_name": task.get("name"), "namespace": ns, "cluster": cluster,
                "pod": pod["name"], "collected_at": utc_now(),
                "log_window": {"start": start.isoformat(), "end": end.isoformat()},
                "log_filter": METRIC_FILTER, "log_limit": limit, "returned_log_lines": len(lines),
                "historical_result": True}
    save_json(root / "execution_receipt.json", evidence)
    result = collect_model(root, version_name)
    result["provenance"]["lpai_task"] = evidence
    if state not in ("succeeded", "failed"):
        result["collection"]["issues"].append({"severity": "error", "code": "task_not_finished",
                                                   "message": "Task has not reached a successful or failed terminal state"})
        if result["metrics"]:
            result["collection"]["status"] = "partial"
    if len(lines) >= limit:
        result["collection"]["issues"].append({"severity": "error", "code": "log_limit_reached",
                                                   "message": "Filtered logs may be truncated; narrow the time range or collect files"})
        if result["metrics"]:
            result["collection"]["status"] = "partial"
    save_json(root / "result.json", result)
    return result
