from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .common import Result, digest

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
N = f"(?:{NUMBER})"


def pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


ROW = re.compile(
    rf"\b(?P<id>\d+|NN)\s+(?P<label>[\w.-]+)\s*\|\s*"
    rf"iou=\s*(?P<iou>{N})%\s+prec=\s*(?P<precision>{N})%\s+recall=\s*(?P<recall>{N})%"
    rf"\s*\|\s*seen=\s*(?P<seen>\d+)\s+correct=\s*(?P<correct>\d+)\s+positive=\s*(?P<positive>\d+)", re.I)


def parse_model_log(path: Path, result: Result) -> None:
    """Parse the final EVAL sections, preserving original values and units.

    Repeated metrics overwrite their earlier occurrence in the same source run.
    TRAIN rows and rolling windows are never substituted for final EVAL rows.
    """
    result.artifact(path, "model_log")
    section = None
    vision_task = None
    vision_map_section = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        locator = f"line:{line_number}"
        if 'Vision Head mAP Results' in line:vision_map_section=True
        header = re.search(r"\[(EVAL[\w]*)\]\s+per-class stats", line)
        if header:
            section = header.group(1)
        elif re.search(r"\[(?:TRAIN|VAL)[\w]*\]\s+per-class stats", line):
            section = None
        row = ROW.search(line)
        if row and section:
            dims = {"head": "occ", "scope": section, "class_name": row["label"], "class_id": row["id"]}
            for key in ("iou", "precision", "recall"):
                result.metric(f"occ.{key}", float(row[key]), "percent", path, locator, **dims)
            for key in ("seen", "correct", "positive"):
                result.metric(f"occ.{key}", int(row[key]), "count", path, locator, **dims)
        for pattern, name in ((rf"(?<![\w_])mIoU:\s*({N})", "occ.miou"),
                              (rf"(?<![\w_])iou2:\s*({N})", "occ.iou2"),
                              (rf"\bpillar_mIoU:\s*({N})", "pillar.miou"),
                              (rf"\bpillar_iou2:\s*({N})", "pillar.iou2")):
            match = re.search(pattern, line, re.I)
            if match:
                result.metric(name, float(match[1]), "percent", path, locator, backend="model")
        match = re.search(rf'DS-IoU:\s*({N})%\s+seen=(\d+)\s+correct=(\d+)\s+positive=(\d+)', line)
        if match:
            result.metric('drivable.iou', float(match[1]), 'percent', path, locator)
            for index, key in enumerate(('seen','correct','positive'), 2):
                result.metric('drivable.'+key, int(match[index]), 'count', path, locator)
        if 'count=' in line and 'min=' in line and 'max=' in line and 'mean=' in line and 'std=' in line:
            for key in ('count','min','max','mean','std'):
                match = re.search(rf'\b{key}=({N})', line)
                if match:
                    result.metric('gaussian.opacity.'+key, int(match[1]) if key=='count' else float(match[1]),
                                  'count' if key=='count' else 'ratio', path, locator)
        for key in ('near_zero_ratio','near_one_ratio'):
            match = re.search(rf'{key}\([^)]*\):\s*({N})%', line)
            if match:result.metric('gaussian.opacity.'+key,float(match[1]),'percent',path,locator)
        match = re.search(rf"\[core\]\s+avg mAP:\s*({N})", line, re.I)
        if match:
            result.metric("vision.avg_map", float(match[1]), "ratio", path, locator)
        match = re.search(rf"\[core\].*\(([A-Za-z_]\w*)\):\s*({N})\s*$", line)
        if match:
            result.metric("vision.map", float(match[2]), "ratio", path, locator, task=match[1])
        if vision_map_section and 'Total samples used:' not in line:
            match=re.search(rf'\b([A-Za-z_]\w*):\s*({N})\s*$',line)
            if match and match[1] not in ('mAP','INFO'):
                result.metric('vision.map',float(match[2]),'ratio',path,locator,task=match[1])
        match=re.search(rf'\(([A-Za-z_]\w*)\):\s*(\d+)/(\d+)\s*=\s*({N})%',line)
        if match:
            result.metric('vision.correct',int(match[2]),'count',path,locator,task=match[1])
            result.metric('vision.total',int(match[3]),'count',path,locator,task=match[1])
            result.metric('vision.accuracy',float(match[4]),'percent',path,locator,task=match[1])
        match = re.search(r"\(([A-Za-z_]\w*)\):\s*(?:acc|mAP)=", line)
        if match:
            vision_task = match[1]
            for key, name, unit in (("acc", "accuracy", "percent"), ("mAP", "map", "ratio"),
                                    ("macro_recall", "macro_recall", "percent")):
                val = re.search(rf"\b{key}=\s*({N})", line)
                if val:
                    result.metric(f"vision.{name}", float(val[1]), unit, path, locator, task=vision_task)
            major = re.search(r"major=(\d+)", line)
            if major:
                result.metric("vision.major_classes", int(major[1]), "count", path, locator, task=vision_task)
        match = re.search(rf"class_(\d+)(.*?)\s*:\s*n=(\d+)\s*\(({N})%\),\s*recall=({N})%,\s*AP=({N})", line)
        if match and vision_task:
            dims = {"task": vision_task, "class_id": match[1], "included_in_map": "少数类" not in match[2]}
            for name, value, unit in (("support", int(match[3]), "count"),
                                       ("support_percent", float(match[4]), "percent"),
                                       ("recall", float(match[5]), "percent"), ("ap", float(match[6]), "ratio")):
                result.metric(f"vision.class.{name}", value, unit, path, locator, **dims)
        match = re.search(r"Total samples used:\s*(\d+)", line)
        if match:
            vision_map_section=False
            result.observe("vision_samples", int(match[1]))
        match = re.search(r"\[EvalSubsample\]\s*\d+\s*(?:→|->)\s*(\d+)\s+samples", line)
        if match:
            result.observe("model_samples", int(match[1]))
        match = re.search(r"(?:Number of|num(?:ber)?\s+of|Total)\s+(?:eval(?:uation)?\s+)?samples\s*[:=]\s*(\d+)", line, re.I)
        if match:
            result.observe("model_samples", int(match[1]))


def parse_scheme(path: Path, data: dict, result: Result) -> None:
    meta = data.get("meta", {})
    if isinstance(meta, dict):
        result.value["provenance"]["scheme_meta"] = meta
        result.observe("scheme_samples", meta.get("kept_samples"))
        if meta.get("skipped_samples", 0):
            result.issue("scheme_skipped_samples", f"Postprocessing skipped {meta['skipped_samples']} samples", "warning")
    if isinstance(data.get("label_space"), dict):
        result.value["provenance"]["label_space"] = data["label_space"]

    def walk(value: Any, parts: list[str]) -> None:
        if isinstance(value, dict):
            if value.get("enabled") is False:
                result.value["provenance"].setdefault("disabled_metric_groups", []).append("/".join(parts))
                return
            for key, child in value.items():
                if key not in ("notes", "matrix", "labels", "enabled", "ready", "required_inputs"):
                    walk(child, parts + [str(key)])
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            leaf = parts[-1]
            unit = "count" if leaf in ("count", "tp", "fp", "fn", "gt_support", "pred_support") else (
                "meter" if leaf.endswith("_m") else "centimeter" if leaf.endswith("_cm") else "source_defined")
            result.metric("scheme." + ".".join(parts), value, unit, path, "/" + "/".join(map(pointer_segment, parts)))

    for key, value in data.items():
        if key not in ("meta", "notes", "label_space", "bev_compare_placeholder"):
            walk(value, [key])


def collect_model(run_dir: str | Path, version_name: str | None = None, *,
                  expected_frames: int | None = None, log_file: str | Path | None = None,
                  scheme_file: str | Path | None = None) -> dict:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    if (root/'trajectory_metrics.json').is_file() and not log_file and not scheme_file:
        from .model_heads import collect_trajectory
        from .common import load_json
        count=expected_frames if expected_frames is not None else load_json(root/'trajectory_metrics.json')['num_frames']
        return collect_trajectory(root,version_name,count)
    result = Result("model_accuracy", version_name, root, expected_frames)
    logs = [Path(log_file).resolve()] if log_file else sorted(set(root.glob("*.log")) | set(root.glob("logs/*.log")))
    if len(logs) > 1:
        result.issue("ambiguous_log", "Multiple logs found; use --log-file to identify one evaluation")
    elif logs and logs[0].is_file():
        parse_model_log(logs[0], result)
    elif log_file:
        result.issue("missing_log", str(log_file))
    schemes = [Path(scheme_file).resolve()] if scheme_file else [p for p in (
        root / "eval_scheme_report.json", root / "vis/eval_scheme_report.json") if p.is_file()]
    if len(schemes) > 1:
        result.issue("ambiguous_scheme", "Multiple scheme reports found; use --scheme-file")
    elif schemes:
        data = result.read(schemes[0], "scheme_report", required=True)
        if isinstance(data, dict):
            parse_scheme(schemes[0], data, result)
    receipt = result.read(root / "execution_receipt.json", "execution_receipt")
    if isinstance(receipt, dict):
        result.value["execution"] = {k: receipt[k] for k in ("state", "exit_code", "task_uid", "source") if k in receipt}
        if receipt.get("source") == "lpai_task":
            result.value["provenance"]["lpai_task"] = receipt
        elif receipt.get('schema') == 'roadmodel.model-execution/v1':
            result.value['provenance']['native_run'] = receipt
            coverage = result.read(root/'coverage_receipt.json', 'native_coverage')
            preflight = result.read(root/'model-preflight.json', 'model_preflight')
            if isinstance(preflight,dict):result.value['provenance']['model_preflight']=preflight
            if isinstance(coverage,dict):
                valid = (coverage.get('state')=='verified' and coverage.get('run_id')==receipt.get('run_id')
                         and coverage.get('selected_frames')==receipt.get('selected_frames')
                         and coverage.get('observed_frames')==receipt.get('expected_frames')
                         and coverage.get('console_sha256')==digest(root/'eval-console.log')
                         and coverage.get('loader_preflight_sha256')==digest(root/'model-preflight.json'))
                if not valid:raise ValueError('Native coverage receipt does not match this execution')
                result.value['provenance']['native_coverage']=coverage
                result.observe('model_samples',coverage['observed_frames'])
    if result.value["execution"].get("state") == "failed":
        result.issue("execution_failed", "The evaluation execution receipt reports failure")
    if not logs and not schemes:
        result.issue("missing_artifact", "No model evaluation log or eval_scheme_report.json found")
    if result.value["coverage"]["observed"].get("model_samples") is None:
        result.issue("unknown_model_coverage", "The source log does not establish model sample count", "warning")
    return result.finish()


def collect_compile(run_dir: str | Path, version_name: str | None = None, *,
                    expected_frames: int | None = None, gt_dir: str | Path | None = None) -> dict:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    result = Result("compile_alignment", version_name, root, expected_frames)
    state = result.read(root / "state.json", "bench_state", required=True)
    tensor_only = isinstance(state, dict) and state.get("config", {}).get("skip_metrics") is True
    gt_receipt = None
    metrics_root = root
    if gt_dir is not None:
        metrics_root = Path(gt_dir).resolve()
        gt_receipt = result.read(metrics_root / 'gt_receipt.json', 'gt_adaptation', required=True)
        source_receipt = result.read(root / 'execution_receipt.json', 'source_execution_receipt', required=True)
        if not isinstance(gt_receipt, dict) or not isinstance(source_receipt, dict):
            raise ValueError('Validated GT adaptation and source run receipts are required')
        if (gt_receipt.get('schema') != 'pcm.gt-adaptation/v1' or gt_receipt.get('state') != 'validated'
                or gt_receipt.get('source_run_id') != source_receipt.get('run_id')
                or gt_receipt.get('selected_frames') != source_receipt.get('selected_frames')
                or gt_receipt.get('num_frames') != len(source_receipt.get('selected_frames', []))
                or gt_receipt.get('num_frames') != len(gt_receipt.get('frames', []))):
            raise ValueError('GT adaptation does not describe the exact source prediction inventory')
        for filename, field in [('execution_receipt.json','source_execution_receipt_sha256'),
                                ('cosine_summary.json','source_cosine_summary_sha256'),
                                ('output_integrity.json','source_output_integrity_sha256')]:
            if digest(root / filename) != gt_receipt.get(field):
                raise ValueError('GT adaptation source identity mismatch: ' + filename)
        for backend in ('board','onnx'):
            if digest(metrics_root / f'metrics_{backend}.json') != gt_receipt.get(backend+'_metrics_sha256'):
                raise ValueError('GT metric content differs from its receipt')
        source_integrity = result.read(root / 'output_integrity.json', 'source_output_integrity', required=True)
        if not isinstance(source_integrity, dict) or source_integrity.get('state') != 'fresh_outputs_checked':
            raise ValueError('GT results require verified fresh source predictions')
        tensor_only = False
        result.value['provenance']['gt_adaptation'] = gt_receipt
    result.value["provenance"]["metric_scope"] = "tensor_alignment" if tensor_only else "tensor_alignment_and_gt"
    counts = {}
    for backend in ("board", "onnx"):
        if tensor_only:
            continue
        path = metrics_root / f"metrics_{backend}.json"
        data = result.read(path, f"{backend}_metrics", required=True)
        if not isinstance(data, dict):
            continue
        result.observe(f"{backend}_metrics", data.get("num_frames"))
        counts[backend] = data.get("num_frames")
        for key in ("occ_iou", "present_miou"):
            if not isinstance(data.get(key), (int, float)) or isinstance(data.get(key), bool):
                result.issue("missing_backend_metric", f"{backend}: missing numeric {key}")
        for key in ("present_miou", "occ_iou", "drivable_iou", "pillar_miou", "pillar_nonempty_iou", "bumper_iou", "manhole_iou"):
            result.metric(key, data.get(key), "percent", path, "/" + key, backend=backend)
        if isinstance(data.get('class_support'), list):
            result.value['provenance'].setdefault('gt_class_support', {})[backend] = data['class_support']
        if data.get('drivable_definition'):
            result.value['provenance'].setdefault('drivable_definition', {})[backend] = data['drivable_definition']
        # Legacy miou treats absent classes as 100%; retain under an explicit name.
        result.metric("legacy_miou", data.get("miou"), "percent", path, "/miou", backend=backend)
        for i, row in enumerate(data.get("present_class_ious", [])):
            if not isinstance(row, dict):
                continue
            dims = {"backend": backend, "class_id": str(row.get("class_id")), "class_name": str(row.get("name"))}
            for key in ("iou", "seen", "correct", "positive"):
                result.metric("occ.class." + key, row.get(key), "percent" if key == "iou" else "count",
                              path, f"/present_class_ious/{i}/{key}", **dims)
    path = root / "cosine_summary.json"
    data = result.read(path, "tensor_comparison", required=True)
    if isinstance(data, dict):
        result.observe("tensor_comparison", data.get("num_frames"))
        counts["cosine"] = data.get("num_frames")
        for key, name in (("mean_cosine_all", "tensor.frame_mean_cosine.mean"),
                          ("min_cosine_all", "tensor.frame_mean_cosine.min")):
            if not isinstance(data.get(key), (int, float)) or isinstance(data.get(key), bool):
                result.issue("missing_cosine_statistic", f"Missing numeric {key}")
            result.metric(name, data.get(key), "ratio", path, "/" + key)
        tensors = data.get("tensors", {})
        if not isinstance(tensors, dict) or not tensors:
            result.issue("empty_tensor_comparison", "No output tensors were compared")
        else:
            result.value["coverage"]["tensor_frames"] = {}
            for name, row in tensors.items():
                if not isinstance(row, dict):
                    result.issue("invalid_tensor_statistic", f"{name}: expected a statistics object")
                    continue
                for key in ("mean_cosine", "min_cosine"):
                    if not isinstance(row.get(key), (int, float)) or isinstance(row.get(key), bool):
                        result.issue("missing_tensor_statistic", f"{name}: missing numeric {key}")
                    result.metric("tensor." + key, row.get(key), "ratio", path,
                                  f"/tensors/{pointer_segment(name)}/{key}", output=name)
                result.value["coverage"]["tensor_frames"][name] = row.get("num_frames")
                if row.get("num_frames") != data.get("num_frames"):
                    result.issue("tensor_coverage_mismatch", f"{name}: tensor frames differ from comparison frames")
        frames = data.get("frames", [])
        if len(frames) != data.get("num_frames"):
            result.issue("comparison_frame_inventory_mismatch", "Per-frame comparison inventory does not match num_frames")
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for output in frame.get("outputs", []):
                if output.get("status") != "ok":
                    result.issue("missing_tensor", f"{frame.get('frame_id')}: {output.get('onnx_output_name')} ({output.get('status')})")
                for key in ("cosine", "max_abs_diff", "mean_abs_diff"):
                    value = output.get(key)
                    if isinstance(value, (float, int)):
                        if not math.isfinite(value):
                            result.issue("non_finite_tensor", f"{frame.get('frame_id')}: {output.get('onnx_output_name')}/{key}")
    if any(not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in counts.values()):
        result.issue("empty_metrics", "Each result must contain a positive num_frames")
    if len(set(c for c in counts.values() if isinstance(c, int))) > 1:
        result.issue("backend_coverage_mismatch", f"Metrics describe different frame counts: {counts}")
    if isinstance(state, dict):
        summary = state.get("summary", {})
        total = state.get("total_frames")
        result.value["coverage"]["state"] = {"exported_frames": total, **summary}
        result.value["provenance"]["script_config"] = state.get("config", {})
        result.value["execution"]["state"] = (
            "failed" if summary.get("failed", 0) else
            "succeeded" if total and summary.get("done") == total and not summary.get("pending", 0) else "unknown"
        )
        if summary.get("failed", 0) or summary.get("pending", 0):
            result.issue("unfinished_frames", f"Frame state: {summary}")
        for frame in state.get("frames", {}).values():
            if isinstance(frame, dict) and frame.get("meta", {}).get("onnx_error"):
                result.issue("onnx_error", f"Frame {frame.get('index')} has onnx_error despite its state")
        if total:
            for backend, count in counts.items():
                if count != total:
                    result.issue("state_coverage_mismatch", f"{backend}: metrics={count}, exported={total}")
        if state.get("config", {}).get("_resume_history"):
            result.issue("resumed_aggregate_scope", "Upstream resumed summaries may cover only the last process; retain per-file counts", "warning")
    validation = result.read(root / "input_validation.json", "input_validation")
    integrity = result.read(root / "output_integrity.json", "output_integrity")
    if isinstance(integrity, dict):
        result.value["provenance"]["output_integrity"] = integrity
        if integrity.get("state") == "stale_outputs":
            result.issue("stale_board_outputs", "Archived sample outputs contaminated at least one frame; aggregate tensor scores are invalid")
            for item in result._metrics.values():
                if item["name"].startswith("tensor."):
                    item["validity"] = "invalid_output_provenance"
        elif integrity.get("state") == "validation_failed":
            result.issue("output_integrity_failed", "Output freshness/finite-value verification failed")
    summary_receipt = result.read(root / "tensor_summary_receipt.json", "tensor_summary_receipt")
    if isinstance(summary_receipt, dict):
        result.value["provenance"]["tensor_summary"] = summary_receipt
    execution_receipt = result.read(root / "execution_receipt.json", "execution_receipt")
    if isinstance(execution_receipt, dict):
        result.value["provenance"]["live_run"] = execution_receipt
        requested = execution_receipt.get("requested_frames")
        if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
            if result.value["coverage"]["expected_frames"] is None:
                result.value["coverage"]["expected_frames"] = requested
            elif result.value["coverage"]["expected_frames"] != requested:
                result.issue("declared_scope_mismatch", "Execution receipt scope differs from the collection request")
        if execution_receipt.get("state") == "failed":
            result.value["execution"]["state"] = "failed"
            result.issue("execution_failed", "The live execution receipt reports failure")
    if gt_receipt and not isinstance(validation, dict):
        validation = {}
    if isinstance(validation, dict):
        if gt_receipt:
            validation = dict(validation, ground_truth={'state':'validated', 'source':'gt_adaptation',
                                                       'num_frames':gt_receipt['num_frames']})
        result.value["provenance"]["input_validation"] = validation
        if validation.get("ground_truth", {}).get("state") == "grid_contract_mismatch":
            result.issue("gt_grid_contract_mismatch", "GT voxel geometry differs from the legacy evaluator contract; GT-based scores are not valid quality evidence")
            for item in result._metrics.values():
                if not item["name"].startswith("tensor."):
                    item["validity"] = "invalid_gt_geometry"
        if validation.get("finite_outputs", {}).get("state") == "failed":
            result.issue("non_finite_output", "The real output tensor finite-value check failed")
    return result.finish()
