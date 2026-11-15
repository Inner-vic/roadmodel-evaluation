"""Synthetic snippets reproduce documented source formats; these are not model scores."""
import json
import shutil
import unittest
import uuid
from pathlib import Path

from pcm_eval.common import save_json, digest
from pcm_eval.parsers import collect_compile, collect_model


class ParserTests(unittest.TestCase):
    def setUp(self):
        task_test_root = Path(__file__).resolve().parent / ".tmp"
        task_test_root.mkdir(exist_ok=True)
        self.root = task_test_root / uuid.uuid4().hex
        self.root.mkdir()
        self.assertTrue(self.root.resolve().is_relative_to(task_test_root.resolve()))
        self.addCleanup(shutil.rmtree, self.root)

    def metric(self, result, name, **dimensions):
        return [m for m in result["metrics"] if m["name"] == name and all(m["dimensions"].get(k) == v for k, v in dimensions.items())]

    def write_compile(self, count=2):
        for backend in ("board", "onnx"):
            save_json(self.root / f"metrics_{backend}.json", {
                "num_frames": count, "occ_iou": 51.2, "present_miou": 19.5,
                "present_class_ious": [{"class_id": 1, "name": "asphalt", "iou": 44.5,
                                        "seen": 10, "correct": 8, "positive": 12}]})
        save_json(self.root / "cosine_summary.json", {
            "num_frames": count, "mean_cosine_all": 0.99, "min_cosine_all": 0.97,
            "tensors": {"opacity": {"mean_cosine": 0.99, "min_cosine": 0.2, "num_frames": count}},
            "frames": [{"frame_id": f"f{i}", "outputs": [{"onnx_output_name": "opacity", "status": "ok"}]} for i in range(count)]})
        save_json(self.root / "state.json", {"total_frames": count, "summary": {"done": count, "failed": 0, "pending": 0},
                                            "config": {}, "frames": {}})

    def change(self, name, callback):
        path = self.root / name
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_model_class_scope_units_and_raw_provenance(self):
        (self.root / "eval.log").write_text(
            "[TRAIN] per-class stats\n01 asphalt | iou=99% prec=99% recall=99% | seen=1 correct=1 positive=1\n"
            "[EVAL] per-class stats\n01 asphalt | iou=50.00% prec=60.00% recall=75.00% | seen=8 correct=6 positive=10\n"
            "mIoU: 50, iou2: 52\n[EVAL_PILLAR] per-class stats\n"
            "01 asphalt | iou=70% prec=80% recall=90% | seen=20 correct=18 positive=22\n"
            "pillar_mIoU: 71, pillar_iou2: 72\n"
            "[core] 雨天 (rainfall_intensity): 0.9983\n[core] avg mAP: 0.9141\n"
            "雨天 (rainfall_intensity): acc=93.91%, mAP=0.9983 (major=1), macro_recall=95.40%\n"
            "class_0: n=111 (96.8%), recall=95.40%, AP=0.9983\n"
            "class_1 [少数类, 不计入mAP]: n=3 (3.2%), recall=10.00%, AP=0.1000\n"
            "Total samples used: 114\n", encoding="utf-8")
        result = collect_model(self.root, "unit-fixture")
        self.assertEqual(self.metric(result, "occ.iou", scope="EVAL")[0]["value"], 50)
        self.assertEqual(self.metric(result, "occ.iou", scope="EVAL_PILLAR")[0]["value"], 70)
        self.assertFalse(self.metric(result, "occ.iou", scope="TRAIN"))
        self.assertEqual(self.metric(result, "occ.miou")[0]["value"], 50)
        self.assertEqual(self.metric(result, "vision.map")[0]["unit"], "ratio")
        self.assertEqual(self.metric(result, "vision.accuracy")[0]["unit"], "percent")
        self.assertEqual(self.metric(result, "vision.class.ap", class_id="1")[0]["dimensions"]["included_in_map"], False)
        self.assertEqual(result["coverage"]["observed"]["vision_samples"], 114)
        self.assertEqual(result["quality_assessment"], "not_assessed")
        self.assertEqual(len(result["artifacts"][0]["sha256"]), 64)
        self.assertTrue(self.metric(result, "occ.miou")[0]["source"]["locator"].startswith("line:"))

    def test_final_repeated_metric_wins_without_averaging_runs(self):
        (self.root / "eval.log").write_text("mIoU: 10, iou2: 12\nmIoU: 50, iou2: 51\n")
        result = collect_model(self.root)
        self.assertEqual(self.metric(result, "occ.miou")[0]["value"], 50)

    def test_nonfinite_is_not_zero_or_invalid_json(self):
        (self.root / "eval.log").write_text("mIoU: nan, iou2: 52\n")
        result = collect_model(self.root)
        self.assertFalse(self.metric(result, "occ.miou"))
        self.assertEqual(result["collection"]["status"], "partial")
        json.dumps(result, allow_nan=False)

    def test_multiple_logs_are_not_silently_combined(self):
        for name in ("first", "second"):
            (self.root / f"{name}.log").write_text("mIoU: 50, iou2: 52\n")
        result = collect_model(self.root)
        self.assertEqual(result["collection"]["status"], "empty")
        result = collect_model(self.root, log_file=self.root / "second.log")
        self.assertEqual(len(self.metric(result, "occ.miou")), 1)

    def test_disabled_scheme_groups_and_metadata_are_not_metrics(self):
        save_json(self.root / "eval_scheme_report.json", {
            "meta": {"kept_samples": 10, "datalink_note": "static-source-note"},
            "noise": {"enabled": False, "mae_m": 999},
            "dimension_A_gs_pillar": {"mIoU": 0.5},
            "bev_compare_placeholder": {"ready": True}})
        result = collect_model(self.root)
        self.assertEqual(len(result["metrics"]), 1)
        self.assertEqual(result["coverage"]["observed"]["scheme_samples"], 10)
        self.assertEqual(result["metrics"][0]["unit"], "source_defined")

    def test_compiler_does_not_confuse_frame_mean_min_with_tensor_min(self):
        self.write_compile()
        result = collect_compile(self.root, expected_frames=2)
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(self.metric(result, "tensor.frame_mean_cosine.min")[0]["value"], 0.97)
        self.assertEqual(self.metric(result, "tensor.min_cosine", output="opacity")[0]["value"], 0.2)
        self.assertEqual(self.metric(result, "occ_iou", backend="board")[0]["unit"], "percent")

    def test_backend_frame_mismatch_and_resume_are_visible(self):
        self.write_compile()
        self.change("metrics_onnx.json", lambda d: d.update(num_frames=1))
        self.change("state.json", lambda d: d["config"].update(_resume_history=[{"at": "fixture"}]))
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertIn("backend_coverage_mismatch", {x["code"] for x in result["collection"]["issues"]})
        self.assertIn("resumed_aggregate_scope", {x["code"] for x in result["collection"]["issues"]})

    def test_onnx_error_cannot_hide_behind_done_state(self):
        self.write_compile()
        self.change("state.json", lambda d: d["frames"].update({"0": {"index": 0, "status": "done", "meta": {"onnx_error": "fixture failure"}}}))
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertIn("onnx_error", {x["code"] for x in result["collection"]["issues"]})

    def test_missing_tensor_and_empty_comparison_are_not_success(self):
        self.write_compile()
        self.change("cosine_summary.json", lambda d: d.update(tensors={}, frames=[{"frame_id": "f0", "outputs": [{"onnx_output_name": "opacity", "status": "missing_board"}]}]))
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertIn("missing_tensor", {x["code"] for x in result["collection"]["issues"]})

    def test_missing_artifacts_never_make_zero_scores(self):
        result = collect_compile(self.root)
        self.assertEqual(result["metrics"], [])
        self.assertEqual(result["collection"]["status"], "empty")

    def test_missing_aggregate_values_are_not_silently_successful(self):
        self.write_compile()
        self.change("cosine_summary.json", lambda d: d.update(mean_cosine_all=None))
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertFalse(self.metric(result, "tensor.frame_mean_cosine.mean"))

    def test_nonfinite_frame_is_visible_even_if_mean_is_finite(self):
        self.write_compile()
        self.change("cosine_summary.json", lambda d: d["frames"][0]["outputs"][0].update(cosine=float("nan")))
        result = collect_compile(self.root)
        self.assertIn("non_finite_tensor", {x["code"] for x in result["collection"]["issues"]})
        json.dumps(result, allow_nan=False)

    def test_tensor_json_locator_escapes_graph_names(self):
        self.write_compile()
        self.change("cosine_summary.json", lambda d: d["tensors"].update({"/a~b": d["tensors"].pop("opacity")}))
        result = collect_compile(self.root)
        self.assertEqual(self.metric(result, "tensor.mean_cosine", output="/a~b")[0]["source"]["locator"], "/tensors/~1a~0b/mean_cosine")

    def test_expected_scope_is_checked_per_backend(self):
        self.write_compile()
        result = collect_compile(self.root, expected_frames=10)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertEqual(result["coverage"]["observed"]["board_metrics"], 2)

    def test_malformed_json_is_collected_as_error(self):
        self.write_compile()
        (self.root / "metrics_board.json").write_text("not JSON")
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertIn("invalid_json", {x["code"] for x in result["collection"]["issues"]})

    def test_wrong_json_shape_cannot_silently_drop_one_backend(self):
        self.write_compile()
        (self.root / "metrics_board.json").write_text("[]")
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertIn("invalid_json_shape", {x["code"] for x in result["collection"]["issues"]})

    def test_invalid_tensor_statistics_cannot_count_as_comparison(self):
        self.write_compile()
        self.change("cosine_summary.json", lambda d: d.update(tensors={"opacity": []}))
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")

    def test_observed_gt_contract_error_invalidates_only_gt_scores(self):
        self.write_compile()
        save_json(self.root / "input_validation.json", {
            "ground_truth": {"state": "grid_contract_mismatch"}, "finite_outputs": {"state": "passed"}})
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertEqual(self.metric(result, "occ_iou", backend="board")[0]["validity"], "invalid_gt_geometry")
        self.assertNotIn("validity", self.metric(result, "tensor.frame_mean_cosine.mean")[0])

    def test_explicit_tensor_only_run_needs_no_gt_metric_files(self):
        self.write_compile()
        self.change("state.json", lambda d: d["config"].update(skip_metrics=True))
        (self.root / "metrics_board.json").unlink()
        (self.root / "metrics_onnx.json").unlink()
        result = collect_compile(self.root, expected_frames=2)
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(result["provenance"]["metric_scope"], "tensor_alignment")
        self.assertTrue(all(m["name"].startswith("tensor.") for m in result["metrics"]))

    def test_stale_sample_outputs_invalidate_aggregate_comparison(self):
        self.write_compile()
        save_json(self.root / "output_integrity.json", {"state": "stale_outputs", "affected_frames": [0]})
        result = collect_compile(self.root)
        self.assertEqual(result["collection"]["status"], "partial")
        self.assertEqual(self.metric(result, "tensor.frame_mean_cosine.mean")[0]["validity"], "invalid_output_provenance")

    def test_gt_postprocessing_is_bound_to_exact_original_scope_and_hashes(self):
        self.write_compile()
        self.change('state.json',lambda d:d['config'].update(skip_metrics=True))
        save_json(self.root/'execution_receipt.json',{'run_id':'source-run','selected_frames':[['scene',1],['scene',2]]})
        save_json(self.root/'output_integrity.json',{'state':'fresh_outputs_checked'})
        gt=self.root/'gt';gt.mkdir()
        for backend in ('board','onnx'):
            shutil.copyfile(self.root/f'metrics_{backend}.json',gt/f'metrics_{backend}.json')
        receipt={'schema':'pcm.gt-adaptation/v1','state':'validated','source_run_id':'source-run',
                 'selected_frames':[['scene',1],['scene',2]],'num_frames':2,'frames':[{},{}]}
        for name,field in [('execution_receipt.json','source_execution_receipt_sha256'),
                           ('cosine_summary.json','source_cosine_summary_sha256'),
                           ('output_integrity.json','source_output_integrity_sha256')]:
            receipt[field]=digest(self.root/name)
        for backend in ('board','onnx'):receipt[backend+'_metrics_sha256']=digest(gt/f'metrics_{backend}.json')
        save_json(gt/'gt_receipt.json',receipt)
        good=collect_compile(self.root,gt_dir=gt)
        self.assertEqual(good['provenance']['metric_scope'],'tensor_alignment_and_gt')
        self.assertTrue(self.metric(good,'present_miou',backend='board'))
        self.change('cosine_summary.json',lambda d:d.update(mean_cosine_all=.123))
        with self.assertRaisesRegex(ValueError,'source identity mismatch'):
            collect_compile(self.root,gt_dir=gt)
        receipt['selected_frames']=[['different-scene',1],['different-scene',2]]
        (gt/'gt_receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'exact source prediction inventory'):
            collect_compile(self.root,gt_dir=gt)


if __name__ == "__main__":
    unittest.main()
