import json
import shlex
import sys
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pcm_eval.common import load_json, save_json, json_digest, digest
from pcm_eval.datalink import inspect_datalink
from pcm_eval.lpai import collect_task, extract_lines, select_pod, timestamp
from pcm_eval.planning import build_plan, ssh_command, verify_plan
from pcm_eval.runner import execute


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        task_test_root = Path(__file__).resolve().parent / ".tmp"
        task_test_root.mkdir(exist_ok=True)
        self.root = task_test_root / uuid.uuid4().hex
        self.root.mkdir()
        self.assertTrue(self.root.resolve().is_relative_to(task_test_root.resolve()))
        self.addCleanup(shutil.rmtree, self.root)

    def model_request(self):
        return {"kind": "model_accuracy", "version_name": "test-fixture", "execution": {
            "transport": "ssh", "ssh_alias": "lpai", "repo_dir": "/repo with space",
            "output_root": "/results", "python": "python3"},
            "dataset": {"path": "/data", "data_link_file": "/data/link.json", "class_name": "SmartChassisDatasetV7"},
            "model": {"checkpoint": "/model/best.pth", "config": "config/4-0-0.py"},
            "evaluation": {"frame_fraction": 1.0}}

    def test_same_fraction_is_translated_to_different_script_units(self):
        request = self.model_request()
        request["evaluation"]["frame_fraction"] = 0.25
        model = build_plan(request)
        argv = model["steps"][0]["argv"]
        self.assertEqual(argv[argv.index("--eval-subsample") + 1], "0.25")
        request["kind"] = "compile_alignment"
        request["model"].update(onnx="/model/model.onnx", packup_tar="/model/pack.tar.gz")
        request["dataset"]["export_root"] = "/data"
        request["execution"]["board_ssh"] = "ssh board"
        plan = build_plan(request)
        argv = plan["steps"][0]["argv"]
        self.assertEqual(argv[argv.index("--eval-subsample") + 1], "25.0")
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["issues"][0]["code"], "v7_pipeline_unverified")

    def test_postprocessing_uses_full_exports_and_effective_work_dir(self):
        plan = build_plan(self.model_request(), "fixture-001")
        argv = plan["steps"][0]["argv"]
        self.assertEqual(argv[argv.index("--sample-freq") + 1], "1")
        self.assertEqual(argv[argv.index("--zip-vis") + 1], "none")
        self.assertEqual(plan["effective_output_dir"], "/results/fixture-001/4-0-0")
        self.assertTrue(plan["ready"])

    def test_missing_inputs_produce_actionable_plan_without_execution(self):
        request = self.model_request()
        request["model"]["checkpoint"] = None
        plan = build_plan(request)
        self.assertIn("model.checkpoint", plan["missing_fields"])
        with self.assertRaises(ValueError):
            verify_plan(plan)

    def test_plan_tampering_is_detected(self):
        plan = build_plan(self.model_request())
        plan["steps"][0]["argv"][0] = "different-executable"
        with self.assertRaises(ValueError):
            verify_plan(plan)

    def test_invalid_fraction_is_not_forwarded(self):
        for fraction in (0, -1, 20, True, float("nan")):
            request = self.model_request()
            request["evaluation"]["frame_fraction"] = fraction
            with self.assertRaises(ValueError):
                build_plan(request)

    def test_bound_configuration_environment_cannot_be_changed_silently(self):
        request=self.model_request()
        request['source']={'require_verified_binding':True,'configuration_env_sha256':json_digest({'DENSE_SURFACE_DDIM_STEPS':'4'})}
        request['execution']['configuration_env']={'DENSE_SURFACE_DDIM_STEPS':'8'}
        with self.assertRaisesRegex(ValueError,'recorded binding'):build_plan(request)

    def test_preflight_verifies_gitless_snapshot_source(self):
        from pcm_eval.runner import PREFLIGHT
        source=self.root/'snapshot';source.mkdir()
        script=source/'eval.py';script.write_text('# pinned evaluator\n')
        manifest=save_json(self.root/'source.json',{'commit':'a'*40,'files':{'eval.py':digest(script)}})
        identity={'files':{str(manifest):digest(manifest)},'source_snapshot':{'root':str(source),'manifest':str(manifest),'commit_id':'a'*40}}
        argv=[sys.executable,'-c',PREFLIGHT,'new',str(self.root/'output'),json.dumps(identity),str(source)]
        self.assertEqual(__import__('subprocess').run(argv,capture_output=True).returncode,0)
        script.write_text('# modified evaluator\n')
        failed=__import__('subprocess').run(argv,capture_output=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertIn(b'Pinned source file differs',failed.stderr)

    def test_ssh_arguments_are_quoted_and_alias_is_not_shell(self):
        argv = ["python3", "eval.py", "--resume-from", "/a/model; touch injected.pth"]
        command = ssh_command("lpai", argv, "/repo with space", {"DATASET_PATH": "/x/$(not-executed)"})
        tokens = shlex.split(command[-1])
        self.assertIn("/a/model; touch injected.pth", tokens)
        self.assertIn("DATASET_PATH=/x/$(not-executed)", tokens)
        with self.assertRaises(ValueError):
            ssh_command("-oProxyCommand=evil", argv)

    def test_real_local_subprocess_roundtrip_and_duplicate_protection(self):
        repo = self.root / "repo"
        (repo / "config").mkdir(parents=True)
        (repo / "config/fixture.py").write_text("# source fixture\n")
        checkpoint = repo / "checkpoint.pth"
        checkpoint.write_bytes(b"not-a-real-model; execution fixture")
        (repo / "data").mkdir()
        (repo / "data/link.json").write_text("{}")
        (repo / "eval.py").write_text(
            "import argparse,pathlib\np=argparse.ArgumentParser()\n"
            "p.add_argument('--work-dir');p.add_argument('--py-config')\na,_=p.parse_known_args()\n"
            "d=pathlib.Path(a.work_dir)/pathlib.Path(a.py_config).stem\nd.mkdir(parents=True)\n"
            "(d/'fixture.log').write_text('mIoU: 50, iou2: 51\\nTotal samples: 2\\n')\n")
        request = self.model_request()
        request["execution"].update(transport="local", repo_dir=str(repo), output_root=str(self.root / "runs"), python=sys.executable)
        request["model"].update(checkpoint=str(checkpoint), config="config/fixture.py")
        request["dataset"].update(path=str(repo / "data"), data_link_file=str(repo / "data/link.json"))
        request["evaluation"]["compute_scheme"] = False
        plan = build_plan(request)
        receipt = execute(plan, self.root / "evidence")
        self.assertEqual(receipt["state"], "succeeded")
        result = load_json(receipt["result_file"])
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(result["execution"]["exit_code"], 0)
        self.assertEqual(result["quality_assessment"], "not_assessed")
        self.assertEqual(result["coverage"]["observed"]["model_samples"], 2)
        with self.assertRaises(ValueError):
            execute(plan, self.root / "evidence")

    def test_datalink_only_selects_requested_eval_split_and_detects_duplicates(self):
        frame = {"frame_index": 10, "time_stamp": 123456, "occ_file": "occ.npy"}
        path = save_json(self.root / "link.json", {"train": {"train-only": {"frames": [frame]}},
                                                   "eval": {"scene-a": {"frames": [frame, frame]}}})
        value = inspect_datalink(path)
        self.assertEqual(value["frame_count"], 1)
        self.assertEqual(value["scene_count"], 1)
        self.assertEqual(value["issues"][0]["code"], "duplicate_frame")
        self.assertEqual(value["frames"][0]["frame_metadata"]["occ_file"], "occ.npy")
        with self.assertRaises(ValueError):
            inspect_datalink(path, ["train"])

    def test_lpai_pod_selection_ignores_prepare_and_system_jobs(self):
        pods = [{"name": "init", "step": "dataPreparation", "pod_type": "user"},
                {"name": "sys-master-0", "step": "task", "pod_type": "sys"},
                {"name": "real-master-0", "step": "task", "pod_type": "user"}]
        self.assertEqual(select_pod(pods)["name"], "real-master-0")

    def test_loki_lines_sorted_and_deduplicated(self):
        rows = extract_lines({"resultType": "streams", "result": [{"values": [["2", "b"], ["1", "a"], ["2", "b"]]}]})
        self.assertEqual(rows, [(1, "a"), (2, "b")])
        self.assertEqual(timestamp({"time": "2026-05-10T16:05:00Z"}).utcoffset().total_seconds(), 0)

    def test_lpai_collection_does_not_save_full_task_configuration(self):
        task = {"name": "fixture-eval", "namespace": "vc-chassis", "cluster": "fixture", "status": "succeeded",
                "data": {"secret": "DO_NOT_PERSIST"}}
        pod = {"name": "fixture-master-0", "step": "task", "pod_type": "user",
               "create_timestamp": {"time": "2026-05-10T16:05:00Z"},
               "finish_timestamp": {"time": "2026-05-10T16:26:22Z"}}
        logs = {"resultType": "streams", "result": [{"values": [["1", "mIoU: 50, iou2: 51"]]}]}
        with patch("pcm_eval.lpai.request", side_effect=[task, [pod], logs]):
            result = collect_task("fixture-uid", self.root / "collected")
        text = (self.root / "collected/result.json").read_text(encoding="utf-8")
        self.assertNotIn("DO_NOT_PERSIST", text)
        self.assertEqual(result["execution"]["task_uid"], "fixture-uid")


if __name__ == "__main__":
    unittest.main()
