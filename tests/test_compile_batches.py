import hashlib
import json
from pathlib import Path
import shutil
import sys
from contextlib import contextmanager
import uuid
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_compile_batches as batches

TMP = Path(__file__).resolve().parent / '.tmp'
TMP.mkdir(exist_ok=True)


@contextmanager
def temporary_directory():
    path = TMP / uuid.uuid4().hex
    path.mkdir()
    assert path.resolve().is_relative_to(TMP.resolve())
    try:
        yield path
    finally:
        shutil.rmtree(path)


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.root = self.temp.__enter__()
        self.addCleanup(self.temp.__exit__, None, None, None)
        scenes = {s: {'frames': [{'frame_index': 1}]} for s in ('a', 'b')}
        link = self.root / 'link.json'
        link.write_text(json.dumps({'eval': scenes, 'joint_eval': scenes, 'train': {'ignored': {}}}))
        self.request = {'run_id': 'fixture', 'kind': 'compile_alignment', 'version_name': 'fixture',
                        'execution': {'output_root': str(self.root / 'runs')},
                        'dataset': {'data_link_file': str(link)}, 'model': {},
                        'evaluation': {'expected_frames': 2, 'batch_size': 1}}
        self.calls = []
        for target, value in [('ROOT', self.root), ('fingerprint', lambda r: {'request': batches.canonical(r)}),
                              ('verify_batch', lambda p, pairs, commit=None: {'receipt': 'fixed'}),
                              ('aggregate', lambda *a: None)]:
            p = patch.object(batches, target, value)
            p.start()
            self.addCleanup(p.stop)
        fake = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=8, flock=lambda *args: None)
        p = patch.dict(sys.modules, fcntl=fake)
        p.start()
        self.addCleanup(p.stop)

    def leaf(self, request):
        self.calls.append(request['run_id'])
        path = Path(request['execution']['output_root']) / request['run_id']
        path.mkdir(parents=True)
        (path / 'execution_receipt.json').write_text(json.dumps({'state': 'succeeded'}))
        return 0

    def test_checkpoint_resume_reuses_committed_batch_and_rejects_changed_request(self):
        self.assertEqual(batches.run(self.request, batch_limit=1, leaf_runner=self.leaf), 75)
        self.assertEqual(len(self.calls), 1)
        changed = dict(self.request, version_name='different')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            batches.run(changed, resume=True, leaf_runner=self.leaf)
        self.assertEqual(batches.run(self.request, resume=True, leaf_runner=self.leaf), 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(batches.run(self.request, resume=True, leaf_runner=self.leaf), 0)
        self.assertEqual(len(self.calls), 2)

    def test_success_before_controller_commit_is_adopted_without_rerun(self):
        batches.run(self.request, batch_limit=1, leaf_runner=self.leaf)
        path = self.root / 'runs/fixture/workflow.json'
        value = json.loads(path.read_text())
        value['batches'][0].pop('committed_path')
        value['batches'][0].pop('hashes')
        path.write_text(json.dumps(value))
        batches.run(self.request, resume=True, leaf_runner=self.leaf)
        self.assertEqual(len(self.calls), 2)

    def test_failed_attempt_is_preserved_and_bounded_retry_uses_new_id(self):
        self.request['evaluation']['batch_retries'] = 1
        count = [0]
        def runner(request):
            count[0] += 1
            if count[0] == 1:
                return 2
            return self.leaf(request)
        self.assertEqual(batches.run(self.request, leaf_runner=runner), 0)
        value = json.loads((self.root / 'runs/fixture/workflow.json').read_text())
        attempts = value['batches'][0]['attempts']
        self.assertEqual([a['state'] for a in attempts], ['failed', 'succeeded'])
        self.assertNotEqual(attempts[0]['path'], attempts[1]['path'])

    def test_busy_bench_does_not_trigger_automatic_retries(self):
        self.request['evaluation']['batch_retries'] = 2
        calls = []
        def busy(request):
            calls.append(request)
            return 73
        self.assertEqual(batches.run(self.request, leaf_runner=busy), 2)
        self.assertEqual(len(calls), 1)


class BatchContentVerificationTests(unittest.TestCase):
    def test_modified_raw_tensor_is_rejected_even_if_metadata_unchanged(self):
        with temporary_directory() as folder:
            root = Path(folder)
            frame = root / 'frame_000000'
            frame.mkdir()
            tensor = frame / 'output.bin'
            tensor.write_bytes(b'original')
            digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
            records = {
                'execution_receipt.json': {'state': 'succeeded', 'selected_frames': [['a', 1]]},
                'output_integrity.json': {'state': 'fresh_outputs_checked'},
                'input_validation.json': {'finite_outputs': {'state': 'passed', 'checks': [
                    {'frame': frame.name, 'backend': 'board', 'output': 'x', 'sha256': digest}]},
                    'input_tensors': {'checks': []}}}
            for name, value in records.items():
                (root / name).write_text(json.dumps(value))
            (frame / 'cosine_board_onnx.json').write_text(json.dumps({
                'board_dir': str(frame), 'outputs': [{'onnx_output_name': 'x', 'board_file': tensor.name}]}))
            commit = batches.verify_batch(root, [['a', 1]])
            tensor.write_bytes(b'corrupt!')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                batches.verify_batch(root, [['a', 1]], commit)


if __name__ == '__main__':
    unittest.main()
