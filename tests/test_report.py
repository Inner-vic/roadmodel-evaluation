import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from pcm_eval.common import load_json
from pcm_eval.planning import build_plan
from pcm_eval.report import build_report, render_html
from pcm_eval.runner import PREFLIGHT, execute
from test_compile_batches import temporary_directory


class ReportAndIdentityTests(unittest.TestCase):
    def compile_result(self):
        return {'schema': 'pcm.single-eval/v1', 'evaluation_type': 'compile_alignment',
                'version_name': 'fixture', 'execution': {'state': 'succeeded'},
                'collection': {'status': 'complete', 'issues': []},
                'coverage': {'observed': {'tensor_comparison': 2}}, 'provenance': {},
                'metrics': [{'name': 'tensor.frame_mean_cosine.mean', 'value': 0.996940,
                             'unit': 'ratio', 'dimensions': {}},
                            {'name': 'tensor.mean_cosine', 'value': 0.996940, 'unit': 'ratio',
                             'dimensions': {'output': 'fixture-output'}},
                            {'name': 'tensor.min_cosine', 'value': 0.942701, 'unit': 'ratio',
                             'dimensions': {'output': 'fixture-output'}}]}

    def test_missing_model_is_separate_and_does_not_reuse_compile_scores(self):
        report = build_report(self.compile_result(), readiness={'missing_fields': ['model.checkpoint']})
        self.assertEqual(report['evaluation_groups']['model_accuracy']['state'], 'not_run')
        self.assertEqual(report['evaluation_groups']['model_accuracy']['metrics'], [])
        self.assertEqual(report['evaluation_groups']['compile_alignment']['state'], 'complete')
        self.assertEqual(report['comparison']['state'], 'not_requested')
        self.assertEqual(report['quality_assessment'], 'not_assessed')
        self.assertIn('PTH 权重路径', render_html(report))

    def test_invalidated_output_scores_are_not_rendered_as_valid_headline_numbers(self):
        result = self.compile_result()
        for metric in result['metrics']:
            metric['validity'] = 'invalid_output_provenance'
        report = build_report(result)
        self.assertEqual(report['evaluation_groups']['compile_alignment']['state'], 'partial')
        self.assertNotIn('0.996940', render_html(report))

    def test_html_escapes_model_names_and_rejects_wrong_result_kind(self):
        result = self.compile_result()
        result['version_name'] = '<script>unsafe</script>'
        html = render_html(build_report(result))
        self.assertNotIn('<script>unsafe</script>', html)
        self.assertIn('&lt;script&gt;', html)
        result['evaluation_type'] = 'model_accuracy'
        with self.assertRaises(ValueError):
            build_report(result)

    def test_missing_model_bindings_block_verified_plan(self):
        root = Path(__file__).resolve().parents[1]
        request = load_json(root / 'examples/model_request.json')
        request['model']['checkpoint_sha256']=None
        request['model']['config_sha256']=None
        request['source']['commit_id']=None
        plan = build_plan(request)
        self.assertFalse(plan['ready'])
        for field in ('model.checkpoint_sha256', 'model.config_sha256', 'source.commit_id'):
            self.assertIn(field, plan['missing_fields'])

    def test_preflight_rejects_changed_checkpoint_before_any_evaluation(self):
        with temporary_directory() as root:
            checkpoint = root / 'model.pth'
            checkpoint.write_bytes(b'verified version')
            identity = json.dumps({'files': {str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}})
            argv = [sys.executable, '-c', PREFLIGHT, 'new', str(root / 'output'), identity, str(checkpoint)]
            self.assertEqual(subprocess.run(argv, capture_output=True).returncode, 0)
            checkpoint.write_bytes(b'different version')
            failed = subprocess.run(argv, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(b'Input content changed', failed.stderr)


if __name__ == '__main__':
    unittest.main()
