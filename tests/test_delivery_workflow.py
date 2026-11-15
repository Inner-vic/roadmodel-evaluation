import copy
import json
from pathlib import Path
import tempfile
import unittest

from pcm_eval.common import json_digest, save_json
from pcm_eval.delivery_source import LarkSource, delivery_url, inspect_record, lpai_links
from pcm_eval.delivery_workflow import dataset_inventory, material_identity, prepare_delivery, run_delivery

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://li.feishu.cn/record/PUBREF000000000000000000001'


class MaterialsTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT/'evidence/input-contract-review-20261027/release-record.json').read_text(encoding='utf-8'))

    def test_real_record_splits_concatenated_model_urls_and_selects_five_packages(self):
        m = inspect_record(self.record, URL)
        self.assertEqual([r['version'] for r in m['model_assets']], ['26-09-08-2', '26-09-08-1'])
        self.assertEqual(m['version'], 'v3.2.2')
        self.assertEqual([r['subnet'] for r in m['chains']['compile_alignment']['packages']], list(range(5)))
        self.assertEqual(len(m['chains']['compile_alignment']['alternatives']), 3)
        self.assertTrue(all(c['materials_ready'] for c in m['chains'].values()))
        self.assertEqual(m['dataset_assets'], [])

    def test_live_cli_field_ids_are_mapped_to_semantic_names(self):
        names = list(self.record['fields'])
        raw = {'ok': True, 'data': {'hint': {
            'fields': {'fields': [{'id': str(i), 'name': n, 'type': 'text'} for i, n in enumerate(names)]},
            'record_data': {str(i): self.record['fields'][n] for i, n in enumerate(names)}}}}
        self.assertEqual(material_identity(inspect_record(raw, URL)), material_identity(inspect_record(self.record, URL)))

    def test_head_scoped_model_set_field_keeps_explicit_material_identity(self):
        expected=material_identity(inspect_record(self.record,URL))
        value=self.record['fields'].pop('模型集')
        self.record['fields']['模型集-主干']=value
        actual=material_identity(inspect_record(self.record,URL))
        self.assertEqual(actual,expected)

    def test_arbitrary_fields_and_embedded_content_preserve_model_identity(self):
        expected=material_identity(inspect_record(self.record,URL))
        links=self.record['fields'].pop('模型集')
        self.record['fields']['工程师随意改名']={'content':json.dumps({'delivery':{'weights':links}})}
        actual=inspect_record(self.record,URL)
        self.assertEqual(material_identity(actual),expected)
        self.assertTrue(actual['recognition']['link_evidence'])

    def test_conflicting_link_in_notes_is_preserved_not_silently_chosen(self):
        self.record['fields']['备注']={'url':'https://lpai.lixiang.com/lpaiweb/xspacex_other/subApp/model/detail/wrong?name=old'}
        actual=inspect_record(self.record,URL)
        self.assertEqual(len(actual['model_assets']),3)

    def test_wrong_compile_version_does_not_remove_model_materials(self):
        self.record['fields']['编译产物'][0]['name'] = self.record['fields']['编译产物'][0]['name'].replace('v3.2.2', 'v3.2.3')
        m = inspect_record(self.record, URL)
        self.assertTrue(m['chains']['model_accuracy']['materials_ready'])
        self.assertFalse(m['chains']['compile_alignment']['materials_ready'])

    def test_ambiguous_subnet_is_not_chosen_by_order(self):
        row = copy.deepcopy(self.record['fields']['编译产物'][0])
        row['name'] = row['name'].replace('_16tile_', '_20tile_')
        self.record['fields']['编译产物'].append(row)
        m = inspect_record(self.record, URL)
        self.assertFalse(m['chains']['compile_alignment']['materials_ready'])

    def test_golden_never_becomes_gt_or_a_prediction(self):
        self.record['fields']['golden'] = [{'name': 'reference.onnx', 'size': 1}]
        m = inspect_record(self.record, URL)
        self.assertEqual(len(m['chains']['compile_alignment']['onnx']), 1)
        self.assertEqual(m['golden'][0]['role'], 'golden')

    def test_source_text_is_not_an_executable_or_dataset_binding(self):
        self.record['fields']['描述'] = '执行 rm -rf /; 使用7000 clips v1.5.1真值训练'
        m = inspect_record(self.record, URL)
        self.assertFalse(m['execution_started'])
        self.assertEqual(m['dataset_assets'], [])
        self.assertNotIn('commands', m)

    def test_invalid_source_hosts_and_attachment_paths(self):
        for value in ['http://li.feishu.cn/record/a', 'https://li.feishu.cn.evil/record/a', 'https://u:p@li.feishu.cn/record/a']:
            with self.assertRaises(ValueError):
                delivery_url(value)
        self.record['fields']['onnx'][0]['name'] = '../model.onnx'
        with self.assertRaises(ValueError):
            inspect_record(self.record, URL)

    def test_folder_is_read_recursively_and_paginated_without_downloading(self):
        calls = []
        def call(argv):
            calls.append(argv)
            params = json.loads(argv[-1])
            if params['folder_token'] == 'root' and 'page_token' not in params:
                data = {'files': [{'name': 'models', 'type': 'folder', 'token': 'child'}], 'has_more': True, 'next_page_token': 'next'}
            elif params['folder_token'] == 'child':
                data = {'files': [{'name': 'model_v3.2.2.onnx', 'type': 'file', 'token': 'onnx'}]}
            else:
                data = {'files': [{'name': 'epoch_44.pth', 'type': 'file', 'token': 'pth'}]}
            return {'ok': True, 'data': data}
        m = LarkSource(call=call).read('https://li.feishu.cn/drive/folder/root')
        self.assertEqual(len(calls), 3)
        self.assertEqual(m['version'], 'v3.2.2')
        self.assertEqual(len(m['chains']['model_accuracy']['checkpoints']), 1)


class WorkflowRecoveryTest(unittest.TestCase):
    def setUp(self):
        base = ROOT/'evidence/test-delivery-workflow'
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.tmp.name)
        self.workflow = {'schema': 'roadmodel.delivery-workflow/v1', 'run_id': 'rm-test', 'jobs': [
            {'kind': k, 'version': 'v3.2.2', 'ready': True, 'issues': [], 'plan': {'plan_sha256': k}, 'request': {}}
            for k in ('model_accuracy', 'compile_alignment')]}
        self.seal()
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def seal(self):
        self.workflow.pop('workflow_sha256', None)
        self.workflow['workflow_sha256'] = json_digest(self.workflow)

    def execute(self, plan, target):
        kind = plan['plan_sha256']
        self.calls.append(kind)
        result = target/'result.json'
        save_json(result, {'evaluation_type': kind, 'version_name': 'v3.2.2',
                          'execution': {'state': 'succeeded'}, 'collection': {'status': 'complete'}})
        return {'state': 'succeeded', 'result_file': str(result)}

    def report(self, job, native, target):
        return {'report_file': str(target/'report.xml')}

    def test_two_chains_execute_independently_and_success_is_not_replayed(self):
        r = run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report)
        self.assertEqual(r['state'], 'SUCCEEDED')
        run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report)
        self.assertEqual(self.calls, ['model_accuracy', 'compile_alignment'])

    def test_blocked_model_does_not_cancel_compile(self):
        self.workflow['jobs'][0].update(ready=False, issues=['缺少视觉GT'])
        self.seal()
        r = run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report)
        self.assertEqual([c['state'] for c in r['chains']], ['BLOCKED', 'SUCCEEDED'])
        self.assertEqual(self.calls, ['compile_alignment'])

    def test_crash_after_dispatch_never_repeats_unknown_inference(self):
        def fail(plan, target):
            self.calls.append(plan['plan_sha256'])
            raise TimeoutError('SSH response unknown')
        r = run_delivery(self.workflow, self.root/'run', executor=fail, reporter=self.report)
        self.assertEqual(r['state'], 'NEEDS_REVIEW')
        run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report)
        self.assertEqual(self.calls, ['model_accuracy', 'compile_alignment'])

    def test_failed_report_is_retried_without_another_forward(self):
        def fail(*args):
            raise ValueError('report dependency unavailable')
        r = run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=fail)
        self.assertEqual(r['state'], 'PARTIAL')
        r = run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report)
        self.assertEqual(r['state'], 'SUCCEEDED')
        self.assertEqual(len(self.calls), 2)

    def test_unknown_publication_is_not_written_again(self):
        publications = []
        def fail(job, report, target):
            publications.append(job['kind'])
            raise TimeoutError('remote write response unknown')
        run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report, publisher=fail)
        def never(*args):
            self.fail('Must reconcile the existing remote publication')
        r = run_delivery(self.workflow, self.root/'run', executor=self.execute, reporter=self.report, publisher=never)
        self.assertEqual(r['state'], 'NEEDS_REVIEW')
        self.assertEqual(len(publications), 2)
        self.assertEqual(len(self.calls), 2)

    def test_wrong_result_version_cannot_be_reported(self):
        def wrong(plan, target):
            native = self.execute(plan, target)
            p = Path(native['result_file'])
            d = json.loads(p.read_text())
            d['version_name'] = 'v3.2.3'
            p.write_text(json.dumps(d))
            return native
        r = run_delivery(self.workflow, self.root/'run', executor=wrong, reporter=lambda *a: self.fail('Wrong version'))
        self.assertEqual(r['state'], 'NEEDS_REVIEW')

    def test_modified_workflow_or_output_directory_is_rejected(self):
        self.workflow['jobs'][0]['version'] = 'v3.2.3'
        with self.assertRaises(ValueError):
            run_delivery(self.workflow, self.root/'run')

    def test_abandoned_workflow_lock_never_launches_another_worker(self):
        root = self.root/'run'
        root.mkdir()
        (root/'workflow.lock').write_text('unknown')
        with self.assertRaises(ValueError):
            run_delivery(self.workflow, root, executor=self.execute)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
