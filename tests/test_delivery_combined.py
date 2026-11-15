import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pcm_eval.business_layout import table
from pcm_eval.common import digest, load_json, save_json
from pcm_eval.combined_delivery import build_combined, write_combined
from pcm_eval.combined_publication import publish_combined
from pcm_eval.delivery_demo_entry import render

ROOT = Path(__file__).resolve().parents[1]


def reference(spec):
    m, c = spec['model_accuracy'], spec['compile_alignment']
    occ = table(['类别', 'IoU', 'precision', 'recall', 'Pillar IoU'],
                [['版本', '', '', '', '']]+[[x['label'], '', '', '', ''] for x in m['occ_classes']]
                +[['可行驶区域iou', '', '', '', ''], ['非空iou', '', '', '', '']])
    vision = table(['类别', 'mAP', '准确率', 'recall', '子类数'],
                   [[x['label'], '', '', '', ''] for x in m['vision_tasks']]+[['avg mAP', '', '', '', '']])
    classes = table(['父类', '子类', '样本', '占比', 'recall', 'AP'],
                    [[x['task'], x['label'], '', '', '', ''] for x in m['vision_major_classes']])
    compiled = table(['类别/指标', 'ONNX fp16', '本次板端'], [[x['label'], '', ''] for x in c['gt_classes']])
    tensors = table(['输出', '含义', '分组', '余弦'], [[x['key'], x['meaning'], x['group'], ''] for x in c['outputs']])
    return '<title>PCM模板</title><h1>开发空间模型评测</h1>'+occ+vision+classes+'<h1>板端产物与fp16</h1>'+compiled+'<h2>输出 tensor</h2>'+tensors


class CombinedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = load_json(ROOT/'configs/report_template.json')
        self.ref = reference(self.spec)
        self.workflow = {'run_id': 'rm-combined-test', 'workflow_sha256': 'workflow', 'scope': 'validation',
            'inputs': {'model_delivery_url': 'https://li.feishu.cn/record/test', 'dataset_ref': 'dataset'},
            'jobs': [{'kind': k, 'version': 'v3.2.2', 'plan': {'run_id': k},
                      'request': {'evaluation': {'expected_frames': 2 if k == 'model_accuracy' else 1}}}
                     for k in ('model_accuracy', 'compile_alignment')]}
        self.result = {'state': 'SUCCEEDED', 'chains': []}
        for kind in ('model_accuracy', 'compile_alignment'):
            native = save_json(self.root/(kind+'-native.json'), {'state': 'succeeded', 'metrics': []})
            report = {'schema': 'roadmodel.model-single-report/v1', 'version_name': 'v3.2.2', 'run_id': kind,
                      'occ_sections': [], 'all_metrics': [], 'vision_metrics': []} if kind == 'model_accuracy' else {
                'schema': 'roadmodel.compile-single-report/v3', 'identity': {'model_version': 'v3.2.2', 'run_id': kind},
                'ground_truth': {'state': 'not_computed', 'classes': [], 'summary': []},
                'tensor_alignment': {'mean_frame_mean_cosine': .99, 'outputs': []}}
            path = save_json(self.root/(kind+'-report.json'), report)
            self.result['chains'].append({'kind': kind, 'state': 'SUCCEEDED', 'issues': [],
                'execution': {'state': 'succeeded', 'result_file': str(native), 'result_sha256': digest(native)},
                'report': {'report_json' if kind == 'model_accuracy' else 'json': str(path)}})

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_combined(self.workflow, self.result, self.ref, self.spec)

    def partial(self):
        self.result['chains'][1] = {'kind': 'compile_alignment', 'state': 'BLOCKED', 'issues': ['等待台架可用']}

    def artifact(self, name):
        xml, data = self.build()
        path = self.root/name
        save_json(path/'report.json', data)
        (path/'report.xml').write_text(xml, encoding='utf-8')
        return {'report_xml': str(path/'report.xml'), 'report_json': str(path/'report.json'), 'source_sha256': name}

    def test_completed_results_share_one_title_and_five_tables(self):
        xml, data = self.build()
        tree = ET.fromstring('<root>'+xml+'</root>')
        self.assertEqual(len(tree.findall('title')), 1)
        self.assertEqual(len(list(tree.iter('table'))), 5)
        self.assertTrue(data['all_evaluations_complete'])
        self.assertIn('99.00', xml)
        self.assertIn('未计算', xml)

    def test_missing_GT_status_is_machine_readable_without_changing_template(self):
        before,_=self.build()
        states={'vision':{'state':'BLOCKED','completed_frames':0}}
        self.result['chains'][0]['state']='PARTIAL'
        self.result['chains'][0]['execution']['head_statuses']=states
        after,data=self.build()
        self.assertEqual(before,after)
        self.assertEqual(data['model_head_statuses'],states)
        self.assertFalse(data['all_evaluations_complete'])

    def test_pending_compile_preserves_model_and_has_no_fabricated_scores(self):
        full, _ = self.build()
        self.partial()
        xml, data = self.build()
        before = list(ET.fromstring('<root>'+full+'</root>').iter('table'))[:3]
        after = list(ET.fromstring('<root>'+xml+'</root>').iter('table'))[:3]
        self.assertEqual([ET.tostring(x) for x in before], [ET.tostring(x) for x in after])
        self.assertIn('等待台架', xml)
        self.assertIn('待评测', xml)
        self.assertNotIn('99.00', xml)
        self.assertFalse(data['all_evaluations_complete'])

    def test_missing_model_does_not_hide_real_compile_results(self):
        self.result['chains'][0] = {'kind': 'model_accuracy', 'state': 'BLOCKED', 'issues': ['权重缺失']}
        xml, data = self.build()
        self.assertIn('模型精度评测尚未完成', xml)
        self.assertIn('99.00', xml)
        self.assertFalse(data['all_evaluations_complete'])

    def test_different_version_is_not_merged(self):
        self.workflow['jobs'][1]['version'] = 'v3.2.3'
        with self.assertRaises(ValueError):
            self.build()

    def test_another_run_cannot_fill_this_task(self):
        path = Path(self.result['chains'][1]['report']['json'])
        data = load_json(path)
        data['identity']['run_id'] = 'historical-run'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.build()

    def test_modified_native_result_cannot_be_merged(self):
        Path(self.result['chains'][0]['execution']['result_file']).write_text('changed')
        with self.assertRaises(ValueError):
            self.build()

    def server(self):
        self.calls = []
        self.document = ''
        def invoke(args, *, cwd):
            self.calls.append(args)
            if args[1] == '+create':
                path = Path(cwd)/args[args.index('--content')+1][3:]
                self.document = path.read_text(encoding='utf-8')
                return {'document': {'document_id': 'one-doc', 'url': 'https://li.feishu.cn/docx/one-doc'}}
            if args[1] == '+fetch':
                tree = ET.fromstring('<root>'+self.document+'</root>')
                for i, n in enumerate(tree):
                    n.set('id', 'block-'+str(i))
                return {'document': {'content': ''.join(ET.tostring(n, encoding='unicode') for n in tree), 'revision_id': 3}}
            if args[1] == '+update':
                path = Path(cwd)/args[args.index('--content')+1][3:]
                title = ET.fromstring('<root>'+self.document+'</root>').find('title')
                self.document = ET.tostring(title, encoding='unicode')+path.read_text(encoding='utf-8')
                return {'result': 'success'}
            if args[1] == '+update-title':
                tree = ET.fromstring('<root>'+self.document+'</root>')
                tree.find('title').text = args[args.index('--title')+1]
                self.document = ''.join(ET.tostring(n, encoding='unicode') for n in tree)
                return {'updated': True}
            self.fail('Unexpected CLI call')
        return invoke

    def test_pending_then_complete_updates_one_document_and_never_creates_second(self):
        completed = copy.deepcopy(self.result)
        self.partial()
        pending = self.artifact('pending')
        invoke = self.server()
        first = publish_combined(pending, self.root/'publication', invoke=invoke)
        self.assertEqual(first['task_sha256'], load_json(pending['report_json'])['task_sha256'])
        self.result = completed
        full = self.artifact('complete')
        second = publish_combined(full, self.root/'publication', invoke=invoke)
        self.assertEqual(first['report_url'], second['report_url'])
        self.assertEqual(second['workflow_sha256'],self.workflow['workflow_sha256'])
        self.assertEqual(sum(c[1] == '+create' for c in self.calls), 1)
        self.assertEqual(sum(c[1] == '+update' for c in self.calls), 1)
        self.assertTrue(second['all_evaluations_complete'])
        count = len(self.calls)
        cached=publish_combined(full, self.root/'publication', invoke=invoke)
        self.assertEqual(cached['workflow_sha256'],self.workflow['workflow_sha256'])
        self.assertEqual(len(self.calls), count)

    def test_unknown_update_does_not_create_or_write_another_document(self):
        pending = self.artifact('pending')
        invoke = self.server()
        publish_combined(pending, self.root/'publication', invoke=invoke)
        newer = self.artifact('newer')
        def fail(args, *, cwd):
            if args[1] == '+update':
                raise TimeoutError('remote response unknown')
            return invoke(args, cwd=cwd)
        with self.assertRaises(TimeoutError):
            publish_combined(newer, self.root/'publication', invoke=fail)
        with self.assertRaises(ValueError):
            publish_combined(newer, self.root/'publication', invoke=invoke)
        self.assertEqual(sum(c[1] == '+create' for c in self.calls), 1)

    def test_background_publisher_keeps_one_document_across_payload_versions(self):
        from pcm_eval.common import json_digest
        from tools.evaluation_agent import publish
        complete=copy.deepcopy(self.result);self.partial()
        invoke=self.server();cfg={'local_state':str(self.root/'agent')}
        urls=[]
        for name,result in [('waiting',self.result),('compiled',complete)]:
            self.result=result;report=self.artifact(name)
            data=load_json(report['report_json'])
            payload={'task':self.workflow['run_id'],'xml':Path(report['report_xml']).read_text(encoding='utf-8'),
                     'data':data,'source_sha256':report['source_sha256']}
            row={'task':payload['task'],'payload':json.dumps(payload),'digest':json_digest(payload)}
            receipt=publish(cfg,row,invoke=invoke)
            self.assertTrue(receipt['requester_read_verified']);urls.append(receipt['report_url'])
        self.assertEqual(urls[0],urls[1])
        self.assertEqual(sum(c[1]=='+create' for c in self.calls),1)
        self.assertEqual(sum(c[1]=='+update' for c in self.calls),1)

    def test_card_only_exposes_the_unified_report_link(self):
        task = {'state': 'PARTIAL', 'inputs': {}, 'chains': [
            {'kind': k, 'state': 'SUCCEEDED', 'detail': {'report_url': 'https://old/'+k,
             'publication': {'report_url': 'https://old/'+k}}} for k in ('model_accuracy', 'compile_alignment')],
            'notice': '', 'report_url': 'https://li.feishu.cn/docx/one-doc'}
        text = json.dumps(render(task), ensure_ascii=False)
        self.assertNotIn('https://old/', text)
        self.assertEqual(text.count('https://li.feishu.cn/docx/one-doc'), 1)
        self.assertIn('同一份评测报告', text)


if __name__ == '__main__':
    unittest.main()
