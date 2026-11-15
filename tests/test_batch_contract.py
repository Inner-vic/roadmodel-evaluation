import copy
import json
import shutil
import uuid
import time
from contextlib import contextmanager
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pcm_eval.common import load_json
from pcm_eval.validation import validate_result
from pcm_eval.intake import resolve_submission,write_resolution
from pcm_eval.delivery import inspect_delivery
from pcm_eval.workflow import run_report_batch
from pcm_eval.placement import migration_plan

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'evidence/model-evaluation/native-v3.2.2-smoke-20261027/normalized_result.json'


@contextmanager
def test_directory():
    base=(ROOT/'evidence/test-batch-contract').resolve()
    base.mkdir(exist_ok=True)
    path=base/('case-'+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved=path.resolve()
        if not resolved.is_relative_to(base) or resolved==base:
            raise ValueError('Test cleanup target escaped its workspace')
        for attempt in range(6):
            try:
                shutil.rmtree(resolved)
                break
            except PermissionError as error:
                if getattr(error,'winerror',None)!=32 or attempt==5:raise
                time.sleep(.1*(attempt+1))


class ContractTests(unittest.TestCase):
    def test_real_result_and_bin_unit_are_accepted(self):
        path=ROOT/'evidence/model-evaluation/remaining-heads-20261027/collected/trajectory-01-recovered/normalized_result-reviewed.json'
        self.assertTrue(validate_result(load_json(path))['valid'])

    def test_nan_bool_duplicate_metric_and_changed_scope_are_rejected(self):
        original=load_json(RESULT)
        mutations=[lambda x:x['metrics'][0].update(value=float('nan')),
                   lambda x:x['metrics'][0].update(value=True),
                   lambda x:x['metrics'].append(copy.deepcopy(x['metrics'][0])),
                   lambda x:x['coverage']['observed'].update(model_samples=1)]
        for mutate in mutations:
            data=copy.deepcopy(original);mutate(data)
            self.assertFalse(validate_result(data)['valid'])

    def test_missing_or_changed_source_is_rejected(self):
        data=load_json(RESULT);data['artifacts'][0]['sha256']='0'*64
        self.assertFalse(validate_result(data)['valid'])
        data['artifacts'][0]['path']='missing-source-file'
        self.assertFalse(validate_result(data)['valid'])

    def test_model_link_resolves_only_to_matching_registered_model(self):
        value=resolve_submission({'schema':'roadmodel.submission/v1','jobs':[{'id':'m','kind':'model_accuracy',
                'model_ref':'https://li.feishu.cn/record/PUBREF000000000000000000001'}]})
        self.assertTrue(value['ready'])
        self.assertEqual(value['jobs'][0]['profile'],'model-v322-small')
        value=resolve_submission({'schema':'roadmodel.submission/v1','jobs':[{'id':'m','kind':'compile_alignment',
                'model_ref':'https://li.feishu.cn/record/PUBREF000000000000000000001'}]})
        self.assertFalse(value['ready'],'Must not substitute the separately supplied v3.2.3 compile model')

    def test_unknown_dataset_and_duplicate_job_are_rejected(self):
        job={'id':'m','profile':'model-v322-small','dataset_ref':'another-release-dataset'}
        self.assertFalse(resolve_submission({'schema':'roadmodel.submission/v1','jobs':[job]})['ready'])
        with self.assertRaises(ValueError):
            resolve_submission({'schema':'roadmodel.submission/v1','jobs':[job,job]})

    def test_replay_cannot_silently_change_measured_scope(self):
        job={'id':'m','profile':'model-v322-small','scope':{'max_frames':1}}
        value=resolve_submission({'schema':'roadmodel.submission/v1','jobs':[job]})
        self.assertFalse(value['ready'])
        self.assertEqual(value['issues'][0]['code'],'existing_scope_frozen')

    def test_gateway_refuses_public_or_invalid_tunnel_configuration(self):
        from pcm_eval.gateway import GatewayTunnel
        base={'ssh_alias':'workstation','board_host':'192.0.2.10','lock_file':'/locks/bench.lock'}
        for port in [80,True,70000]:
            with self.assertRaises(ValueError):GatewayTunnel(dict(base,local_port=port))
        with self.assertRaises(ValueError):GatewayTunnel(dict(base,ssh_alias='-oProxyCommand=bad'))

    def test_prepared_jobs_export_independently_verifiable_execution_plans(self):
        from pcm_eval.planning import verify_plan
        with test_directory() as tmp:
            output=tmp/'resolution.json'
            value=write_resolution(ROOT/'examples/prepare_registered.json',output)
            self.assertTrue(value['ready'])
            for job in value['jobs']:
                verify_plan(load_json(job['plan_file']))
            with self.assertRaises(ValueError):write_resolution(ROOT/'examples/prepare_registered.json',output)

    def test_batch_business_layout_preserves_data_and_portable_delivery(self):
        # Workspace output is needed by the native publisher's @./ paths.
        base=ROOT/'evidence/test-batch-contract';base.mkdir(exist_ok=True)
        with test_directory() as tmp:
            out=Path(tmp)/'batch'
            receipt=run_report_batch(load_json(ROOT/'examples/batch_reports.json'),out)
            self.assertEqual(receipt['state'],'succeeded')
            model=receipt['jobs'][0];compile_job=receipt['jobs'][1]
            data=load_json(model['report']['report_json'])
            self.assertEqual(len(data['all_metrics']),426)
            xml=Path(model['report']['report_xml']).read_text(encoding='utf-8')
            tree=ET.fromstring('<root>'+xml+'</root>')
            headers=[[ ''.join(c.itertext()) for c in t.find('thead').findall('tr/th')]
                     for t in tree.iter('table') if t.find('thead') is not None]
            self.assertIn(['类别','IoU（%）','precision（%）','recall（%）','Pillar IoU（%）'],headers)
            self.assertNotIn('v3.0.4',xml)
            compile_data=load_json(compile_job['report']['json'])
            self.assertEqual(len(compile_data['tensor_alignment']['records']),360)
            self.assertEqual(len(compile_data['ground_truth']['counts']),48)
            cxml=Path(compile_job['report']['xml']).read_text(encoding='utf-8')
            self.assertIn('ONNX fp16（%）',cxml)
            self.assertIn('本次板端 min（%）',cxml)
            moved=Path(tmp)/'moved'
            shutil.copytree(Path(model['delivery']['manifest']).parent,moved)
            self.assertEqual(inspect_delivery(moved)['state'],'verified')
            (moved/'report/report.json').write_text('{}')
            with self.assertRaises(ValueError):inspect_delivery(moved)

    def test_failed_preflight_does_not_generate_partial_report(self):
        base=ROOT/'evidence/test-batch-contract';base.mkdir(exist_ok=True)
        with test_directory() as tmp:
            receipt=run_report_batch({'schema':'roadmodel.submission/v1','jobs':[{'id':'m','profile':'missing'}]},Path(tmp)/'batch')
            self.assertEqual(receipt['state'],'preflight_failed')
            self.assertEqual(receipt['jobs'],[])

    def test_migration_cannot_claim_readiness_without_route_or_board_exclusivity(self):
        request=load_json(ROOT/'examples/reference_compile_multiscene.json')
        value=migration_plan(request,{}, {'ready':False})
        self.assertFalse(value['ready'])
        self.assertFalse(value['raw_dataset_copy_to_workstation'])
        self.assertEqual(value['request']['model']['onnx_sha256'],request['model']['onnx_sha256'])


if __name__=='__main__':unittest.main()
