import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from pcm_eval.common import digest, json_digest, load_json, save_json
from pcm_eval.compile_preparation import prepare_compile
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.evaluation_entry import render
from pcm_eval.combined_delivery import build_combined
from pcm_eval.evaluation_source import prepare as prepare_source
from tools.delivery_demo_plugin import card_route
from tools.evaluation_worker import evaluate
from tools.evaluation_agent import publish
from tests.test_delivery_combined import reference

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://li.feishu.cn/record/PUBREF000000000000000000001'
OCC = 'https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/roads?version=occ'
VISION = 'https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/roads?version=vision'


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = self.root/'config.json'
        save_json(self.cfg, {'owner': 'ou_owner', 'agent_id': 'agent', 'python': '/python'})
        self.rt = EvaluationRuntime(self.root/'gateway/roadmodel/evaluation', config_file=self.cfg)
        self.owner, self.session = 'ou_owner', 'feishu:oc_test:ou_owner'
        self.task = self.rt.open(self.owner, self.session)
        self.rt.bind_message(self.task['id'], self.owner, 'om_card', self.session)
        self.event = {'action': 'roadmodel_submit', 'event_id': 'e1', 'operator_id': self.owner,
                      'message_id': 'om_card', 'chat_id': 'oc_test',
                      'value': {'task_id': self.task['id'], 'nonce': self.task['nonce']},
                      'form_value': {'model_delivery_url': URL, 'dataset_url': OCC}}

    def tearDown(self):
        self.temp.cleanup()

    def workflow(self):
        value = {'schema': 'roadmodel.delivery-workflow/v1', 'run_id': self.task['id'], 'scope': 'full',
                 'inputs': {'model_delivery_url': URL, 'dataset_ref': OCC, 'vision_dataset_ref': VISION},
                 'jobs': [{'kind': kind, 'version': 'v3.2.2', 'ready': False, 'issues': ['missing GT'],
                           'request': None, 'plan': None} for kind in ('model_accuracy', 'compile_alignment')]}
        value['workflow_sha256'] = json_digest(value)
        return value

    def test_card_has_two_fields_full_scope_and_no_demo_selector(self):
        card = render(self.task)
        form = next(e for e in card['body']['elements'] if e['tag'] == 'form')
        self.assertEqual([e['name'] for e in form['elements'] if e['tag'] == 'input'], ['model_delivery_url', 'dataset_url'])
        text = json.dumps(card, ensure_ascii=False)
        self.assertIn('全部评测项', text)
        self.assertIn('完整验证范围', text)
        self.assertNotIn('2帧', text)

    def test_space_action_readback_requeues_original_submission_without_replaying_action(self):
        import time
        cfg = load_json(self.cfg)
        cfg['profile'] = {'space_name': 'mine', 'namespace': 'team', 'ssh_host': 'approved-host',
                          'environment_template': {'accesses': {'ssh_credential': 'existing-key'}}}
        save_json(self.cfg, cfg, overwrite=True)
        self.rt.submit(self.owner, self.session, self.event)
        context = {'profile': cfg['profile'], 'dataset_ref': OCC}
        queued = self.rt.request_space_action(self.task['id'], 'start', context)
        row = self.rt.claim_space_action('agent', self.owner)
        self.assertEqual(row['id'], queued['action_id'])
        self.assertIsNone(self.rt.claim_space_action('agent', self.owner))
        self.rt.complete_space_action('agent', row['id'], {'state': 'ACTION_RETURNED', 'real_browser_call_completed': True})
        observed = self.rt.claim_space_action('agent', self.owner)
        self.assertEqual(observed['action'], 'observe')
        page = {'source': 'LPAI_UI', 'space_name': 'mine', 'namespace': 'team', 'observed_at': time.time(),
                'exists': True, 'space': {'runtime_access': {'host': 'approved-host', 'port': 32000, 'credential_name': 'existing-key'}}}
        self.rt.complete_space_action('agent', observed['id'], {'state': 'OBSERVED', 'real_browser_call_completed': True, 'observation': page})
        self.assertEqual(self.rt.space_observation()['space']['runtime_access']['port'], 32000)
        with self.assertRaises(ValueError):
            self.rt.complete_space_action('agent', row['id'], {'state': 'ACTION_RETURNED', 'real_browser_call_completed': True})
        page['space']['runtime_access']['host'] = 'other-host'
        with self.assertRaises(ValueError):
            self.rt.record_space_observation(page)

    def test_path_callback_is_idempotent_and_recovers_route_without_schema(self):
        tid = self.rt.submit(self.owner, self.session, self.event)
        self.assertEqual(self.rt.submit(self.owner, self.session, self.event), tid)
        self.assertEqual(card_route(self.root/'gateway', self.session, self.event),
                         ('evaluation_entry', 'roadmodel-evaluation/v1'))
        self.rt.tick(spawn=lambda *a, **k: self.fail('Material read is not inference'))
        row = self.rt.claim_materials('agent', self.owner)
        self.assertEqual(json.loads(row['request'])['dataset_ref'], OCC)
        with self.rt.connect() as c:
            self.assertEqual(c.execute('select count(*) from materials').fetchone()[0], 1)

    def test_official_space_observation_is_UID_bound_and_does_not_require_browser_flag(self):
        import time
        cfg=load_json(self.cfg)
        cfg.update(space_backend='official_cli',browser_backend_verified=False,
                   profile={'space_name':'mine','space_uid':'real-uid','namespace':'team','ssh_host':'approved-host',
                            'environment_template':{'accesses':{'ssh_credential':'existing'}}})
        save_json(self.cfg,cfg,overwrite=True)
        observation={'source':'LPAI_OFFICIAL_API','space_name':'mine','namespace':'team','observed_at':time.time(),'exists':True,
                     'space':{'metadata':{'uid':'real-uid'},'status':{'phase':'ready'},'verification':{'source':'LPAI_OFFICIAL_API'},
                              'runtime_access':{'host':'approved-host','port':30331,'credential_name':'existing'}}}
        self.rt.record_space_observation(observation)
        self.assertEqual(self.rt.space_observation()['space']['metadata']['uid'],'real-uid')
        observation['space']['metadata']['uid']='other'
        with self.assertRaises(ValueError):self.rt.record_space_observation(observation)

    def test_pending_API_start_is_retained_without_claiming_it_as_ready_or_browser_success(self):
        cfg=load_json(self.cfg)
        cfg.update(space_backend='official_cli',profile={'space_name':'mine','namespace':'team'})
        save_json(self.cfg,cfg,overwrite=True)
        self.rt.submit(self.owner,self.session,self.event)
        self.rt.request_space_action(self.task['id'],'start',{'profile':cfg['profile'],'dataset_ref':OCC})
        action=self.rt.claim_space_action('agent',self.owner)
        self.rt.complete_space_action('agent',action['id'],{'state':'WAITING_APPROVAL',
            'provider':'LPAI_OFFICIAL_API','real_api_call_completed':True})
        with self.rt.connect() as db:
            self.assertEqual(db.execute('select state from space_actions where id=?',(action['id'],)).fetchone()['state'],'WAITING_APPROVAL')
        next_action=self.rt.claim_space_action('agent',self.owner)
        self.assertEqual(next_action['action'],'observe')

    def test_API_pending_is_reconciled_by_authoritative_ready_readback(self):
        import time
        cfg=load_json(self.cfg)
        cfg.update(space_backend='official_cli',profile={'space_name':'mine','space_uid':'real-uid','namespace':'team',
            'ssh_host':'approved-host','environment_template':{'accesses':{'ssh_credential':'existing'}}})
        save_json(self.cfg,cfg,overwrite=True);self.rt.submit(self.owner,self.session,self.event)
        self.rt.request_space_action(self.task['id'],'start',{'profile':cfg['profile'],'dataset_ref':OCC})
        action=self.rt.claim_space_action('agent',self.owner)
        self.rt.complete_space_action('agent',action['id'],{'state':'WAITING_APPROVAL','provider':'LPAI_OFFICIAL_API','real_api_call_completed':True})
        observation={'source':'LPAI_OFFICIAL_API','space_name':'mine','namespace':'team','observed_at':time.time(),'exists':True,
            'space':{'metadata':{'uid':'real-uid'},'status':{'phase':'ready'},'verification':{'source':'LPAI_OFFICIAL_API'},
                     'runtime_access':{'host':'approved-host','port':30331,'credential_name':'existing'}}}
        self.rt.record_space_observation(observation)
        with self.rt.connect() as db:
            row=db.execute('select state,detail from space_actions where id=?',(action['id'],)).fetchone()
        self.assertEqual(row['state'],'OBSERVED')
        self.assertEqual(json.loads(row['detail'])['platform_request_reconciled']['phase'],'ready')

    def test_API_pending_response_with_already_ready_readback_is_not_left_waiting(self):
        import time
        cfg=load_json(self.cfg)
        cfg.update(space_backend='official_cli',profile={'space_name':'mine','space_uid':'real-uid','namespace':'team',
            'ssh_host':'approved-host','environment_template':{'accesses':{'ssh_credential':'existing'}}})
        save_json(self.cfg,cfg,overwrite=True);self.rt.submit(self.owner,self.session,self.event)
        self.rt.request_space_action(self.task['id'],'start',{'profile':cfg['profile'],'dataset_ref':OCC})
        action=self.rt.claim_space_action('agent',self.owner)
        observation={'source':'LPAI_OFFICIAL_API','space_name':'mine','namespace':'team','observed_at':time.time(),'exists':True,
            'space':{'metadata':{'uid':'real-uid'},'status':{'phase':'ready'},'verification':{'source':'LPAI_OFFICIAL_API'},
                     'runtime_access':{'host':'approved-host','port':30331,'credential_name':'existing'}}}
        self.rt.complete_space_action('agent',action['id'],{'state':'WAITING_APPROVAL','provider':'LPAI_OFFICIAL_API',
            'real_api_call_completed':True,'observation':observation})
        with self.rt.connect() as db:self.assertEqual(db.execute('select state from space_actions where id=?',(action['id'],)).fetchone()['state'],'OBSERVED')

    def test_invalid_API_readback_does_not_commit_an_action_as_successful(self):
        import time
        cfg=load_json(self.cfg);cfg.update(space_backend='official_cli',profile={'space_name':'mine','space_uid':'real-uid','namespace':'team'})
        save_json(self.cfg,cfg,overwrite=True);self.rt.submit(self.owner,self.session,self.event)
        self.rt.request_space_action(self.task['id'],'start',{'profile':cfg['profile'],'dataset_ref':OCC})
        action=self.rt.claim_space_action('agent',self.owner)
        observation={'source':'LPAI_OFFICIAL_API','space_name':'mine','namespace':'team','observed_at':time.time(),'exists':True,
            'space':{'metadata':{'uid':'other'},'status':{'phase':'ready'},'verification':{'source':'LPAI_OFFICIAL_API'}}}
        with self.assertRaises(ValueError):self.rt.complete_space_action('agent',action['id'],{'state':'OBSERVED',
            'provider':'LPAI_OFFICIAL_API','real_api_call_completed':True,'observation':observation})
        with self.rt.connect() as db:self.assertEqual(db.execute('select state from space_actions where id=?',(action['id'],)).fetchone()['state'],'EXECUTING')

    def test_other_user_and_changed_callback_are_rejected(self):
        event = copy.deepcopy(self.event)
        event['operator_id'] = 'ou_other'
        with self.assertRaises(ValueError):
            self.rt.submit(self.owner, self.session, event)
        self.rt.submit(self.owner, self.session, self.event)
        event = copy.deepcopy(self.event)
        event['form_value']['dataset_url'] = '/different'
        with self.assertRaises(ValueError):
            self.rt.submit(self.owner, self.session, event)

    def test_space_query_failure_still_prepares_compile_and_a_reportable_workflow(self):
        profile = {'namespace': 'team', 'space_name': 'mine', 'space_uid': 'space', 'cluster': 'cluster'}
        with patch('pcm_eval.evaluation_source.prepare_delivery', return_value=self.workflow()):
            workflow = prepare_source({'model_delivery_url': URL, 'dataset_ref': OCC},
                self.task['id'], profile=profile, query=lambda args: (_ for _ in ()).throw(ValueError('space missing')),
                probe=lambda plan: self.fail('Unavailable space must not be probed'))
        self.assertEqual(len(workflow['jobs']), 2)
        self.assertEqual(workflow['dataset_preparation']['state'], 'WAITING_SPACE')
        self.assertTrue(any('space missing' in issue for issue in workflow['jobs'][0]['issues']))
        self.assertFalse(any('space missing' in issue for issue in workflow['jobs'][1]['issues']))

    def test_full_entry_rejects_small_scope_or_modified_workflow(self):
        self.rt.submit(self.owner, self.session, self.event)
        self.rt.tick()
        row = self.rt.claim_materials('agent', self.owner)
        workflow = self.workflow()
        workflow['scope'] = 'validation'
        with self.assertRaises(ValueError):
            self.rt.complete_materials('agent', self.task['id'], row['digest'], {'workflow': workflow})

    def test_bench_absence_prepares_materials_without_either_inference(self):
        workflow = self.workflow()
        result_workflow, result = evaluate(workflow, self.root/'run', {'bench_available': False},
                            executor=lambda *args: self.fail('No inference for blocked materials'))
        self.assertEqual(result['chains'][1]['state'], 'BLOCKED')
        self.assertFalse(result['chains'][1]['preparation']['bench_contacted'])
        spec = load_json(ROOT/'configs/report_template.json')
        xml, data = build_combined(result_workflow, result, reference(spec), spec)
        self.assertEqual(data['scope'], 'full')
        self.assertFalse(data['all_evaluations_complete'])
        self.assertIn('待评测', xml)

    def test_model_can_finish_while_compile_waits_and_success_is_not_replayed(self):
        workflow = self.workflow()
        model = workflow['jobs'][0]
        model.update(ready=True, issues=[], environment='lpai_devspace', request={'execution': {}},
                     plan={'run_id': 'model'})
        workflow['dataset'] = {'mount_receipt': {'state': 'MOUNT_VERIFIED'}}
        workflow.pop('workflow_sha256')
        workflow['workflow_sha256'] = json_digest(workflow)
        calls = []
        def execute(plan, target):
            calls.append(plan)
            target.mkdir()
            path = save_json(target/'result.json', {'evaluation_type': 'model_accuracy', 'version_name': 'v3.2.2',
                             'execution': {'state': 'succeeded'}, 'collection': {'status': 'complete'}})
            return {'state': 'succeeded', 'result_file': str(path)}
        cfg = {'bench_available': False, 'model_space': {'host': 'space', 'user': 'root', 'port': 22, 'identity_file': '/key'}}
        with patch('tools.evaluation_worker.build_plan', return_value={'run_id': 'model', 'plan_sha256': 'model'}):
            for _ in range(2):
                _, result = evaluate(copy.deepcopy(workflow), self.root/'run', cfg, executor=execute,
                                     reporter=lambda *args: {'report_json': 'report.json'})
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['state'], 'PARTIAL')
        self.assertEqual(result['chains'][0]['state'], 'SUCCEEDED')

    def test_report_queue_is_single_claim_and_requires_requester_readback(self):
        self.rt.submit(self.owner, self.session, self.event)
        self.rt.tick()
        run = self.rt.root/'runs'/self.task['id']
        workflow = self.workflow()
        save_json(run/'workflow.json', workflow)
        spec = load_json(ROOT/'configs/report_template.json')
        result = {'state': 'WAITING_BENCH', 'chains': [
            {'kind': j['kind'], 'state': 'BLOCKED', 'issues': j['issues']} for j in workflow['jobs']]}
        xml, data = build_combined(workflow, result, reference(spec), spec)
        report = {'report_xml': str(run/'report.xml'), 'report_json': str(run/'report.json'), 'source_sha256': 'source'}
        (run/'report.xml').write_text(xml, encoding='utf-8')
        save_json(run/'report.json', data)
        self.rt.queue_publication(self.task['id'], report, result)
        row = self.rt.claim_publication('agent', self.owner)
        self.assertIsNone(self.rt.claim_publication('agent', self.owner))
        detail = {'state': 'VERIFIED', 'table_count': 5, 'all_metric_cells_match': True,
                  'task_sha256': data['task_sha256'], 'workflow_sha256': workflow['workflow_sha256'],
                  'document_id': 'newdoc', 'report_url': 'https://li.feishu.cn/docx/newdoc',
                  'requester_read_verified': True}
        self.rt.complete_publication('agent', self.task['id'], row['digest'], detail)

        current = self.rt.get(self.task['id'], self.owner)
        self.assertEqual(current['state'], 'WAITING_BENCH')
        self.assertEqual(current['report_url'], detail['report_url'])
        self.rt.complete_publication('agent', self.task['id'], row['digest'], detail)

    def test_waiting_bench_then_available_keeps_plan_and_does_not_repeat_model(self):
        workflow=self.workflow()
        for job in workflow['jobs']:
            job.update(ready=True,issues=[],plan={'run_id':job['kind'],'plan_sha256':job['kind']},
                       request={'execution':{}},native_adapter='lpai_mounted_dataset')
        workflow.pop('workflow_sha256');workflow['workflow_sha256']=json_digest(workflow)
        calls=[]
        def execute(plan,target):
            calls.append(plan['run_id']);target.mkdir()
            result=save_json(target/'result.json',{'evaluation_type':plan['run_id'],'version_name':'v3.2.2',
                'execution':{'state':'succeeded'},'collection':{'status':'complete'}})
            return {'state':'succeeded','result_file':str(result)}
        with patch('tools.evaluation_worker.prepare_compile',return_value={'state':'PREPARED','issues':[]}):
            for available in (False,True,False):
                _,result=evaluate(copy.deepcopy(workflow),self.root/'switch',{'bench_available':available},
                                  executor=execute,reporter=lambda *a:{'report_json':'fixture'})
        self.assertEqual(calls,['model_accuracy','compile_alignment'])
        self.assertEqual(result['state'],'SUCCEEDED')
        self.assertEqual(load_json(self.root/'switch/effective/workflow.json'),workflow)

    def test_verified_report_accepts_later_compile_result_and_same_payload_is_idempotent(self):
        self.test_report_queue_is_single_claim_and_requires_requester_readback()
        run=self.rt.root/'runs'/self.task['id']
        before=self.rt.get(self.task['id'],self.owner)['state']
        with self.rt.connect() as db:old=dict(db.execute('select * from publications').fetchone())
        payload=json.loads(old['payload']);result=load_json(run/'progress.json')
        report={'report_xml':str(run/'report.xml'),'report_json':str(run/'report.json'),'source_sha256':'source'}
        self.rt.queue_publication(self.task['id'],report,result)
        self.assertEqual(self.rt.get(self.task['id'],self.owner)['state'],before)
        xml=Path(report['report_xml']);xml.write_text(xml.read_text(encoding='utf-8')+'<p>offline update fixture</p>',encoding='utf-8')
        report['source_sha256']='new-source'
        self.rt.queue_publication(self.task['id'],report,result)
        row=self.rt.claim_publication('agent',self.owner)
        self.assertNotEqual(row['digest'],old['digest'])
        self.assertEqual(self.rt.get(self.task['id'],self.owner)['report_url'],'https://li.feishu.cn/docx/newdoc')
        self.assertEqual(json.loads(row['payload'])['workflow_sha256'],payload['workflow_sha256'])
        self.assertTrue((run/'superseded-publications'/(old['digest']+'.json')).is_file())

    def test_publication_failure_retains_scores_and_does_not_dispatch(self):
        self.test_report_queue_is_single_claim_and_requires_requester_readback()
        with self.rt.connect() as c:
            c.execute("update publications set state='WRITING'")
            row = dict(c.execute('select * from publications').fetchone())
        self.rt.complete_publication('agent', self.task['id'], row['digest'], {'state': 'NEEDS_REVIEW'})
        self.rt.tick(spawn=lambda *a, **k: self.fail('Publication failure must not rerun inference'))
        self.assertEqual(self.rt.get(self.task['id'], self.owner)['state'], 'NEEDS_REVIEW')


class CompilePreparationTest(unittest.TestCase):
    def test_compiler_artifacts_and_cached_receipt_are_content_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            onnx = root/'model.onnx'
            onnx.write_bytes(b'onnx-fixture')
            archives = []
            for stage in range(5):
                path = root/f'sub{stage}.tar.gz'
                raw = json.dumps({'elf_path': 'SubNetOps.elf', 'input': [], 'output': []}).encode()
                with tarfile.open(path, 'w:gz') as tar:
                    info = tarfile.TarInfo('model_info.json')
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
                archives.append({'stage': stage, 'path': str(path), 'sha256': digest(path)})
            job = {'version': 'v3.2.2', 'request': {'model': {'onnx': str(onnx), 'onnx_sha256': digest(onnx), 'subnet_archives': archives}}}
            with patch('pcm_eval.compile_preparation.chain_contract', return_value={'edges': []}):
                value = prepare_compile(job, root/'output')
            self.assertEqual(value['state'], 'PREPARED')
            self.assertFalse(value['bench_contacted'])
            onnx.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare_compile(job, root/'output')


if __name__ == '__main__':
    unittest.main()
