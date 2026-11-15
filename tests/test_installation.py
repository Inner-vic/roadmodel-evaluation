import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from pcm_eval.common import load_json
from pcm_eval.installation import Installation, export_agent_config, export_server_config, validate_request, validate_runtime_binding
from pcm_eval.installation_browser import initialization_prompt
from pcm_eval.space_api import OfficialSpaces


def request():
    return {'schema':'roadmodel.installation-request/v1',
        'account':{'lpai_account':'engineer@example.com','lpai_login':'engineer','feishu_user_id':'ou_engineer','feishu_app_id':'cli_testapp'},
        'template':{'id':'roadmodel-a100-v1','namespace':'team','cluster':'approved-cluster','project':'C000002',
            'resource_queue':'reviewed-queue','image':'registry.example.com/approved/roadmodel:v1',
            'resources':{'gpuType':'A100-SHARE','resourceQueue':'reviewed-queue','resourceComboId':'26','resourceComboNum':1},
            'accesses':{'visibility':'personal','ports':[22,8888]},
            'idle_shutdown':{'enabled':True,'minutes':240},'ssh_host':'approved-host'},
        'ssh':{'credential_name':'engineer-existing-key','identity_file':'/existing/approved/key'}}


def space(name, owner='engineer@example.com', uid='12345678-abcd'):
    r=request();t=r['template']
    return {'metadata':{'name':name,'namespace':'team','uid':uid,
        'labels':{'lpai.org/visible':'personal','lpai.org/resourceQueue':'reviewed-queue'},
        'annotations':{'lpai.org/owner':owner,'lpai.org/gems-project':'C000002'}},
        'spec':{'clusterName':'approved-cluster','image':t['image'],'resources':t['resources'],
            'autoCulling':{'enable':True,'windowMinutes':'240'},'sshSecretName':'engineer-existing-key',
            'accesses':[{'name':'ssh','pubkeyName':'engineer-existing-key','address':'ssh -p 30331 root@approved-host'}],
            'dataVols':[]},'status':{'phase':'stopped'}}


def identity(user='ou_engineer',app='cli_testapp'):
    return {'open_id':user,'app_id':app}


class InstallationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.installation=Installation(self.root/'installation')
        self.request=request();self.detail=self.installation.register(self.request,identity())

    def observation(self, **fields):
        return {'source':'CODEX_CUA','real_browser_call_completed':True,'lpai_login':'engineer',
            'namespace':'team','cluster':'approved-cluster','space_name':self.detail['space_name'],
            'exists':False,'observed_at':time.time(),**fields}

    def absent(self,args):raise ValueError('Cannot read space; absence has not been confirmed')

    def bind(self):
        return self.installation.observe(self.detail['key'],lambda args:{'data':space(self.detail['space_name'])})

    def test_idempotent_same_account_and_different_accounts_get_dedicated_names(self):
        self.assertEqual(self.installation.register(self.request,identity()),self.detail)
        other=request();other['account'].update(lpai_account='another@example.com',lpai_login='another',feishu_user_id='ou_another')
        second=self.installation.register(other,identity('ou_another'))
        self.assertNotEqual(self.detail['space_name'],second['space_name'])
        another_machine=Installation(self.root/'other-installation').register(self.request,identity())
        self.assertNotEqual(self.detail['space_name'],another_machine['space_name'])

    def test_wrong_verified_feishu_account_and_changed_template_are_rejected(self):
        with self.assertRaises(ValueError):self.installation.register(self.request,identity('ou_somebody_else'))
        with self.assertRaises(ValueError):self.installation.register(self.request,identity(app='cli_otherapp'))
        self.request['template']['image']='registry.example.com/other:v2'
        with self.assertRaises(ValueError):self.installation.register(self.request,identity())

    def test_copied_machine_state_requires_fresh_initialization(self):
        with patch('pcm_eval.installation.machine_identity',return_value='different-machine'):
            with self.assertRaises(ValueError):Installation(self.root/'installation')

    def test_cli_failure_never_means_absence_and_does_not_authorize_creation(self):
        result=self.installation.observe(self.detail['key'],self.absent)
        self.assertEqual(result['state'],'WAITING_BROWSER_OBSERVATION')
        with self.assertRaises(ValueError):self.installation.claim_creation(self.detail['key'])
        self.assertNotIn('create_intent',self.installation.get(self.detail['key']))

    def failed_create(self):
        from datetime import datetime,timezone
        self.installation.record_observation(self.detail['key'],self.observation())
        d=self.installation.claim_creation(self.detail['key'])
        d['create_intent']['requested_at']=datetime.fromtimestamp(time.time()-180,timezone.utc).isoformat()
        with self.installation.connect() as db:
            db.execute('update bindings set detail=? where key=?',(json.dumps(d),d['key']))
        return self.observation(create_intent_id=d['create_intent']['id'],http_status=500)

    def test_explicit_failed_create_retry_is_once_and_requires_full_official_absence(self):
        receipt=self.failed_create()
        full=lambda args:{'data':{'items':[],'total':0}}
        d=self.installation.retry_failed_creation(self.detail['key'],receipt,full)
        self.assertEqual(d['creation_retry']['http_status'],500)
        with self.assertRaises(ValueError):self.installation.retry_failed_creation(d['key'],receipt,full)

    def test_unknown_failure_partial_list_or_existing_space_never_authorize_retry(self):
        receipt=self.failed_create()
        for changes in ({'http_status':None},{'exists':True},{'observed_at':time.time()-600},{'namespace':'other'}):
            with self.assertRaises(ValueError):
                self.installation.retry_failed_creation(self.detail['key'],{**receipt,**changes},lambda args:{'items':[],'total':0})
        for listing in ({'items':[],'total':1},{'items':[space(self.detail['space_name'])],'total':1}):
            with self.assertRaises(ValueError):
                self.installation.retry_failed_creation(self.detail['key'],receipt,lambda args:listing)

    def test_browser_account_namespace_freshness_and_real_tool_proof_are_required(self):
        for changes in ({'lpai_login':'someone_else'},{'namespace':'other'},
                        {'real_browser_call_completed':False},{'exists':None},
                        {'observed_at':time.time()-400},{'observed_at':time.time()+90}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.installation.record_observation(self.detail['key'],self.observation(**changes))

    def test_concurrent_creation_is_claimed_once_and_unknown_result_is_not_repeated(self):
        self.installation.record_observation(self.detail['key'],self.observation())
        with ThreadPoolExecutor(max_workers=2) as executor:
            values=list(executor.map(lambda _:self.installation.claim_creation(self.detail['key']),range(2)))
        self.assertEqual(sum(x is not None for x in values),1)
        result=self.installation.observe(self.detail['key'],self.absent)
        self.assertEqual(result['state'],'CREATE_UNKNOWN')
        self.assertIsNone(self.installation.claim_creation(self.detail['key']))
        self.assertEqual(self.installation.get(self.detail['key'])['create_intent'],next(x for x in values if x)['create_intent'])

    def test_stale_writer_cannot_erase_original_creation_intent(self):
        stale=copy.deepcopy(self.detail)
        self.installation.record_observation(self.detail['key'],self.observation())
        claimed=self.installation.claim_creation(self.detail['key'])
        self.installation.save(stale)
        self.assertEqual(self.installation.get(self.detail['key'])['create_intent'],claimed['create_intent'])

    def test_capacity_rejection_is_explicit_and_does_not_retry_or_delete(self):
        self.installation.record_observation(self.detail['key'],self.observation())
        claimed=self.installation.claim_creation(self.detail['key'])
        receipt={'source':'CODEX_CUA','real_browser_call_completed':True,'space_name':claimed['space_name'],
            'create_intent_id':claimed['create_intent']['id'],'outcome':'CAPACITY_EXCEEDED','changed':False,'observed_at':time.time()}
        self.installation.record_creation_result(claimed['key'],receipt)
        result=self.installation.observe(claimed['key'],self.absent)
        self.assertEqual(result['state'],'BLOCKED_CAPACITY')
        self.assertIsNone(self.installation.claim_creation(claimed['key']))
        receipt['create_intent_id']='another-intent'
        with self.assertRaises(ValueError):self.installation.record_creation_result(claimed['key'],receipt)

    def test_exact_official_owner_and_template_are_required_before_binding(self):
        for mutate,expected in [(lambda s:s['metadata']['annotations'].update({'lpai.org/owner':'other@example.com'}),'owner'),
            (lambda s:s['spec']['autoCulling'].update(windowMinutes='60'),'four_hour_shutdown'),
            (lambda s:s['spec']['resources'].update(resourceComboId='unreviewed'),'resources'),
            (lambda s:s['spec'].update(image='registry.example.com/wrong:v2'),'image')]:
            raw=space(self.detail['space_name']);mutate(raw)
            result=self.installation.observe(self.detail['key'],lambda args:raw)
            self.assertEqual(result['state'],'CONFIGURATION_MISMATCH');self.assertIn(expected,result['issues'])
            self.assertFalse(Path(self.detail['binding_file']).exists())

    def test_actual_UID_and_port_are_bound_and_no_dataset_or_keys_are_copied(self):
        bound=self.bind();self.assertEqual(bound['state'],'BOUND')
        self.assertEqual(bound['profile']['space_uid'],'12345678-abcd')
        self.assertEqual(bound['profile']['ssh_port'],30331)
        self.assertEqual(bound['profile']['identity_file'],'/existing/approved/key')
        self.assertFalse(bound['stop_after_evaluation'])
        self.assertEqual(load_json(bound['binding_file']),bound)
        self.assertFalse(any(p.suffix in ('.npy','.jpg','.pth','.pem') for p in self.root.rglob('*')))

    def test_replaced_UID_or_deleted_bound_space_is_not_silently_recreated(self):
        self.bind()
        result=self.installation.observe(self.detail['key'],lambda args:space(self.detail['space_name'],uid='99999999-ffff'))
        self.assertEqual(result['state'],'CONFIGURATION_MISMATCH');self.assertIn('uid',result['issues'])
        result=self.installation.observe(self.detail['key'],self.absent)
        self.assertEqual(result['state'],'BOUND_UNAVAILABLE')
        self.installation.record_observation(self.detail['key'],self.observation())
        self.assertIsNone(self.installation.claim_creation(self.detail['key']))

    def test_adoption_does_not_create_a_replacement(self):
        req=request();req['account'].update(lpai_account='legacy@example.com',lpai_login='legacy',feishu_user_id='ou_legacy')
        req['adopt_existing']={'name':'known-existing','uid':'12345678-abcd'}
        registered=self.installation.register(req,identity('ou_legacy'))
        obs=self.observation(space_name='known-existing',lpai_login='legacy')
        self.installation.record_observation(registered['key'],obs)
        with self.assertRaises(ValueError):self.installation.claim_creation(registered['key'])

    def test_exported_runtime_requires_same_machine_account_and_space(self):
        bound=self.bind();target=self.root/'agent.json'
        config=export_agent_config(bound,{'release':'reviewed-release','feishu_app_id':'cli_testapp'},target,local_state=self.root/'agent-state')
        server=export_server_config(bound,{'feishu_app_id':'cli_testapp'},self.root/'server.json')
        for field in ('owner','agent_id','profile'):self.assertEqual(server[field],config[field])
        validate_runtime_binding(config)
        for change in ('owner','space','installation'):
            wrong=copy.deepcopy(config)
            if change=='owner':wrong['owner']='ou_else'
            if change=='space':wrong['profile']['space_uid']='different'
            if change=='installation':wrong['installation_id']='different'
            with self.assertRaises(ValueError):validate_runtime_binding(wrong)
        with patch('pcm_eval.installation.machine_identity',return_value='other-machine'):
            with self.assertRaises(ValueError):validate_runtime_binding(config)
        with self.assertRaises(FileExistsError):export_agent_config(bound,{'feishu_app_id':'cli_testapp'},target,local_state=self.root/'agent-state')
        with self.assertRaises(ValueError):export_agent_config(bound,{'feishu_app_id':'cli_wrongapp'},self.root/'wrong.json',local_state=self.root/'agent-state')

    def test_initialization_does_not_accept_secrets_or_one_hour_template(self):
        for mutate in (lambda r:r.update(api_key='forbidden'),
                       lambda r:r['template']['resources'].update(request={'token':'forbidden'}),
                       lambda r:r['template']['idle_shutdown'].update(minutes=60)):
            req=request();mutate(req)
            with self.assertRaises(ValueError):validate_request(req)

    def test_first_dataset_is_checked_against_its_submitted_version(self):
        from tests.test_dataset_mount import DatasetMountTest, URL
        fixture=DatasetMountTest();fixture.setUp()
        req=request();req['dataset_ref']=URL
        req['account'].update(feishu_user_id='ou_dataset')
        registered=self.installation.register(req,identity('ou_dataset'))
        raw=space(registered['space_name'])
        def query(args):return {'data':raw if args[0]=='devspaces' else fixture.raw}
        blocked=self.installation.observe(registered['key'],query)
        self.assertEqual(blocked['state'],'CONFIGURATION_MISMATCH')
        self.assertIn('initial_dataset_mount',blocked['issues'])
        raw['spec']['dataVols']=[fixture.volume()]
        self.assertEqual(self.installation.observe(registered['key'],query)['state'],'BOUND')

    def test_runtime_checks_initialized_LPAI_owner_even_when_UID_is_unchanged(self):
        bound=self.bind();wrong=space(self.detail['space_name'],owner='other@example.com')
        with self.assertRaises(ValueError):OfficialSpaces(bound['profile'],invoke=lambda args:wrong).read()

    def test_creation_prompt_requires_saved_intent_and_current_account_and_no_dataset_copy(self):
        with self.assertRaises(ValueError):initialization_prompt('create',self.detail)
        self.installation.record_observation(self.detail['key'],self.observation())
        claimed=self.installation.claim_creation(self.detail['key']);text=initialization_prompt('create',claimed)
        for expected in ('engineer@example.com','240','不下载或复制数据','仅创建一次'):
            self.assertIn(expected,text)

    def test_api_capacity_receipt_requires_exact_owner_and_zero_writes(self):
        self.installation.record_observation(self.detail['key'],self.observation())
        claimed=self.installation.claim_creation(self.detail['key'])
        receipt={'source':'LPAI_NOTEBOOK_API','real_api_call_completed':True,'platform_writes':0,
            'space_name':claimed['space_name'],'create_intent_id':claimed['create_intent']['id'],
            'owner':'engineer@example.com','namespace':'team','cluster':'approved-cluster',
            'outcome':'CAPACITY_EXCEEDED','changed':False,'observed_at':time.time()}
        for invalid in ({'owner':'other@example.com'},{'platform_writes':1},{'cluster':'other-cluster'}):
            with self.assertRaises(ValueError):
                self.installation.record_creation_result(claimed['key'],{**receipt,**invalid})
        self.assertEqual(self.installation.record_creation_result(claimed['key'],receipt)['state'],'BLOCKED_CAPACITY')


if __name__=='__main__':unittest.main()
