import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcm_eval.common import load_json, save_json
from pcm_eval.space_api import OfficialSpaces, PROVIDER
from tests import test_dataset_mount as mount_fixtures
from pcm_eval.space_lifecycle import advance

URL = mount_fixtures.URL


class OfficialSpaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture = mount_fixtures.DatasetMountTest();fixture.setUp()
        self.asset, self.raw_asset, self.space = fixture.asset, fixture.raw, fixture.space
        self.space['spec'].update(clusterName='cluster', dataVols=[fixture.volume()],
            sshSecretName='existing', accesses=[{'name':'ssh','pubkeyName':'existing','address':'ssh -p 30331 root@approved-host'}])
        self.space['status']['phase'] = 'stopped'
        self.profile = {'namespace':'team','space_name':'mine','space_uid':'space-1','cluster':'cluster',
                        'ssh_host':'approved-host','ssh_port':9999,'lifecycle_provider':'official_cli',
                        'environment_template':{'namespace':'team','cluster':'cluster','image':'image',
                         'resources':{'gpu_type':'registered'},'accesses':{'ssh_credential':'existing'},
                         'idle_shutdown':{'enabled':True,'minutes':240}}}
        self.context = {'profile':self.profile,'asset':self.asset,'dataset_ref':URL,'intent_key':'frozen-intent'}
        self.calls = []
        self.reply = {'status':'success','data':{}}
        self.error = None

    def invoke(self, args):
        self.calls.append(args)
        if args[1] == 'get':return {'data':copy.deepcopy(self.space)}
        if self.error:raise self.error
        return self.reply

    def client(self):return OfficialSpaces(self.profile, invoke=self.invoke)
    def writes(self):return [a for a in self.calls if a[1] in ('start','stop')]
    def execute(self, action='start', path='action', **kw):
        return self.client().execute(action,self.context,self.root/path,task='frozen-task',**kw)

    def test_actual_UID_mount_and_current_API_SSH_port_are_used(self):
        data = self.client().read()
        self.assertEqual(data['runtime_access']['port'],30331)
        self.assertEqual(data['metadata']['uid'],'space-1')
        self.assertEqual(data['verification']['source'],PROVIDER)
        self.assertNotIn('accesses',data['spec'])

    def test_different_UID_namespace_or_SSH_credential_prevents_start(self):
        original = copy.deepcopy(self.space)
        for container,key,value in [('metadata','uid','another'),('metadata','namespace','another'),('spec','sshSecretName','another')]:
            self.space = copy.deepcopy(original);self.space[container][key] = value
            with self.assertRaises(ValueError):self.execute()
        self.assertEqual(self.writes(),[])

    def test_unknown_host_and_missing_port_are_rejected(self):
        for address in ('ssh -p 30331 root@foreign-host','ssh root@approved-host'):
            self.space['spec']['accesses'][0]['address'] = address
            with self.assertRaises(ValueError):self.client().read()

    def test_ready_or_starting_space_does_not_receive_another_start(self):
        for phase in ('ready','starting','pending','waiting'):
            self.space['status']['phase'] = phase
            result = self.execute(path=phase)
            self.assertEqual(result['state'],'OBSERVED')
            self.assertEqual(result['new_requests'],0)
        self.assertEqual(self.writes(),[])

    def test_requested_dataset_must_match_before_start(self):
        self.space['spec']['dataVols'][0]['version'] = 'old'
        result = self.execute()
        self.assertEqual(result['state'],'NEEDS_ASSISTANCE')
        self.assertEqual(self.writes(),[])

    def test_start_is_journaled_once_and_accepted_is_not_ready(self):
        result = self.execute()
        self.assertEqual(result['state'],'ACTION_RETURNED')
        self.assertEqual(result['observation']['space']['status']['phase'],'stopped')
        self.execute()
        self.assertEqual(len(self.writes()),1)
        self.assertEqual(load_json(self.root/'action/request.json')['state'],'SUBMITTED')

    def test_platform_pending_approval_is_preserved_without_resubmission(self):
        self.reply = {'status':'pending_approval','data':{'approval_id':'opaque-id'}}
        for _ in range(2):
            result = self.execute()
            self.assertEqual(result['state'],'WAITING_APPROVAL')
            self.assertFalse(result.get('real_browser_call_completed',False))
        self.assertEqual(len(self.writes()),1)
        self.assertEqual(load_json(self.root/'action/request.json')['state'],'WAITING_APPROVAL')

    def test_timeout_is_unknown_and_is_never_automatically_retried(self):
        self.error = TimeoutError('fixture-secret-must-not-escape')
        first = self.execute();second = self.execute()
        self.assertEqual(first['state'],'NEEDS_REVIEW')
        self.assertEqual(second['state'],'NEEDS_REVIEW')
        self.assertEqual(len(self.writes()),1)
        self.assertEqual(load_json(self.root/'action/request.json')['state'],'UNKNOWN')
        self.assertNotIn('fixture-secret',json.dumps(first))
        self.space['status']['phase']='ready'
        reconciled=self.execute()
        self.assertEqual(reconciled['state'],'OBSERVED')
        self.assertEqual(len(self.writes()),1)

    def test_same_action_directory_cannot_be_rebound_to_another_task(self):
        self.execute()
        with self.assertRaises(ValueError):self.client().execute('start',self.context,self.root/'action',task='other-task')
        self.assertEqual(len(self.writes()),1)

    def test_stop_requires_idle_and_stopped_is_not_repeated(self):
        self.space['status']['phase']='ready'
        self.assertEqual(self.execute('stop',idle=lambda s:False)['state'],'NEEDS_ASSISTANCE')
        self.assertEqual(self.writes(),[])
        self.execute('stop',path='idle-stop',idle=lambda s:True)
        self.assertEqual(len(self.writes()),1)
        self.space['status']['phase']='stopped'
        self.assertEqual(self.execute('stop',path='already-stopped')['new_requests'],0)

    def test_creation_and_mount_mutations_are_not_claimed_as_supported(self):
        for action in ('create','configure_and_start'):
            self.assertEqual(self.execute(action)['state'],'NEEDS_ASSISTANCE')
        self.assertEqual(self.calls,[])

    def test_preparation_polls_same_UID_and_resumes_original_dataset_after_ready(self):
        journal={};requests=[];probes=[]
        def query(args):return self.raw_asset if args[0]=='assets' else self.client().read()
        def tick():
            return advance(URL,self.profile,journal,query=query,observe_ui=lambda p:None,
                request_ui=lambda action,context:requests.append((action,context)),save=lambda x:None,
                assert_idle=lambda s:True,probe=lambda plan:probes.append(plan) or {
                 'state':'MOUNT_VERIFIED','asset_binding_sha256':self.asset['binding_sha256'],
                 'dataset_source':{'source_ref':URL,'frames_by_head':{'occ':[['scene',31],['scene',32]]}}})
        journal=tick();self.assertEqual(requests[0][0],'start')
        journal=tick();self.assertEqual(len(requests),1);self.assertEqual(probes,[])
        self.space['status']['phase']='starting';journal=tick()
        self.assertEqual(len(requests),1);self.assertEqual(probes,[])
        self.space['status']['phase']='ready';journal=tick()
        self.assertEqual(journal['state'],'DATA_READY')
        self.assertEqual(journal['dataset_source']['source_ref'],URL)
        self.assertEqual(len(probes),1)

    def test_actor_uses_API_without_enabling_browser_permissions(self):
        from tools.evaluation_space_actor import execute
        cfg={'profile':self.profile,'space_backend':'official_cli','browser_backend_verified':False,
             'lpai_cli':'official-cli','local_state':str(self.root/'actor')}
        row={'id':'a'*64,'task':'task','action':'observe','context':json.dumps(self.context)}
        with patch('pcm_eval.space_api.OfficialSpaces') as sdk, patch('tools.evaluation_space_actor.run_browser_action',side_effect=AssertionError('No browser')):
            sdk.return_value.execute.return_value={'state':'OBSERVED'}
            self.assertEqual(execute(cfg,row)['state'],'OBSERVED')
            sdk.assert_called_once_with(self.profile,cli='official-cli')

    def test_transient_get_after_start_accepts_only_fresh_official_observation(self):
        saved=[];requests=[]
        def query(args):
            if args[0]=='devspaces':raise ValueError('Transient platform read unavailable')
            return self.raw_asset
        # A previously issued start must not be resubmitted after one failed get.
        first=advance(URL,self.profile,{},query=lambda a:self.raw_asset if a[0]=='assets' else self.client().read(),
            observe_ui=lambda p:None,request_ui=lambda a,c:requests.append((a,c)),probe=lambda p:None,
            save=saved.append,assert_idle=lambda s:True)
        observation=self.client().observation()
        second=advance(URL,self.profile,first,query=query,observe_ui=lambda p:observation,
            request_ui=lambda a,c:requests.append((a,c)),probe=lambda p:self.fail('Stopped space must not probe'),
            save=saved.append,assert_idle=lambda s:True)
        self.assertEqual(second['state'],'PREPARING');self.assertEqual(len(requests),1)
        observation['source']='LPAI_UI'
        rejected=advance(URL,self.profile,first,query=query,observe_ui=lambda p:observation,
            request_ui=lambda a,c:self.fail('Wrong source must not issue action'),probe=lambda p:None,
            save=saved.append,assert_idle=lambda s:True)
        self.assertEqual(rejected['state'],'NEEDS_REVIEW')


if __name__=='__main__':unittest.main()
