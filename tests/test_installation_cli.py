import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pcm_eval.installation_cli import main, verified_feishu_identity
from tests.test_installation import request, identity, space


class InitializationCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);p=self.root/'request.json';p.write_text(json.dumps(request()))
        self.args=['--request',str(p),'--state-dir',str(self.root/'state'),'--output',str(self.root/'output'),'--run-browser']
        self.created=None;self.get_calls=[];self.browser_calls=[]

    def query(self,args,**kwargs):
        self.get_calls.append(args);self.assertEqual(args[:2],['devspaces','get'])
        if self.created:return {'data':self.created}
        raise ValueError('not readable')

    def browser(self,action,detail,**kwargs):
        self.browser_calls.append(action)
        if action=='observe':return {'state':'OBSERVED'}, {
            'source':'CODEX_CUA','real_browser_call_completed':True,'lpai_login':'engineer',
            'namespace':'team','cluster':'approved-cluster','space_name':detail['space_name'],
            'exists':False,'observed_at':time.time()}
        self.created=space(detail['space_name'])
        return {'state':'ACTION_RETURNED'},None

    def run_init(self,browser=None):
        with patch('pcm_eval.installation_cli.verified_feishu_identity',return_value=identity()), \
             patch('pcm_eval.installation_cli.request',side_effect=self.query), \
             patch('pcm_eval.installation_cli.run_initialization_browser',side_effect=browser or self.browser), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code=main(self.args)
        return exit_code,json.loads(output.getvalue())

    def test_complete_first_initialization_then_reuse_never_creates_again(self):
        code,value=self.run_init();self.assertEqual(code,0);self.assertEqual(value['state'],'BOUND')
        self.assertEqual(self.browser_calls,['observe','create'])
        self.browser_calls=[]
        code,again=self.run_init();self.assertEqual(code,0)
        self.assertEqual(again['space_name'],value['space_name']);self.assertEqual(self.browser_calls,[])
        self.assertEqual(value['new_inference_frames'],0);self.assertFalse(value['dataset_copied'])

    def test_unknown_browser_submission_is_reconciled_without_another_create(self):
        def unknown(action,detail,**kwargs):
            if action=='observe':return self.browser(action,detail,**kwargs)
            self.browser_calls.append(action);return {'state':'NEEDS_REVIEW'},None
        code,value=self.run_init(unknown)
        self.assertEqual(code,2);self.assertEqual(value['state'],'CREATE_UNKNOWN')
        self.assertEqual(self.browser_calls,['observe','create'])
        self.browser_calls=[]
        code,value=self.run_init(unknown)
        self.assertEqual(value['state'],'CREATE_UNKNOWN');self.assertEqual(self.browser_calls,[])

    def test_capacity_rejection_is_terminal_without_repeat_submission(self):
        def full(action,detail,**kwargs):
            if action=='observe':return self.browser(action,detail,**kwargs)
            self.browser_calls.append(action)
            return {'state':'BLOCKED_CAPACITY'},{'source':'CODEX_CUA','real_browser_call_completed':True,
                'space_name':detail['space_name'],'create_intent_id':detail['create_intent']['id'],
                'outcome':'CAPACITY_EXCEEDED','changed':False,'observed_at':time.time()}
        code,value=self.run_init(full)
        self.assertEqual(code,2);self.assertEqual(value['state'],'BLOCKED_CAPACITY')
        self.browser_calls=[]
        code,value=self.run_init(full)
        self.assertEqual(value['state'],'BLOCKED_CAPACITY');self.assertEqual(self.browser_calls,[])

    def test_auth_probe_returns_application_scoped_identity_without_tokens(self):
        from unittest.mock import Mock
        response={'data':{'appId':'cli_testapp','verified':True,'identities':{'user':{'openId':'ou_engineer','verified':True,'tokenStatus':'valid'}}}}
        with patch('pcm_eval.installation_cli.subprocess.run',return_value=Mock(returncode=0,stdout=json.dumps(response))) as invoke:
            result=verified_feishu_identity('lark-cli','reviewed-profile')
        self.assertEqual(result,identity())
        self.assertEqual(invoke.call_args.args[0][:3],['lark-cli','--profile','reviewed-profile'])

    def test_existing_deployment_auth_reuses_verified_bot_identity(self):
        cfg=self.root/'deployment.json';cfg.write_text(json.dumps({'owner':'ou_engineer'}))
        self.args += ['--auth-via-deployment',str(cfg)]
        with patch('tools.evaluation_agent.exchange',return_value=identity()) as probe:
            code,value=self.run_init()
        self.assertEqual(code,0)
        probe.assert_called_once_with({'owner':'ou_engineer'},'verify_user_identity')

    def test_deployment_identity_rejects_extra_credential_fields(self):
        cfg=self.root/'deployment.json';cfg.write_text('{}')
        self.args += ['--auth-via-deployment',str(cfg)]
        with patch('tools.evaluation_agent.exchange',return_value={**identity(),'token':'never-save'}):
            code,value=self.run_init()
        self.assertEqual(code,2)
        self.assertEqual(value['state'],'NEEDS_INITIALIZATION')

    def test_official_sdk_creation_requires_verified_404_and_reuses_binding(self):
        value=request();value['dataset_ref']='https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/data?version=v1'
        (self.root/'request.json').write_text(json.dumps(value))
        self.args=[x for x in self.args if x!='--run-browser']+['--create-provider','official_sdk','--sdk-python','existing-python']
        def create(body):
            state=json.loads((self.root/'state/installation.json').read_text())
            from pcm_eval.installation import Installation
            instance=Installation(self.root/'state')
            with instance.connect() as db:d=json.loads(db.execute('select detail from bindings').fetchone()['detail'])
            self.created=space(d['space_name']);return {'state':'POST_RETURNED'}
        def observe(detail,body):
            return {'exists':False,'receipt':{'source':'LPAI_NOTEBOOK_API','real_api_call_completed':True,'http_status':404,
                'space_name':detail['space_name'],'lpai_login':'engineer','owner':'engineer@example.com',
                'namespace':'team','cluster':'approved-cluster','exists':False,'observed_at':time.time()}}
        with patch('pcm_eval.notebook_api.NotebookApi') as factory, \
             patch('pcm_eval.dataset_mount.resolve_dataset',return_value={'namespace':'team','name':'data','version':'v1'}), \
             patch('pcm_eval.dataset_mount.mount_plan',return_value={'state':'MOUNT_CONFIGURED'}):
            api=factory.return_value;api.prepare.return_value=self.root/'body.json';api.observe.side_effect=observe;api.create.side_effect=create
            code,result=self.run_init();self.assertEqual(code,0);self.assertEqual(result['state'],'BOUND')
            self.assertEqual(self.browser_calls,[]);api.create.assert_called_once()
            code,result=self.run_init();self.assertEqual(code,0);api.create.assert_called_once()

    def test_sdk_capacity_is_explicit_and_rechecks_same_intent_without_post_replay(self):
        value=request();value['dataset_ref']='https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/data?version=v1'
        (self.root/'request.json').write_text(json.dumps(value))
        self.args=[x for x in self.args if x!='--run-browser']+['--create-provider','official_sdk','--sdk-python','existing-python']
        def observe(detail,body):
            return {'exists':False,'receipt':{'source':'LPAI_NOTEBOOK_API','real_api_call_completed':True,'http_status':404,
                'space_name':detail['space_name'],'lpai_login':'engineer','owner':'engineer@example.com',
                'namespace':'team','cluster':'approved-cluster','exists':False,'observed_at':time.time()}}
        with patch('pcm_eval.notebook_api.NotebookApi') as factory, \
             patch('pcm_eval.dataset_mount.resolve_dataset',return_value={'namespace':'team','name':'data','version':'v1'}), \
             patch('pcm_eval.dataset_mount.mount_plan',return_value={'state':'MOUNT_CONFIGURED'}):
            api=factory.return_value;api.prepare.return_value=self.root/'body.json';api.observe.side_effect=observe
            api.create.return_value={'state':'BLOCKED_CAPACITY','platform_writes':0}
            code,result=self.run_init();self.assertEqual(code,2);self.assertEqual(result['state'],'BLOCKED_CAPACITY')
            first=json.loads((self.root/'output/initialization.json').read_text())
            def create(body):
                from pcm_eval.installation import Installation
                current=Installation(self.root/'state').get(first['key'])
                self.assertEqual(current['create_intent'],first['create_intent'])
                self.created=space(current['space_name']);return {'state':'POST_RETURNED','platform_writes':1}
            api.create.side_effect=create
            code,result=self.run_init();self.assertEqual(code,0);self.assertEqual(result['state'],'BOUND')
            self.assertEqual(self.browser_calls,[])


if __name__=='__main__':unittest.main()
