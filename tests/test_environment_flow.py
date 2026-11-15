import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from pcm_eval.common import digest
from pcm_eval.lpai_environment import verify_bundle
from pcm_eval.space_browser import SCHEMA, run_browser_action
from pcm_eval.delivery_source import LarkSource


class EnvironmentFlowTest(unittest.TestCase):
    def test_existing_grant_refresh_retries_only_the_same_read_once(self):
        denied=Mock(returncode=1,stdout='',stderr=json.dumps({'ok':False,'error':{'message':'need_user_authorization'}}))
        verified=Mock(returncode=0,stdout='{}',stderr='')
        success=Mock(returncode=0,stdout=json.dumps({'ok':True,'data':{'version':'v3.2.2'}}),stderr='')
        with patch('pcm_eval.delivery_source.subprocess.run',side_effect=[denied,verified,success]) as invoke:
            self.assertTrue(LarkSource(executable='lark-cli').query(['base','+url-resolve','--url','same-link'])['ok'])
        self.assertEqual(invoke.call_args_list[0].args[0],invoke.call_args_list[2].args[0])
        self.assertEqual(invoke.call_args_list[1].args[0],['lark-cli','auth','status','--json','--verify'])

    def test_bundle_rejects_dataset_members_even_with_valid_hashes(self):
        import hashlib
        for name, valid in [('source/eval.py', True), ('weights/epoch_44.pth', True),
                            ('dataset/scene/occ.npy', False), ('source/../../data/file', False)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'bundle.tar.gz';body=b'test-model-file'
                manifest={'files':{name:hashlib.sha256(body).hexdigest()}}
                with tarfile.open(p,'w:gz') as tar:
                    for key,raw in [('environment-manifest.json',json.dumps(manifest).encode()),(name,body)]:
                        info=tarfile.TarInfo(key);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
                if valid:self.assertEqual(verify_bundle(p),manifest)
                else:
                    with self.assertRaises(ValueError):verify_bundle(p)

    def test_runtime_schema_requires_every_property_and_allows_unknown_identity(self):
        self.assertEqual(set(SCHEMA['required']),set(SCHEMA['properties']))
        self.assertIn('null',SCHEMA['properties']['space_exists']['type'])
        self.assertIn('null',SCHEMA['properties']['ssh_port']['type'])

    def invoke(self, *, tool=True, complete=True, forbidden=False):
        reply={'outcome':'OBSERVED','instance_name':'mine','visible_state':'运行中','changed':False,'problem':'',
               'space_exists':True,'ssh_host':'approved-host','ssh_port':30331,'credential_name':'existing','mounts':[]}
        rows=[]
        if tool:rows.append({'type':'item.completed','item':{'type':'mcp_tool_call','server':'cua_repl','tool':'js','status':'completed'}})
        if forbidden:rows.append({'type':'item.completed','item':{'type':'command_execution'}})
        rows.append({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(reply)}})
        if complete:rows.append({'type':'turn.completed'})
        child=Mock(stdout=io.BytesIO(b'\n'.join(json.dumps(r).encode() for r in rows)),stderr=io.BytesIO(),stdin=io.BytesIO(),returncode=0,pid=123)
        child.wait.return_value=0
        with patch('pcm_eval.space_browser.installed_binary',return_value='codex'), \
             patch('pcm_eval.space_browser.prompt',return_value='Read only'), \
             patch('pcm_eval.space_browser.subprocess.Popen',return_value=child):
            return run_browser_action('observe',{'metadata':{'name':'mine'}},{},schema_path='schema.json',cwd=Path.cwd(),timeout=1)

    def test_completed_cua_turn_is_accepted_independently_of_shutdown_output(self):
        result=self.invoke()
        self.assertEqual(result['state'],'OBSERVED')
        self.assertTrue(result['turn_completed'])

    def test_claimed_observation_without_real_tool_or_completed_turn_is_rejected(self):
        for options in ({'tool':False},{'complete':False},{'forbidden':True}):
            with self.subTest(options=options):self.assertEqual(self.invoke(**options)['state'],'NEEDS_REVIEW')


if __name__=='__main__':unittest.main()
