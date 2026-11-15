import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
import uuid
from unittest.mock import patch

from pcm_eval.common import load_json
from pcm_eval.feishu_entry import render
from pcm_eval.runtime import Runtime
from pcm_eval.two_link import resolve_pair, validate_inputs
from pcm_eval.runner import PREFLIGHT
from pcm_eval.common import digest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = load_json(ROOT / 'examples/two_link_submission.json')
CATALOG = ROOT/'evidence/feishu-entry-20261029/catalog.pending.json'
if not CATALOG.exists():CATALOG=ROOT/'configs/two_link_catalog.json'
OWNER = 'ou_test_owner'
SESSION = 'feishu:oc_test_chat:' + OWNER


@contextmanager
def directory():
    base = ROOT / 'evidence/test-two-link'
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try: yield path
    finally:
        assert path.resolve().is_relative_to(base.resolve())
        shutil.rmtree(path)


def submission(rt):
    task = rt.open(OWNER, SESSION)
    rt.bind_message(task['id'], OWNER, 'om_test_card', SESSION)
    event = {'action':'roadmodel_submit','operator_id':OWNER,'chat_id':'oc_test_chat','message_id':'om_test_card',
             'event_id':uuid.uuid4().hex,'value':{'nonce':task['nonce'],'task_id':task['id']},'form_value':INPUTS}
    return task, event


class TwoLinkTests(unittest.TestCase):
    def test_exact_two_business_fields(self):
        self.assertEqual(INPUTS,validate_inputs(INPUTS))
        for extra in ('command','kind','scope','owner','execute_all'):
            with self.assertRaises(ValueError):validate_inputs({**INPUTS,extra:'anything'})
        for bad in ('https://evil.example/a','https://li.feishu.cn@evil.example/a','file:///a','http://li.feishu.cn/a'):
            with self.assertRaises(ValueError):validate_inputs({**INPUTS,'model_delivery_url':bad})

    def test_task_id_cannot_escape_output_directory(self):
        for task in ('../escape','bad/path','x'*60):
            with self.assertRaises(ValueError):resolve_pair(INPUTS,task)

    def test_actual_datalink_cannot_expand_the_two_frame_scope(self):
        selected=[['scene',18],['scene',19]]
        with directory() as path:
            link=path/'data.json'
            def run(frames):
                data={k:{'scene':{'frames':[{'frame_index':f} for f in frames]}} for k in ('eval','joint_eval')}
                link.write_text(json.dumps(data),encoding='utf-8')
                identity={'files':{str(link):digest(link)},'model_scope':{'data_link_file':str(link),'selected_frames':selected}}
                return subprocess.run([sys.executable,'-c',PREFLIGHT,'new',str(path/'not-created'),json.dumps(identity),str(link)],capture_output=True)
            self.assertEqual(0,run([18,19]).returncode)
            rejected=run([18,19,20])
            self.assertNotEqual(0,rejected.returncode)
            self.assertIn(b'frozen small scope',rejected.stderr)

    def test_original_materials_do_not_silently_replace_compile_version(self):
        resolved=resolve_pair(INPUTS,'rm-test',catalog_file=CATALOG)
        self.assertEqual(2,len(resolved['jobs']))
        self.assertEqual({'v3.2.2'},{x['version'] for x in resolved['jobs']})
        self.assertTrue(resolved['ready'])
        self.assertEqual(5,len(resolved['jobs'][1]['request']['model']['subnet_archives']))
        self.assertEqual('subnet_chain_workstation',resolved['jobs'][1]['request']['execution']['adapter'])

    def test_unknown_links_preserve_two_blocked_chains(self):
        value=resolve_pair({**INPUTS,'model_delivery_url':'https://li.feishu.cn/record/unknown'},'rm-test')
        self.assertEqual(2,len(value['jobs']))
        self.assertTrue(all(not x['ready'] for x in value['jobs']))

    def test_duplicate_callback_does_not_duplicate_job(self):
        with directory() as path:
            rt=Runtime(path);task,event=submission(rt)
            self.assertEqual(task['id'],rt.submit(OWNER,SESSION,event))
            self.assertEqual(task['id'],rt.submit(OWNER,SESSION,event))
            self.assertEqual(2,len(rt.get(task['id'],OWNER)['chains']))
            event['event_id']='different-event'
            with self.assertRaises(ValueError):rt.submit(OWNER,SESSION,event)

    def test_identity_nonce_original_card_and_scope_are_enforced(self):
        for key,new in [('operator_id','ou_someone_else'),('chat_id','oc_elsewhere'),('message_id','om_replaced')]:
            with directory() as path:
                rt=Runtime(path);task,event=submission(rt);event[key]=new
                with self.assertRaises(ValueError):rt.submit(OWNER,SESSION,event)
                self.assertEqual('DRAFT',rt.get(task['id'],OWNER)['state'])
        with directory() as path:
            rt=Runtime(path);task,event=submission(rt);event['value']['nonce']='wrong'
            with self.assertRaises(ValueError):rt.submit(OWNER,SESSION,event)

    def test_two_ready_chains_start_concurrently(self):
        with directory() as path:
            rt=Runtime(path);task,event=submission(rt);rt.submit(OWNER,SESSION,event)
            barrier=threading.Barrier(2,timeout=4)
            seen=[]
            def execute(job,target):
                seen.append(job['kind']);barrier.wait()
                return {'state':'SUCCEEDED','test_only':True}
            resolved={'jobs':[{'kind':kind,'ready':True} for kind in ('model_accuracy','compile_alignment')]}
            with patch('pcm_eval.runtime.resolve_pair',return_value=resolved):
                value=rt.tick(execute)
            self.assertEqual('SUCCEEDED',value['state'])
            self.assertEqual(2,len(seen))
            self.assertIsNone(rt.tick(execute))

    def test_one_missing_chain_does_not_cancel_other_chain(self):
        with directory() as path:
            rt=Runtime(path);task,event=submission(rt);rt.submit(OWNER,SESSION,event)
            resolved={'jobs':[{'kind':'model_accuracy','ready':True},
                              {'kind':'compile_alignment','ready':False,'issues':['missing same-version pack']}]}
            with patch('pcm_eval.runtime.resolve_pair',return_value=resolved):
                value=rt.tick(lambda *_:{'state':'SUCCEEDED'})
            self.assertEqual('PARTIAL',value['state'])
            self.assertEqual({'SUCCEEDED','BLOCKED'},{x['state'] for x in value['chains']})

    def test_unknown_execution_never_retried_after_restart(self):
        with directory() as path:
            rt=Runtime(path);task,event=submission(rt);rt.submit(OWNER,SESSION,event)
            with rt.connect() as c:
                c.execute("UPDATE tasks SET state='RUNNING'")
                c.execute("UPDATE chains SET state='RUNNING'")
            restarted=Runtime(path);restarted.recover()
            self.assertIsNone(restarted.tick(lambda *_:self.fail('must not run again')))
            self.assertEqual('NEEDS_REVIEW',restarted.get(task['id'],OWNER)['state'])

    def test_real_material_preflight_does_not_run_inference(self):
        with directory() as path:
            rt=Runtime(path,catalog_file=CATALOG);task,event=submission(rt);rt.submit(OWNER,SESSION,event)
            value=rt.tick(lambda *_:{'state':'PARTIAL','test_only':True})
            self.assertEqual('PARTIAL',value['state'])
            self.assertEqual(2,len(value['chains']))

    def test_card_has_exact_two_required_fields_and_no_hidden_scope(self):
        with directory() as path:
            rt=Runtime(path);task,_=submission(rt)
            card=render(task)
            form=card['body']['elements'][1]
            fields=[x for x in form['elements'] if x['tag']=='input']
            self.assertEqual({'model_delivery_url','dataset_url'},{x['name'] for x in fields})
            self.assertTrue(all(x['required'] for x in fields))
            self.assertEqual('submit',form['elements'][-1]['form_action_type'])


if __name__=='__main__': unittest.main()
