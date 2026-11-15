import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pcm_eval.model_ticket import EXAMPLE_INPUTS as DEFAULTS,Tickets,inputs,resolve
from pcm_eval.model_ticket_entry import render


class ModelTicketTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.index={'eval':{f's{i}':{'frames':[{'frame_index':j} for j in range(5)]} for i in range(12)},'vision_eval':{'s0':{'frames':[]}}}
        self.path=self.root/'main.json';self.path.write_text(json.dumps(self.index))
        self.cat={'models':[{'url':DEFAULTS['model_delivery_url'],'version':'v3.2.2','evaluation_heads':['occ','vision']}],
         'datasets':[{'id':'main','version':'main','aliases':[DEFAULTS['dataset_ref']],
          'index_path':str(self.path),'index_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}],
         'runtime_config':{},'delivery_by_owner':{'ou_owner':{'document_id':'doc'}},'python':'/python','worker':'/worker'}
        self.catalog=self.root/'catalog.json';self.save();self.rt=Tickets(self.root/'state',self.catalog)
        self.owner='ou_owner';self.session='feishu:oc_test:ou_owner'
    def save(self):self.catalog.write_text(json.dumps(self.cat))
    def complete(self):
        self.index['vision_eval']={'v0':{'frames':[{'frame_index':j} for j in range(4)]}}
        self.path.write_text(json.dumps(self.index));self.cat['datasets'][0]['index_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest();self.save()
    def tearDown(self):self.temp.cleanup()
    def event(self,t,**changes):
        return {'value':{'task_id':t['id'],'nonce':t['nonce'],'entry_schema':'model-ticket/v2'},'event_id':'event1','message_id':'om_card',
                'operator_id':self.owner,'chat_id':'oc_test','form_value':{**DEFAULTS,**changes}}
    def confirmed(self):
        t=self.rt.open(self.owner,self.session);self.rt.bind(t,'om_card');self.rt.submit(self.owner,self.session,self.event(t));return t
    def test_missing_one_gt_blocks_entire_evaluation(self):
        r=resolve(DEFAULTS,self.cat);self.assertFalse(r['ready']);self.assertEqual(r['heads']['vision']['frame_count'],0)
        self.assertEqual(r['evaluation_heads'],['occ','vision'])
    def test_one_dataset_for_all_items_and_full_scope(self):
        self.complete();r=resolve(DEFAULTS,self.cat);self.assertTrue(r['ready']);self.assertEqual(r['scope'],'full')
        self.assertEqual(r['heads']['occ']['frame_count'],60);self.assertEqual(r['heads']['vision']['frame_count'],4)
        self.assertEqual({h['dataset_id'] for h in r['heads'].values()},{'main'})
        self.assertEqual({h['source_ref'] for h in r['heads'].values()},{DEFAULTS['dataset_ref']})
    def test_removed_fields_cannot_bypass_full_evaluation(self):
        for key,value in [('target','occ'),('scope','small'),('vision_policy','delivery'),('vision_ref','/other')]:
            with self.assertRaises(ValueError):inputs({**DEFAULTS,key:value})
    def test_other_dataset_is_never_implicitly_used(self):
        self.cat['datasets'].append({'id':'other','aliases':['/other'],'index_path':'does-not-exist'})
        r=resolve(DEFAULTS,self.cat);self.assertFalse(r['ready']);self.assertEqual(r['heads']['vision']['dataset_id'],'main')
    def test_unknown_path_does_not_substitute_registered_dataset(self):
        r=resolve({**DEFAULTS,'dataset_ref':'/unregistered'},self.cat);self.assertFalse(r['ready']);self.assertNotIn('occ',r['heads'])
    def test_new_draft_has_no_default_model_or_dataset(self):
        self.assertEqual(self.rt.open(self.owner,self.session)['inputs'],{'model_delivery_url':'','dataset_ref':''})
    def test_lpai_execution_never_reads_registered_workstation_cache(self):
        self.complete();self.cat['runtime_config']['execution_location']='lpai'
        r=resolve(DEFAULTS,self.cat);self.assertFalse(r['ready']);self.assertEqual(r['heads'],{})
    def test_lpai_source_receipt_must_match_submitted_input(self):
        self.complete();self.cat['runtime_config']['execution_location']='lpai'
        raw=self.path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        def source(ref):
            return {'id':'dynamic-new','version':'new','source_ref':ref,'index':self.index,
                'index_bytes':raw,'index_path':'/lpai/mounted/data_link_new.json','index_sha256':sha,
                'asset_binding_sha256':'asset-hash', 'mount_receipt':{'state':'MOUNT_VERIFIED',
                'execution_location':'lpai','workstation_data_used':False,'index_sha256':sha,'asset_binding_sha256':'asset-hash'}}
        r=resolve(DEFAULTS,self.cat,dataset_source=source);self.assertTrue(r['ready'])
        self.assertEqual({h['dataset_id'] for h in r['heads'].values()},{'dynamic-new'})
        def bad(ref):
            d=source(ref);d['mount_receipt']['workstation_data_used']=True;return d
        with self.assertRaises(ValueError):resolve(DEFAULTS,self.cat,dataset_source=bad)
    def test_callback_owner_card_nonce_and_event_deduplication(self):
        t=self.rt.open(self.owner,self.session);self.rt.bind(t,'om_card');event=self.event(t)
        with self.assertRaises(ValueError):self.rt.submit(self.owner,self.session,{**event,'operator_id':'ou_other'})
        tid=self.rt.submit(self.owner,self.session,event);self.assertEqual(tid,self.rt.submit(self.owner,self.session,event))
        self.assertEqual(self.rt.get(tid,self.owner)['state'],'CHECKING')
        with self.assertRaises(ValueError):self.rt.submit(self.owner,self.session,{**event,'event_id':'another'})
    def test_form_has_two_inputs_no_selectors_and_one_confirm(self):
        card=render(self.rt.open(self.owner,self.session));found=[]
        def walk(v):
            if isinstance(v,dict):
                if v.get('tag') in ('input','button','select_static'):found.append(v)
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(card);self.assertEqual([x['name'] for x in found if x['tag']=='input'],['model_delivery_url','dataset_ref'])
        self.assertFalse(any(x['tag']=='select_static' for x in found));buttons=[x for x in found if x['tag']=='button'];self.assertEqual(len(buttons),1)
        self.assertEqual(buttons[0]['behaviors'][0]['value']['entry_schema'],'model-ticket/v2')
    def test_missing_gt_never_dispatches(self):
        t=self.confirmed();self.rt.tick(spawn=lambda *a,**kw:self.fail('No partial execution'))
        self.assertEqual(self.rt.get(t['id'],self.owner)['state'],'BLOCKED')
    def test_confirmed_request_freezes_one_source_and_dispatches_once(self):
        self.complete();t=self.confirmed();seen=[]
        def spawn(*a,**kw):seen.append(a);return SimpleNamespace(pid=999999999)
        self.rt.tick(spawn=spawn);self.rt.tick(spawn=spawn);self.assertEqual(len(seen),1)
        r=json.loads((self.root/'state/runs'/t['id']/'request.json').read_text())['resolved']
        self.assertTrue(r['single_dataset']);self.assertEqual(r['scope'],'full');self.assertFalse(r['compile_enabled'])
    def test_dead_worker_with_old_progress_is_not_replayed(self):
        self.complete();t=self.confirmed();self.rt.tick(spawn=lambda *a,**kw:SimpleNamespace(pid=999999999))
        (self.root/'state/runs'/t['id']/'progress.json').write_text(json.dumps({'state':'RUNNING'}))
        self.rt.tick(spawn=lambda *a,**kw:self.fail('No retry'));self.assertEqual(self.rt.get(t['id'],self.owner)['state'],'NEEDS_REVIEW')
    def test_all_model_items_require_binding_no_silent_skip(self):
        self.complete();self.cat['models'][0]['evaluation_heads'].append('trajectory')
        self.assertFalse(resolve(DEFAULTS,self.cat)['ready'])
    def test_old_mixed_dataset_native_adapter_cannot_run_new_source(self):
        self.complete();self.cat['runtime_config']['native_dataset_bindings']={'occ':'main','vision':'other'}
        self.assertFalse(resolve(DEFAULTS,self.cat)['ready'])
    def test_path_is_not_parent_traversal_or_external_url(self):
        for path in ['/data/../private','https://example.com/data']:
            with self.assertRaises(ValueError):inputs({**DEFAULTS,'dataset_ref':path})


if __name__=='__main__':unittest.main()
