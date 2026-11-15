import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pcm_eval.model_ticket import EXAMPLE_INPUTS, Tickets
from pcm_eval.space_preparation_queue import PreparationQueue


def source(ref):
    frames={'occ':[['scene',3]],'vision':[]}
    sha=hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    receipt={'state':'MOUNT_VERIFIED','execution_location':'lpai','workstation_data_used':False,
             'space_uid':'space-1','index_sha256':'index-sha','asset_binding_sha256':'asset-sha','frame_manifest_sha256':sha}
    return {'id':'version-uid','version':'submitted-version','source_ref':ref,'index_path':'/mnt/requested/data_link.json',
            'index_sha256':'index-sha','asset_binding_sha256':'asset-sha','frames_by_head':frames,
            'frame_manifest_sha256':sha,'mount_receipt':receipt}


class PreparationQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.q=PreparationQueue(self.root/'queue')
        self.tid='rmm-'+'1'*20;self.owner='ou_owner';self.profile={'space_uid':'space-1'};self.client='a'*32
    def tearDown(self):self.tmp.cleanup()
    def enqueue(self,revision='first'):
        return self.q.ensure(self.tid,self.owner,EXAMPLE_INPUTS,self.profile,revision=revision)
    def test_same_submission_deduplicates_but_new_confirmation_rechecks(self):
        first=self.enqueue();self.assertEqual(first['id'],self.enqueue()['id'])
        self.assertNotEqual(first['id'],self.enqueue('second')['id'])
    def test_claim_is_exclusive_and_own_restart_preserves_lease(self):
        self.enqueue();job=self.q.claim(self.client)
        self.assertIsNone(self.q.claim('b'*32));self.assertEqual(job['lease'],self.q.claim(self.client)['lease'])
    def test_other_worker_cannot_update_and_original_intent_survives_error(self):
        self.enqueue();j=self.q.claim(self.client)
        with self.assertRaises(ValueError):self.q.update(j['id'],'b'*32,j['lease'],'DATA_READY',{})
        self.q.update(j['id'],self.client,j['lease'],'RUNNING',{'journal':{'intent':{'action':'start'}}})
        self.q.update(j['id'],self.client,j['lease'],'NEEDS_REVIEW',{'reason':'transport failed'})
        d=self.enqueue()['detail'];self.assertEqual(d['journal']['intent']['action'],'start')
    def test_wrong_space_or_data_never_becomes_ready(self):
        self.enqueue();j=self.q.claim(self.client);d=source(EXAMPLE_INPUTS['dataset_ref']);d['mount_receipt']['space_uid']='other'
        with self.assertRaises(ValueError):self.q.update(j['id'],self.client,j['lease'],'DATA_READY',{'dataset_source':d})
    def test_confirmed_ticket_prepares_then_blocks_missing_head_without_inference(self):
        catalog={'models':[{'url':EXAMPLE_INPUTS['model_delivery_url'],'version':'v3.2.2','evaluation_heads':['occ','vision']}],
          'datasets':[], 'delivery_by_owner':{self.owner:{'document_id':'doc'}},'python':'/python','worker':'/worker',
          'runtime_config':{'execution_location':'lpai','space_preparation':{'by_owner':{self.owner:self.profile}}}}
        cat=self.root/'catalog.json';cat.write_text(json.dumps(catalog));tickets=Tickets(self.root/'tickets',cat)
        session='feishu:oc_chat:'+self.owner;t=tickets.open(self.owner,session);tickets.bind(t,'om_card')
        tickets.submit(self.owner,session,{'value':{'task_id':t['id'],'nonce':t['nonce'],'entry_schema':'model-ticket/v2'},
            'event_id':'isolated-test','message_id':'om_card','operator_id':self.owner,'chat_id':'oc_chat','form_value':EXAMPLE_INPUTS})
        spawn=lambda *a,**k:self.fail('Missing visual data cannot launch partial evaluation')
        tickets.tick(spawn=spawn);self.assertEqual(tickets.get(t['id'],self.owner)['state'],'PREPARING')
        q=PreparationQueue(self.root/'tickets/space-preparation');job=q.claim(self.client)
        q.update(job['id'],self.client,job['lease'],'DATA_READY',{'dataset_source':source(EXAMPLE_INPUTS['dataset_ref'])})
        tickets.tick(spawn=spawn);r=tickets.get(t['id'],self.owner)
        self.assertEqual(r['state'],'BLOCKED');self.assertEqual(r['detail']['heads']['occ']['frame_count'],1)
        self.assertEqual(r['detail']['heads']['vision']['frame_count'],0)
        self.assertEqual(r['detail']['dataset_mount']['space_uid'],'space-1')


if __name__=='__main__':unittest.main()
