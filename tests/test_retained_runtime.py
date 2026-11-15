import copy
import json
from pathlib import Path
import tempfile
import shutil
import uuid
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from pcm_eval.common import load_json, save_json
from pcm_eval.runtime import Runtime
from pcm_eval.two_link import resolve_pair
from pcm_eval.retained import checked_file

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'evidence/trial-entry-20261030/package'
INPUTS=load_json(ROOT/'examples/two_link_submission.json')

@contextmanager
def directory():
    base=ROOT/'evidence/test-retained';base.mkdir(exist_ok=True)
    path=base/uuid.uuid4().hex;path.mkdir()
    try:yield path
    finally:
        assert path.resolve().is_relative_to(base.resolve())
        shutil.rmtree(path)


class RetainedRuntimeTests(unittest.TestCase):
    def catalog(self, directory):
        catalog=load_json(PACKAGE/'configs/two_link_catalog.json')
        catalog['material_root']=str(PACKAGE)
        path=Path(directory)/'catalog.json';save_json(path,catalog)
        return path

    def test_real_two_link_result_flow_never_dispatches_inference(self):
        with directory() as d:
            catalog=self.catalog(d)
            rt=Runtime(Path(d)/'runtime',catalog_file=catalog)
            owner='ou_trial_test';session='feishu:oc_trial_test:'+owner
            task=rt.open(owner,session);rt.bind_message(task['id'],owner,'om_trial_card',session)
            event={'action':'roadmodel_submit','event_id':'trial-event','operator_id':owner,'chat_id':'oc_trial_test',
                   'message_id':'om_trial_card','value':{'task_id':task['id'],'nonce':task['nonce']},'form_value':INPUTS}
            rt.submit(owner,session,event)
            with patch('pcm_eval.runner.execute',side_effect=AssertionError('Must not repeat inference')):
                result=rt.tick()
            self.assertEqual('SUCCEEDED',result['state'],result)
            self.assertEqual({'SUCCEEDED'},{x['state'] for x in result['chains']})
            self.assertIn('DEMO00000000000000000000002',result['notice'])
            delivery=load_json(rt.root/'runs'/task['id']/'combined/delivery.json')
            self.assertTrue(delivery['five_tables_match'])
            self.assertEqual(0,delivery['new_inference_frames'])
            self.assertEqual(task['id'],rt.submit(owner,session,event))
            self.assertIsNone(rt.tick())

    def test_changed_registered_evidence_blocks_both_chains(self):
        with directory() as d:
            path=self.catalog(d);catalog=load_json(path)
            for spec in catalog['models'][0]['chains'].values():spec['retained_result']['sha256']='0'*64
            save_json(path,catalog,overwrite=True)
            result=resolve_pair(INPUTS,'rm-changed',catalog_file=path)
            self.assertFalse(result['ready'])
            self.assertTrue(all(not j['ready'] for j in result['jobs']))

    def test_retained_path_cannot_escape_bundle(self):
        with self.assertRaises(ValueError):checked_file(PACKAGE,{'path':'../outside','sha256':'0'*64})

    def test_unknown_model_cannot_receive_current_models_report(self):
        with directory() as d:
            path=self.catalog(d)
            result=resolve_pair({**INPUTS,'model_delivery_url':'https://li.feishu.cn/record/new-version'},'rm-new',catalog_file=path)
            self.assertFalse(result['ready'])
            self.assertTrue(all(not j['ready'] for j in result['jobs']))

if __name__=='__main__':unittest.main()
