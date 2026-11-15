import copy
from pathlib import Path
import unittest
from tests.test_two_link_runtime import directory,ROOT,INPUTS
from pcm_eval.atomic_result import prepare_model_result
from pcm_eval.common import load_json
from pcm_eval.two_link import resolve_pair
from pcm_eval.validation import validate_result


class AtomicResultTests(unittest.TestCase):
    def test_actual_completed_two_frame_result_has_current_run_identity(self):
        base=ROOT/'evidence/feishu-entry-20261029/rm-small-20261029-01/model_accuracy'
        if not base.exists():self.skipTest('Optional real integration fixture not distributed')
        receipt=load_json(base/'execution/execution_receipt.json')
        job=resolve_pair(INPUTS,'rm-fixture')['jobs'][0]
        with directory() as path:
            result=load_json(prepare_model_result(job,receipt,path/'result.json'))
            self.assertEqual('rm-small-20261029-01-model',result['provenance']['native_run']['run_id'])
            self.assertEqual(2,result['coverage']['observed']['model_samples'])
            self.assertTrue(validate_result(result)['valid'])
            self.assertEqual(315,len(result['metrics']))
            changed=copy.deepcopy(job);changed['request']['evaluation']['selected_frames'].pop()
            with self.assertRaises(ValueError):prepare_model_result(changed,receipt,path/'bad.json')
