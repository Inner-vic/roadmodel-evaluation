import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import multiframe_full_controller as full


class FullGuardTests(unittest.TestCase):
    def test_pause_prevents_even_configuration_or_engine_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'STOP_AFTER_FRAME').write_text('explicit user pause')
            with patch.object(full.original,'FULL',root),patch.object(full,'configuration') as config,patch.object(full,'Engine') as engine:
                with self.assertRaisesRegex(ValueError,'Paused by user'):full.main()
                config.assert_not_called();engine.assert_not_called()
            self.assertFalse((root/'controller.lock').exists())

    def fixture(self,base):
        (base/'request.json').write_text(json.dumps({'frames':[['scene',i] for i in range(4)]}))
        proof={'state':'VERIFIED_BOUNDED_MULTIFRAME_PIPELINE','batch_size':4,
               'full_checkpoint_unchanged':True,'full_resumed':False,'GT':[{'GT_oracle':'VERIFIED'}]*4,
               'native_frame_count':4,'indices':list(range(4)),'frames':[['scene',i] for i in range(4)]}
        (base/'proof.json').write_text(json.dumps(proof));(base/'driver.py').write_text('frozen code')
        (base/'validation-request.json').write_text(json.dumps({'indices':list(range(4)),'frames':proof['frames']}))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        config={'frozen_request_sha256':sha(base/'request.json'),'batch_size':4,
                'verification_path':str(base/'proof.json'),'verification_sha256':sha(base/'proof.json'),
                'verification_request_path':str(base/'validation-request.json'),
                'verification_request_sha256':sha(base/'validation-request.json'),
                'code_sha256':{'driver.py':sha(base/'driver.py')}}
        (base/'multiframe-runtime.json').write_text(json.dumps(config));return config

    def test_code_change_blocks_resume_before_any_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);self.fixture(base);(base/'driver.py').write_text('changed code')
            with patch.object(full,'HERE',base),patch.object(full.original,'FULL',base):
                with self.assertRaisesRegex(ValueError,'candidate code changed'):full.configuration()

    def test_batch_size_requires_matching_real_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);config=self.fixture(base);config['batch_size']=8
            (base/'multiframe-runtime.json').write_text(json.dumps(config))
            with patch.object(full,'HERE',base),patch.object(full.original,'FULL',base):
                with self.assertRaisesRegex(ValueError,'validation is incomplete'):full.configuration()


if __name__=='__main__':unittest.main()
