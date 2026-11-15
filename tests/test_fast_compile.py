"""Run in the actual isolated Linux worker: no inference or board connection."""
import hashlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
if not Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020').is_dir():
    raise unittest.SkipTest('Run optimizer tests on the deployed Linux worker with its original native package')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import board_validate_subnet as helper
from fast_subnet_driver import stream_read


class ValidatorTests(unittest.TestCase):
    def stage(self):return {'stage':0,'outputs':[{'filename':'tensor.1x2.float16.bin','declared_data_path':'output/RESULT_tensor_float16.bin','tensor':'tensor','dtype':'float16','nbytes':4}]}
    def test_finite_output_is_hashed_and_linked_without_tensor_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'sub0/output';out.mkdir(parents=True);body=np.array([1,2],np.float16).tobytes();(out/'RESULT_tensor_float16.bin').write_bytes(body)
            with patch.object(helper,'checked_root',return_value=root):r=helper.validate_stage(root,self.stage())
            self.assertTrue(r[0]['finite']);self.assertEqual(hashlib.sha256(body).hexdigest(),r[0]['sha256']);self.assertEqual(body,(root/'sub0/export/tensor.1x2.float16.bin').read_bytes())
    def test_nonfinite_intermediate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'sub0/output';out.mkdir(parents=True);(out/'RESULT_tensor_float16.bin').write_bytes(np.array([1,np.nan],np.float16).tobytes())
            with patch.object(helper,'checked_root',return_value=root),self.assertRaisesRegex(ValueError,'Non-finite'):helper.validate_stage(root,self.stage())
    def test_ambiguous_different_fresh_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'sub0/output';out.mkdir(parents=True);(out/'RESULT_tensor_float16.bin').write_bytes(np.array([1,2],np.float16).tobytes());(out/'tensor.1x2.float16.bin').write_bytes(np.array([1,3],np.float16).tobytes())
            with patch.object(helper,'checked_root',return_value=root),self.assertRaisesRegex(ValueError,'Ambiguous'):helper.validate_stage(root,self.stage())
    def test_root_escape_is_rejected(self):
        for path in ('/tmp/test','/userdata/benchEval/models/../escape','/userdata/benchEval/models/shared'):
            with self.assertRaises(ValueError):helper.checked_root(path)


class StreamTests(unittest.TestCase):
    def client(self,body):
        channel=SimpleNamespace(recv_exit_status=lambda:0,close=lambda:None)
        stdout=io.BytesIO(body);stdout.channel=channel
        return SimpleNamespace(_ssh=SimpleNamespace(exec_command=lambda *a,**k:(io.BytesIO(),stdout,io.BytesIO())))
    def test_exact_bytes_and_hash_are_preserved(self):
        body=b'\x00\xffhello';self.assertEqual(body,stream_read(self.client(body),'/private/output',len(body),hashlib.sha256(body).hexdigest()))
    def test_truncated_or_extra_data_is_rejected(self):
        for body in (b'a',b'abc'):
            with self.assertRaisesRegex(ValueError,'byte count'):stream_read(self.client(body),'/private/output',2,'unused')
    def test_changed_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'SHA'):stream_read(self.client(b'ab'),'/private/output',2,'0'*64)

if __name__=='__main__':unittest.main()


class StaticGlobalChecks(unittest.TestCase):
    def test_all_controller_branches_have_defined_global_names(self):
        import symtable,builtins
        base=Path(__file__).resolve().parents[1]/'tools'
        for name in ('formal_compile_remote.py','fast_subnet_driver.py','board_validate_subnet.py'):
            tree=symtable.symtable((base/name).read_text(),name,'exec')
            declared={s.get_name() for s in tree.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            allowed=declared|set(dir(builtins))|{'__file__','__name__','__package__'}
            def check(table):
                for symbol in table.get_symbols():
                    if symbol.is_referenced() and symbol.is_global():
                        self.assertIn(symbol.get_name(),allowed,(name,table.get_name(),symbol.get_name()))
                for child in table.get_children():check(child)
            for child in tree.get_children():check(child)
