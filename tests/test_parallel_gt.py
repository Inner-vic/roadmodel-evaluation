import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
try:import numpy as np
except ImportError:raise unittest.SkipTest('Count checkpoint checks require NumPy on the evaluation worker')
from pcm_eval.parallel_gt import OrderedCheckpoint,process_retained_gt
from pcm_eval.common import digest


class CheckpointTests(unittest.TestCase):
    def receipt(self,root,name,value=1,dtype=np.int64):
        p=root/(name+'.npz')
        with p.open('wb') as f:np.savez(f,**{b+'_confusion':np.full((27,27),value,dtype) for b in ('board','onnx')},**{b+'_ds':np.full(3,value,dtype) for b in ('board','onnx')})
        return {'state':'VERIFIED','counts_path':str(p),'counts_sha256':digest(p)}
    def test_order_and_duplicates_cannot_change_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);cp=OrderedCheckpoint(r/'checkpoint.npz');receipt=self.receipt(r,'one');cp.commit(0,receipt);before=digest(cp.path)
            for index in (0,2):
                with self.assertRaisesRegex(ValueError,'order|duplicate'):cp.commit(index,receipt)
            self.assertEqual(before,digest(cp.path));self.assertEqual(1,cp.cursor)
    def test_raw_counts_are_summed_without_averaging(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);cp=OrderedCheckpoint(r/'checkpoint.npz');cp.commit(0,self.receipt(r,'one',2));cp.commit(1,self.receipt(r,'two',3))
            with np.load(cp.path) as value:
                self.assertEqual(2,int(value['completed']));self.assertTrue((value['board_confusion']==5).all());self.assertTrue((value['onnx_ds']==5).all())
    def test_changed_counts_are_rejected_before_commit(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);cp=OrderedCheckpoint(r/'checkpoint.npz');receipt=self.receipt(r,'one');Path(receipt['counts_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):cp.commit(0,receipt)
            self.assertFalse(cp.path.exists());self.assertEqual(0,cp.cursor)
    def test_negative_or_float_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            for label,value,dtype in [('negative',-1,np.int64),('float',1,np.float64)]:
                cp=OrderedCheckpoint(r/(label+'-cp.npz'))
                with self.assertRaisesRegex(ValueError,'Invalid'):cp.commit(0,self.receipt(r,label,value,dtype))
                self.assertFalse(cp.path.exists())
    def test_initial_cursor_must_match_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);cp=OrderedCheckpoint(r/'old.npz');cp.commit(0,self.receipt(r,'one'))
            with self.assertRaisesRegex(ValueError,'cursor'):OrderedCheckpoint(r/'new.npz',cursor=0,initial=cp.path)


class JobBoundaryTests(unittest.TestCase):
    def test_wrong_frame_source_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);job={'frame_index':0,'frozen_request':str(r/'frozen/request.json'),'source_root':str(r/'wrong'),'output':str(r/'out/frame0'),'output_root':str(r/'out')}
            with patch('pcm_eval.parallel_gt._context',return_value=(None,{'frames':[['s',1]]},None,None)),self.assertRaisesRegex(ValueError,'frozen frame'):process_retained_gt(job)
            self.assertFalse(Path(job['output']).exists())
    def test_canonical_results_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);source=r/'frozen/compile/frames/frame_000000';job={'frame_index':0,'frozen_request':str(r/'frozen/request.json'),'source_root':str(source),'output':str(r/'frozen/overwrite'),'output_root':str(r)}
            with patch('pcm_eval.parallel_gt._context',return_value=(None,{'frames':[['s',1]]},None,None)),self.assertRaisesRegex(ValueError,'isolated'):process_retained_gt(job)
            self.assertFalse(Path(job['output']).exists())

if __name__=='__main__':unittest.main()
