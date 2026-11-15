"""Order, cache reuse and failure boundaries without any model/board execution."""
import json
import tempfile
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pcm_eval.reference_prefetch import ReferencePrefetch


class FakePool:
    def __init__(self, **kwargs):self.jobs=[]
    def submit(self, function, job):
        self.jobs.append(job);root=Path(job['root']);root.mkdir();future=Future();future.set_result({'root':str(root)})
        return future
    def shutdown(self, **kwargs):pass


class PrefetchTests(unittest.TestCase):
    def engine(self):return SimpleNamespace(binding={'onnx_fp16_sha256':'same-model'},producers=ThreadPoolExecutor(max_workers=1))
    def config(self):return {'workers':2,'onnx_fp16_sha256':'same-model'}

    def test_completed_native_and_reference_are_reused_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'frame_000000';a.mkdir();(a/'execution_receipt.json').write_text(json.dumps({'selected_frames':[['scene',0]]}))
            b=root/'frame_000001';b.mkdir();(b/'prepare.json').write_text(json.dumps({'state':'READY','onnx_fp16_sha256':'same-model'}))
            (b/'input-reuse.json').write_text(json.dumps({'frame':{'meta':{'scene_name':'scene','frame_idx':1}}}))
            manifest=SimpleNamespace(meta={'scene_name':'scene','frame_idx':2},frame_dir=root/'sample')
            with patch('pcm_eval.reference_prefetch.ProcessPoolExecutor',FakePool):
                prefetch=ReferencePrefetch(self.engine(),[0,1,2],[['scene',i] for i in range(3)],iter([manifest]),root,root,self.config())
                try:
                    self.assertEqual([root/f'frame_{i:06d}' for i in range(3)],[prefetch.get(i) for i in range(3)])
                    self.assertEqual(1,len(prefetch.pool.jobs))
                    with self.assertRaises(KeyError):prefetch.get(2)
                finally:prefetch.close()

    def test_partial_preparation_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'frame_000000').mkdir()
            with patch('pcm_eval.reference_prefetch.ProcessPoolExecutor',FakePool):
                prefetch=ReferencePrefetch(self.engine(),[0],[['scene',0]],iter([]),root,root,self.config())
                try:
                    with self.assertRaisesRegex(ValueError,'Partial preparation'):prefetch.get(0)
                    self.assertEqual([],prefetch.pool.jobs)
                finally:prefetch.close()

    def test_export_order_mismatch_prevents_reference_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=SimpleNamespace(meta={'scene_name':'scene','frame_idx':9},frame_dir=root/'sample')
            with patch('pcm_eval.reference_prefetch.ProcessPoolExecutor',FakePool):
                prefetch=ReferencePrefetch(self.engine(),[0],[['scene',0]],iter([manifest]),root,root,self.config())
                try:
                    with self.assertRaisesRegex(ValueError,'order differs'):prefetch.get(0)
                    self.assertEqual([],prefetch.pool.jobs)
                finally:prefetch.close()


if __name__=='__main__':unittest.main()
