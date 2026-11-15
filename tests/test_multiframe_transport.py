import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from multiframe_subnet_driver import pack_batch,unpack_batch
from multiframe_subnet_driver import Engine
import fast_subnet_driver as original_driver


class FakeEngine:
    cache='/userdata/benchEval/models/pcm-v322-fast-cache-test'
    def input_packet(self,root,board):return (str(root)+'|'+board).encode(),[]


class TransportTests(unittest.TestCase):
    def test_dispatcher_does_not_initialize_an_unused_reference_session(self):
        engine=Engine()
        try:
            self.assertIsNone(engine.session)
            self.assertEqual(len(engine.public),18)
            self.assertEqual(len(engine.graph['stages']),5)
            with self.assertRaisesRegex(ValueError,'dedicated CPU session'):
                engine.prepare(None,None)
        finally:engine.close()

    def test_reference_worker_keeps_original_provider_and_session_options(self):
        engine=original_driver.Engine()
        try:
            self.assertEqual(engine.session.get_providers(),['CPUExecutionProvider'])
            opts=engine.session.get_session_options()
            self.assertEqual(opts.intra_op_num_threads,0)
            self.assertEqual(opts.graph_optimization_level,original_driver.ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            self.assertEqual(len(engine.output_names),18)
        finally:engine.close()

    def test_one_packet_contains_all_frames_and_unique_hashes(self):
        blob,manifest,sha=pack_batch(FakeEngine(),[Path('/frame0'),Path('/frame1')],'/userdata/benchEval/models/pcm-v322-subnets-full-test')
        with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:
            self.assertEqual(set(archive.getnames()),{'manifest.json','frame_0.tar','frame_1.tar'})
            raw=archive.extractfile('manifest.json').read()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),sha)
            for i,row in enumerate(manifest['frames']):
                self.assertEqual(hashlib.sha256(archive.extractfile(f'frame_{i}.tar').read()).hexdigest(),row['sha256'])
        self.assertNotEqual(manifest['frames'][0]['sha256'],manifest['frames'][1]['sha256'])

    def test_wrong_batch_identity_rejected_before_unpacking(self):
        with self.assertRaisesRegex(ValueError,'reply identity'):
            unpack_batch(b'',{'batch_packet_sha256':'wrong','frames':2,'invocations':5},{},'expected',[Path('/0'),Path('/1')],[])

    def test_unexpected_archive_path_rejected_before_writes(self):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:') as archive:
            entry=tarfile.TarInfo('../outside');entry.size=1;archive.addfile(entry,io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError,'inventory'):
            unpack_batch(stream.getvalue(),{'batch_packet_sha256':'sha','frames':2,'invocations':5},{},'sha',[Path('/0'),Path('/1')],[])


if __name__=='__main__':unittest.main()
