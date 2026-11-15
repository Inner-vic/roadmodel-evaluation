import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
from board_multiframe_subnet import stage_info, collect_stage, public_ports
import board_multiframe_subnet as board_module


class MultiFrameContractTests(unittest.TestCase):
    def test_four_frame_five_stage_chain_with_actual_prefixed_dump_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';cache=Path(tmp)/'pcm-v322-fast-cache-test'
            frames=[Path(str(root)+f'-f{i}') for i in range(4)]
            contracts=[];number=0
            for stage,count in enumerate((15,15,9,8,2)):
                outputs=[]
                for i in range(count):
                    outputs.append({'tensor':f't{number}','filename':f't{number}.bin',
                        'declared_data_path':f'RESULT_t{number}.bin','nbytes':4,'dtype':'float16'})
                    number+=1
                contracts.append({'stage':stage,'inputs':[{}],'outputs':outputs})
            public=[dict(p,stage=s['stage']) for s in contracts for p in s['outputs']][:18]
            manifest={'schema':'roadmodel.multiframe-subnets/v1','root':str(root),'cache':str(cache),
                      'frames':[{'root':str(p),'packet':str(p)+'.request.tar','sha256':'dummy'} for p in frames]}
            raw=json.dumps(manifest).encode();path=Path(str(root)+'.json');path.write_bytes(raw)
            def initialize(p,c):
                p.mkdir();(p/'inputs').mkdir()
                for i in range(5):
                    (p/f'sub{i}/output').mkdir(parents=True);(p/f'sub{i}/export').mkdir()
            def packet(p,sha,frame,c):
                slot=frames.index(frame);files={'inputs/x.bin':np.array([slot+1,slot+1],dtype=np.float16).tobytes()}
                for i,contract in enumerate(contracts):
                    source=frame/'inputs/x.bin' if i==0 else frame/f'sub{i-1}/export'/contracts[i-1]['outputs'][0]['filename']
                    info={'input':[{'data_paths':[str(source)]}],
                          'output':[{'data_path':p['declared_data_path']} for p in contract['outputs']]}
                    files[f'sub{i}/model_info.json']=json.dumps(info).encode()
                return {'public_outputs':public},files,contracts
            calls=[]
            def native(argv,*,cwd,**kwargs):
                calls.append(cwd);info=json.loads((Path(cwd)/'model_info.json').read_text())
                self.assertEqual(info['infer_num'],4)
                for slot,source in enumerate(info['input'][0]['data_paths']):
                    values=np.fromfile(source,dtype=np.float16)
                    self.assertEqual(float(values[0]),float(slot+1))
                    for port in info['output']:
                        dest=Path(port['data_path']);dest.parent.mkdir(exist_ok=True)
                        values.tofile(dest.with_name(f'Frame_{slot}_'+dest.name))
                return SimpleNamespace(returncode=0,stdout=b'four frames',stderr=b'')
            with patch.object(board_module,'checked_root',side_effect=Path),patch.object(board_module,'initialize_frame',side_effect=initialize),patch.object(board_module,'read_packet',side_effect=packet):
                result=board_module.execute(path,hashlib.sha256(raw).hexdigest(),run=native)
            self.assertEqual(len(calls),5)
            self.assertEqual(result['frames'],4)
            self.assertEqual(len(result['bundles']),4)
            for slot,frame in enumerate(frames):
                for contract in contracts:
                    for port in contract['outputs']:
                        np.testing.assert_array_equal(np.fromfile(frame/f"sub{contract['stage']}/export"/port['filename'],dtype=np.float16),[slot+1,slot+1])
            cache.mkdir();(cache/'keep').write_text('immutable model cache')
            with patch.object(board_module,'checked_root',side_effect=Path):
                clean=board_module.cleanup(path,hashlib.sha256(raw).hexdigest())
            self.assertEqual(clean['state'],'CLEANED')
            self.assertFalse(root.exists())
            self.assertTrue(all(not p.exists() for p in frames))
            self.assertTrue((cache/'keep').exists())

    def test_supported_batch_sizes_preserve_tensor_shapes(self):
        for count in (1,2,4,8):
            templates=[{'input':[{'shape':[1,3],'data_paths':[f'/f{i}/x']}],
                        'output':[{'shape':[1,7],'data_path':'old'}]} for i in range(count)]
            info=stage_info(templates,{'inputs':[{}],'outputs':[{'declared_data_path':'out.bin'}]},
                            [Path(f'/f{i}/sub0') for i in range(count)])
            self.assertEqual(info['infer_num'],count)
            self.assertEqual(info['input'][0]['shape'],[1,3])
            self.assertEqual(info['output'][0]['shape'],[1,7])
            self.assertEqual(len(info['input'][0]['data_paths']),count)

    def test_public_tensor_producer_is_derived_from_contract(self):
        graph={'stages':[{'stage':3,'outputs':[{'tensor':'classification'}]},
                         {'stage':4,'outputs':[{'tensor':'trajectory'}]}]}
        ports=[{'tensor':'trajectory','filename':'t.bin'},{'tensor':'classification','filename':'c.bin'}]
        self.assertEqual([p['stage'] for p in public_ports(graph,ports)],[4,3])
        self.assertNotIn('stage',ports[0])

    def test_all_29_inputs_preserve_actual_subnet_order(self):
        templates = []
        for frame in range(2):
            templates.append({'input':[{'tensor':str(i),'data_paths':[f'/frame{frame}/port{i}']}
                                       for i in range(29)],
                              'output':[{'data_path':'old','golden_paths':['sample']}],
                              'infer_num':1,'buffer_batch_num':1})
        original = copy.deepcopy(templates)
        stage = {'inputs':[{}]*29,'outputs':[{'declared_data_path':'value.bin'}]}
        info = stage_info(templates, stage, [Path('/f0/sub2'),Path('/f1/sub2')])
        self.assertEqual(templates, original)
        self.assertEqual(info['infer_num'], 2)
        self.assertEqual(info['buffer_batch_num'], 1)
        for i, port in enumerate(info['input']):
            self.assertEqual(port['data_paths'], [f'/frame0/port{i}',f'/frame1/port{i}'])
        self.assertEqual(info['output'][0]['golden_paths'], [])
        self.assertFalse(info['do_compare'])

    def fixture(self, root):
        shared = root/'batch/sub0'
        shared.mkdir(parents=True)
        for i in range(2):
            own = root/f'f{i}/sub0'
            (own/'export').mkdir(parents=True)
            (own/'output').mkdir()
        stage = {'stage':0, 'outputs':[{'tensor':'x','filename':'x.bin','declared_data_path':'x.bin',
                                      'nbytes':4,'dtype':'float16'}]}
        return shared, stage

    def test_prefixed_outputs_are_bound_to_correct_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);shared,stage = self.fixture(root)
            for i in range(2):
                np.array([i+1,i+2],dtype=np.float16).tofile(shared/f'Frame_{i}_x.bin')
            for i in range(2):
                rows = collect_stage(shared,root/f'f{i}',stage,i)
                self.assertEqual(rows[0]['frame_slot'],i)
                self.assertEqual(rows[0]['sha256'],hashlib.sha256((shared/f'Frame_{i}_x.bin').read_bytes()).hexdigest())
                np.testing.assert_array_equal(np.fromfile(rows[0]['remote_file'],dtype=np.float16),[i+1,i+2])

    def test_bare_shared_dump_cannot_be_assigned_to_either_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared,stage=self.fixture(root)
            np.ones(2,dtype=np.float16).tofile(shared/'x.bin')
            with self.assertRaisesRegex(ValueError,'Missing frame-specific'):
                collect_stage(shared,root/'f0',stage,0)

    def test_frame1_prefixed_dump_in_frame0_directory_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared,stage=self.fixture(root)
            np.array([7,8],dtype=np.float16).tofile(root/'f0/sub0/output/Frame_1_x.bin')
            rows=collect_stage(shared,root/'f1',stage,1,[root/'f0',root/'f1'])
            np.testing.assert_array_equal(np.fromfile(rows[0]['remote_file'],dtype=np.float16),[7,8])

    def test_recovery_reuses_only_identical_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared,stage=self.fixture(root)
            np.ones(2,dtype=np.float16).tofile(shared/'Frame_0_x.bin')
            first=collect_stage(shared,root/'f0',stage,0)
            again=collect_stage(shared,root/'f0',stage,0,allow_existing=True)
            self.assertEqual(first[0]['sha256'],again[0]['sha256'])
            with self.assertRaisesRegex(ValueError,'Never overwrite'):
                collect_stage(shared,root/'f0',stage,0)

    def test_conflicting_aliases_fail_instead_of_silently_selecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared,stage=self.fixture(root)
            np.ones(2,dtype=np.float16).tofile(shared/'Frame_0_x.bin')
            np.zeros(2,dtype=np.float16).tofile(root/'f0/sub0/output/x.bin')
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                collect_stage(shared,root/'f0',stage,0)

    def test_nonfinite_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared,stage=self.fixture(root)
            np.array([float('nan'),0],dtype=np.float16).tofile(shared/'Frame_1_x.bin')
            with self.assertRaisesRegex(ValueError,'Non-finite'):
                collect_stage(shared,root/'f1',stage,1)


if __name__ == '__main__':
    unittest.main()
