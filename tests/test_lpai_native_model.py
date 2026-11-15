import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pcm_eval.common import digest, json_digest, load_json, save_json
from pcm_eval.lpai_native_model import select_frames, prepare_occ_input, run_batch, build_plan, run
from pcm_eval.space_lifecycle import advance
from tests import test_dataset_mount as mount_fixtures

URL = mount_fixtures.URL


class NativeModelTest(unittest.TestCase):
    def test_empty_vision_index_never_invents_frame_GT_from_scene_tags(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weight=root/'weight';weight.write_bytes(b'fixture-weight')
            manifest=save_json(root/'source.json',{'commit':'fixture','files':{}})
            profile={'main_checkpoint':{'path':str(weight),'sha256':digest(weight)},
                     'classifier_checkpoint':{'path':str(weight),'sha256':digest(weight)},
                     'source_manifest':str(manifest),'source_root':str(root),'source_commit':'fixture'}
            plan={'profile':profile,'heads':{'occ':[['s',29],['s',30]],'vision':[]},'run_id':'audit',
                  'batch_size':2,'dataset':{'scene_tags':{'s':{'vlm_tags_list':{'rainfall_intensity':1}}}}}
            plan['plan_sha256']=json_digest(plan)
            output=root/'out';save_json(output/'occ-aggregate/normalized-result.json',{'fixture':True})
            with patch('pcm_eval.lpai_native_model.run_batch',return_value={}) as batch:
                result=run(plan,output)
            self.assertEqual(batch.call_count,1)
            self.assertEqual(batch.call_args.args[1],'occ')
            self.assertEqual(result['head_result_files'],[])
            self.assertIn('vision_eval',result['incomplete_heads'][0])
            self.assertIn('GT绑定',result['incomplete_heads'][0])
            self.assertNotIn('缺少本数据源',result['incomplete_heads'][0])

    def test_model_and_compile_use_the_same_dynamic_frame_selection(self):
        from pcm_eval.compile_scope import build_plan as compile_plan
        frames = [[f'scene-{s}', i] for s in range(139) for i in range(61)]
        dataset = {'frames_by_head': {'occ': frames}, 'version': 'submitted-version', 'index_sha256': 'submitted-index'}
        request = {'version_name': 'v3.2.2'}
        full = compile_plan(request, dataset, 'rm-full-compile')
        self.assertEqual(full['frames'], frames)
        selection = {'scene_count': 2, 'frames_per_scene': 3}
        small = compile_plan(request, dataset, 'rm-small-compile', selection=selection)
        self.assertEqual(small['frames'], select_frames(frames, selection))
        self.assertEqual(small['expected_frames'], 6)

    def test_complete_scope_has_no_historical_frame_limit(self):
        frames = [[f'scene-{scene}', frame] for scene in range(137) for frame in range(67)]
        self.assertEqual(select_frames(frames), frames)
        self.assertEqual(select_frames(frames, {'scene_count': 2, 'frames_per_scene': 3}),
                         [['scene-0', i] for i in range(3)]+[['scene-1', i] for i in range(3)])
        with self.assertRaises(ValueError):
            select_frames(frames+[frames[0]])

    def test_validation_can_select_fresh_frames_without_expanding_the_scope(self):
        frames=[[s,i] for s in ('a','b') for i in range(5)]
        selection={'scene_count':1,'frames_per_scene':2,'scene_offset':1,'frame_offset':2}
        self.assertEqual(select_frames(frames,selection),[['b',2],['b',3]])
        self.assertEqual(select_frames([],selection),[])
        for invalid in (-1,True):
            with self.assertRaises(ValueError):select_frames(frames,dict(selection,frame_offset=invalid))

    def test_gt_is_adapted_inside_run_without_copying_raw_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root/'dataset'
            fr = dataset/'scene/dataset/18'
            fr.mkdir(parents=True)
            gt = np.array([[1, 1, 50, 1, 1], [799, 1, 50, 2, 0]], dtype=np.int32)
            np.save(fr/'occ.npy', gt)
            (fr/'18_f_ca_120.jpg').write_bytes(b'kept-at-mount')
            index = save_json(dataset/'data_link.json', {'eval': {'scene': {'scene_name': 'scene', 'frames': [
                {'frame_index': 18, 'occ_file': 'occ.npy', 'height_field_file': 'height.npy'}]}}, 'joint_eval': {}})
            contract = {'source_min': [5, -3.6, -2], 'source_shape': [800, 48, 200],
                        'voxel_size': [.05, .15, .02], 'target_min': [5, -3.6, -1], 'target_shape': [400, 48, 100]}
            plan = {'dataset': {'mount_receipt': {'mount_path': str(dataset)}, 'index_path': str(index), 'index_sha256': digest(index)},
                    'profile': {'binding': {'gt_contract': contract}}}
            path = prepare_occ_input(plan, [['scene', 18]], root/'run')
            frame = load_json(path)['eval']['scene']['frames'][0]
            self.assertEqual(np.load(frame['occ_file']).tolist(), [[1, 1, 0, 1, 1]])
            self.assertTrue(np.array_equal(np.load(fr/'occ.npy'), gt))
            self.assertFalse(list((root/'run').rglob('*.jpg')))
            self.assertIsNone(frame['height_field_file'])
            receipt = load_json(root/'run/gt-preparation.json')
            self.assertEqual(receipt['execution_location'], 'lpai')
            self.assertFalse(receipt['raw_data_copied_to_workstation'])
            index.write_text('{}')
            with self.assertRaises(ValueError):
                prepare_occ_input(plan, [['scene', 18]], root/'changed')

    def test_existing_failed_batch_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            save_json(Path(folder)/'execution_receipt.json', {'state': 'failed', 'selected_frames': [['s', 1]]})
            with patch('pcm_eval.lpai_native_model.subprocess.run', side_effect=AssertionError('No replay')):
                with self.assertRaises(ValueError):
                    run_batch({}, 'occ', [['s', 1]], folder)

    def test_native_plan_requires_mounted_data_and_bound_weight_identity(self):
        profile = {'heads': ['occ', 'vision'], 'source_commit': 'commit', 'batch_size': 100,
                   'binding': {'code': {'evaluation_commit': 'commit'}, 'main_checkpoint': {'sha256': 'main'},
                               'classifier_checkpoint': {'sha256': 'classifier'}},
                   'main_checkpoint': {'sha256': 'main'}, 'classifier_checkpoint': {'sha256': 'classifier'}}
        dataset = {'mount_receipt': {'state': 'MOUNT_VERIFIED'}, 'frames_by_head': {'occ': [['s', i] for i in range(203)]}}
        plan = build_plan(profile, dataset, 'rm-full-model')
        self.assertEqual(len(plan['heads']['occ']), 203)
        self.assertEqual(plan['heads']['vision'], [])
        self.assertEqual(plan['scope'], 'full')
        self.assertEqual(plan['plan_sha256'], json_digest({k: v for k, v in plan.items() if k != 'plan_sha256'}))
        profile['main_checkpoint']['sha256'] = 'other'
        with self.assertRaises(ValueError):
            build_plan(profile, dataset, 'rm-other')


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        fixture = mount_fixtures.DatasetMountTest()
        fixture.setUp()
        self.raw, self.space = fixture.raw, fixture.space
        self.volume = fixture.volume
        self.space['spec']['clusterName'] = 'cluster'
        self.template = {'namespace': 'team', 'cluster': 'cluster', 'image': 'registered-image',
                         'resources': {'gpu_type': 'registered-a100'}, 'accesses': {'ssh_credential': 'registered-key'},
                         'idle_shutdown': {'enabled': True, 'minutes': 240}}
        self.profile = {'namespace': 'team', 'space_name': 'mine', 'space_uid': 'space-1',
                        'cluster': 'cluster', 'environment_template': self.template}
        self.actions, self.saved = [], []

    def tick(self, journal, *, observation=None, cli_missing=False, idle=True):
        def query(args):
            if args[0] == 'assets':
                return self.raw
            if cli_missing:
                raise ValueError('404')
            return self.space
        return advance(URL, self.profile, journal, query=query, observe_ui=lambda profile: observation,
            request_ui=lambda action, context: self.actions.append((action, context)), save=self.saved.append,
            assert_idle=lambda space: idle,
            probe=lambda plan: {'state': 'MOUNT_VERIFIED', 'asset_binding_sha256': plan['asset']['binding_sha256'],
                                'dataset_source': {'source_ref': URL, 'frames_by_head': {'occ': [['s', 1]]}}})

    def observation(self, *, exists=True):
        import time
        return {'source': 'LPAI_UI', 'namespace': 'team', 'space_name': 'mine', 'observed_at': time.time(),
                'exists': exists, 'space': copy.deepcopy(self.space) if exists else None}

    def test_cli_404_requests_observation_instead_of_creating_a_duplicate(self):
        result = self.tick({}, cli_missing=True)
        self.assertEqual(result['state'], 'WAITING_UI')
        self.assertEqual([a[0] for a in self.actions], ['observe'])

    def test_creation_requires_explicit_page_absence_and_complete_template(self):
        result = self.tick({}, cli_missing=True, observation=self.observation(exists=False))
        self.assertEqual(self.actions[0][0], 'create')
        self.assertEqual(self.actions[0][1]['template'], self.template)
        self.tick(result, cli_missing=True, observation=self.observation(exists=False))
        self.assertEqual(len(self.actions), 1)

    def test_expired_observation_waits_without_repeating_submitted_start(self):
        self.space['spec']['dataVols'] = [self.volume()]
        self.space['status']['phase'] = 'stopped'
        state = self.tick({})
        observation = self.observation()
        observation['observed_at'] -= 301
        resumed = self.tick(state, cli_missing=True, observation=observation)
        self.assertEqual(resumed['state'], 'WAITING_UI')
        self.assertEqual([a[0] for a in self.actions], ['start', 'observe'])
        self.assertEqual(resumed['intent'], state['intent'])

    def test_mount_change_preserves_busy_space_and_does_not_probe_other_data(self):
        self.space['spec']['dataVols'] = [dict(self.volume(), version='old')]
        result = self.tick({}, idle=False)
        self.assertEqual(result['state'], 'PREPARING')
        self.assertEqual(self.actions, [])

    def test_stop_mount_start_and_verified_data_resume_the_same_request(self):
        self.space['spec']['dataVols'] = [dict(self.volume(), version='old')]
        state = self.tick({})
        self.assertEqual(self.actions[-1][0], 'stop')
        self.space['status']['phase'] = 'stopped'
        state = self.tick(state)
        self.assertEqual(self.actions[-1][0], 'configure_and_start')
        self.space['spec']['dataVols'].append(self.volume())
        self.space['status']['phase'] = 'ready'
        state = self.tick(state)
        self.assertEqual(state['state'], 'DATA_READY')
        self.assertEqual(state['dataset_source']['source_ref'], URL)
        self.assertEqual(len(self.actions), 2)

    def test_stopped_space_with_correct_mount_only_requests_start_once(self):
        self.space['spec']['dataVols'] = [self.volume()]
        self.space['status']['phase'] = 'stopped'
        state = self.tick({})
        self.tick(state)
        self.assertEqual([a[0] for a in self.actions], ['start'])


if __name__ == '__main__':
    unittest.main()
