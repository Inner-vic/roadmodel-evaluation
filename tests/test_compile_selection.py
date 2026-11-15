import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_compile_workstation import select_frames
from pcm_eval.planning import build_plan


class CompileSelectionTests(unittest.TestCase):
    def link(self):
        scenes = {'scene-' + str(n): {'scene_name': 'scene-' + str(n),
                  'frames': [{'frame_index': f, 'occ_file': str(n) + '-' + str(f) + '.npy'} for f in range(4)]}
                  for n in range(5)}
        return {'eval': scenes, 'joint_eval': scenes, 'joint_train': {'training-only': {'frames': []}}}

    def test_uniform_scene_selection_preserves_real_metadata_and_excludes_training(self):
        source = self.link()
        result, frames = select_frames(source, {'selection': {'scene_count': 3, 'frames_per_scene': 2}})
        self.assertEqual(list(result['eval']), ['scene-0', 'scene-2', 'scene-4'])
        self.assertEqual(len(frames), 6)
        self.assertEqual(result['eval']['scene-2']['frames'][1], source['eval']['scene-2']['frames'][1])
        self.assertEqual(result['joint_train'], {})
        self.assertEqual(result['eval'], result['joint_eval'])

    def test_fraction_and_max_frames_apply_once_to_frozen_scope(self):
        result, frames = select_frames(self.link(), {'frame_fraction': 0.5, 'max_frames': 3})
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames, [('scene-0', 0), ('scene-0', 2), ('scene-1', 0)])

    def test_explicit_frames_keep_metadata_and_reject_unknown_duplicate_or_truncation(self):
        pairs = [['scene-2', 3], ['scene-4', 1]]
        source = self.link()
        result, actual = select_frames(source, {'selection': {'frames': pairs}, 'max_frames': 2})
        self.assertEqual(actual, [tuple(x) for x in pairs])
        self.assertEqual(result['eval']['scene-2']['frames'][0], source['eval']['scene-2']['frames'][3])
        self.assertEqual(result['joint_train'], {})
        for selection in ([['training-only', 1]], [['scene-2', 9]], pairs + [pairs[0]],
                          [['scene-2', 1], ['scene-4', 1], ['scene-2', 2]]):
            with self.assertRaises(ValueError):
                select_frames(source, {'selection': {'frames': selection}})
        with self.assertRaises(ValueError):
            select_frames(source, {'selection': {'frames': pairs}, 'max_frames': 1})

    def test_driver_plan_binds_hashes_manifest_and_run_id(self):
        request = {'kind': 'compile_alignment', 'version_name': 'fixture',
                   'execution': {'transport': 'ssh', 'ssh_alias': 'ubuntu-workstation',
                                 'repo_dir': '/repo', 'output_root': '/runs', 'board_ssh': 'ssh root@board',
                                 'adapter': 'workstation_driver', 'driver_script': '/tools/run_compile_workstation.py'},
                   'dataset': {'class_name': 'SmartChassisDatasetV7', 'path': '/data',
                               'data_link_file': '/data/link.json', 'export_root': '/data', 'required_manifest': '/manifest.json'},
                   'model': {'onnx': '/model.onnx', 'packup_tar': '/packup.tar.gz',
                             'onnx_sha256': 'a' * 64, 'packup_sha256': 'b' * 64},
                   'evaluation': {'v7_pipeline_verified': True, 'compute_gt_metrics': False}}
        plan = build_plan(request, 'fixture-001')
        self.assertTrue(plan['ready'])
        argv = plan['steps'][0]['argv']
        self.assertEqual(argv[1], '/tools/run_compile_workstation.py')
        self.assertEqual(json.loads(argv[-1])['run_id'], 'fixture-001')
        self.assertIn('/manifest.json', plan['input_paths'])


if __name__ == '__main__':
    unittest.main()
