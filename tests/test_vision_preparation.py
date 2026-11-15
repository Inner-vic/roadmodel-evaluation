import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcm_eval.common import digest, json_digest, load_json, save_json
from pcm_eval.vision_preparation import prepare, verify
from pcm_eval.vision_gt_binding import BindingRejected
from pcm_eval.lpai_native_model import build_plan, launch_batch, run
from pcm_eval.lpai_model_transport import completed_heads


class VisionPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'submitted'
        self.frames = [['scene', 18], ['scene', 19]]
        for _, fid in self.frames:
            folder = self.data/'scene/dataset'/str(fid)
            folder.mkdir(parents=True)
            (folder/f'{fid}_f_ca_120.jpg').write_bytes(b'fixture-image-'+str(fid).encode())
        self.index = save_json(self.data/'data_link.json', {
            'eval': {'scene': {'frames': [{'frame_index': f} for _, f in self.frames]}},
            'vision_eval': {'scene': {'frames': []}}})
        self.dataset = {'version': 'submitted', 'index_path': str(self.index), 'index_sha256': digest(self.index),
                        'frames_by_head': {'occ': self.frames, 'vision': []},
                        'mount_receipt': {'state': 'MOUNT_VERIFIED', 'mount_path': str(self.data)}}
        weight = self.root/'weight';weight.write_bytes(b'fixture-weights')
        manifest = save_json(self.root/'source.json', {'commit': 'fixture', 'files': {}})
        checkpoint = {'sha256': digest(weight), 'path': str(weight)}
        self.profile = {'heads': ['occ', 'vision'], 'source_commit': 'fixture', 'source_root': str(self.root),
                        'source_manifest': str(manifest), 'py_config': 'config.py', 'version': 'fixture',
                        'gpu_index': 0, 'environment': {}, 'main_checkpoint': checkpoint,
                        'classifier_checkpoint': checkpoint, 'binding': {'code': {'evaluation_commit': 'fixture'},
                        'main_checkpoint': checkpoint, 'classifier_checkpoint': checkpoint}}

    def label(self, fid, *, image_key=None, name=None, values=None):
        return save_json(self.data/'scene/dataset'/str(fid)/(name or f'img_label_{fid}.json'),
                         {image_key or f'{fid}_f_ca_120.jpg': {'label': values or {'rainfall_intensity': 0, 'basic_road_surface': 1}}})

    def plan(self):
        return build_plan(self.profile, self.dataset, 'core-test', selection={'scene_count': 1, 'frames_per_scene': 2})

    def test_empty_vision_split_finds_exact_per_image_labels_without_changing_input(self):
        for _, f in self.frames:self.label(f)
        before = digest(self.index)
        plan = self.plan()
        self.assertEqual(plan['heads']['vision'], [])
        self.assertEqual(plan['vision_candidates'], self.frames)
        launch = launch_batch(plan, 'vision', self.frames, self.root/'launch')
        ready = verify(launch['vision_preparation'], ['rainfall_intensity', 'basic_road_surface'])
        self.assertEqual(ready['selected_frames'], self.frames)
        self.assertEqual(ready['source_split'], 'eval')
        self.assertEqual(digest(self.index), before)
        self.assertEqual(launch['environment']['DATASET_PATH'], str(self.data))
        self.assertEqual(len(load_json(ready['index_path'])['vision_eval']['scene']['frames']), 2)
        self.assertFalse(list((self.root/'launch').rglob('*.jpg')))

    def test_scene_tags_and_other_timestamp_labels_do_not_become_frame_GT(self):
        save_json(self.data/'scene/tags.json', {'vlm_tags_list': {'basic_road_surface': 0}})
        self.label(18, image_key='18_f_ca_120_old_capture.jpg')
        receipt = prepare(self.dataset, self.frames, self.root/'prepared')
        self.assertEqual(receipt['state'], 'BLOCKED')
        self.assertEqual([x['code'] for x in receipt['issues']], ['PER_IMAGE_GT_NOT_FOUND']*2)
        self.assertFalse((self.root/'prepared/data-link.json').exists())

    def test_missing_one_label_does_not_silently_shrink_evaluation_scope(self):
        self.label(18)
        receipt = prepare(self.dataset, self.frames, self.root/'prepared')
        self.assertEqual(receipt['state'], 'BLOCKED')
        self.assertEqual(receipt['selected_frames'], self.frames)
        self.assertEqual(receipt['issues'][0]['frame_index'], 19)
        self.assertFalse((self.root/'prepared/data-link.json').exists())

    def test_conflicting_labels_are_not_resolved_by_filename_sorting(self):
        self.label(18)
        self.label(18, name='img_label_second.json', values={'basic_road_surface': 2})
        receipt = prepare(self.dataset, [['scene', 18]], self.root/'prepared')
        self.assertEqual(receipt['issues'][0]['code'], 'LABEL_MISSING_OR_AMBIGUOUS')

    def test_explicit_label_declaration_does_not_fall_back_to_another_label(self):
        valid = self.label(18)
        wrong = self.label(18, name='img_label_old.json', image_key='old_capture.jpg')
        data = load_json(self.index)
        data['vision_eval']['scene']['frames'] = [{'frame_index': 18, 'img_file': '18_f_ca_120.jpg', 'img_label_file': wrong.name}]
        self.index.write_text(__import__('json').dumps(data), encoding='utf-8')
        self.dataset['index_sha256'] = digest(self.index)
        receipt = prepare(self.dataset, [['scene', 18]], self.root/'prepared')
        self.assertTrue(valid.exists())
        self.assertEqual(receipt['issues'][0]['code'], 'NO_EXACT_IMAGE_LABEL')

    def test_required_tasks_are_checked_from_actual_model_configuration(self):
        self.label(18, values={'rainfall_intensity': 0})
        receipt = prepare(self.dataset, [['scene', 18]], self.root/'prepared')
        with self.assertRaises(BindingRejected) as caught:verify(receipt, ['basic_road_surface'])
        self.assertEqual(caught.exception.code, 'REQUIRED_TASK_LABELS_MISSING')

    def test_labels_changed_after_preparation_are_rejected_before_inference(self):
        label = self.label(18)
        receipt = prepare(self.dataset, [['scene', 18]], self.root/'prepared')
        label.write_text('{}')
        with self.assertRaises(BindingRejected) as caught:verify(receipt)
        self.assertEqual(caught.exception.code, 'SOURCE_CHANGED')

    def test_changed_index_and_non_validation_frame_are_rejected(self):
        receipt = prepare(self.dataset, [['scene', 99]], self.root/'prepared')
        self.assertEqual(receipt['issues'][0]['code'], 'NOT_IN_SUBMITTED_VALIDATION')
        self.index.write_text('{}')
        with self.assertRaises(BindingRejected) as caught:prepare(self.dataset, self.frames, self.root/'changed')
        self.assertEqual(caught.exception.code, 'INDEX_CHANGED')

    def test_invalid_values_and_escaping_paths_are_blocked(self):
        self.label(18, values={'basic_road_surface': True})
        receipt = prepare(self.dataset, [['scene', 18]], self.root/'invalid')
        self.assertEqual(receipt['issues'][0]['code'], 'INVALID_CLASSIFICATION_LABELS')
        receipt = prepare(self.dataset, [['../outside', 18]], self.root/'escape')
        self.assertEqual(receipt['issues'][0]['code'], 'INVALID_FRAME_PATH')

    def _run(self, vision_failure=False):
        plan = self.plan();output = self.root/'run'
        save_json(output/'occ-aggregate/normalized-result.json', {'fixture': True})
        def batch(plan, kind, frames, folder):
            if kind == 'vision' and vision_failure:raise ValueError('fixture native failure')
            save_json(Path(folder)/'normalized-result.json', {'fixture': kind})
        with patch('pcm_eval.lpai_native_model.run_batch', side_effect=batch) as invoke:
            result = run(plan, output)
        return plan, result, invoke

    def test_missing_GT_still_returns_completed_OCC_and_precise_frame_issues(self):
        plan, result, invoke = self._run()
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(result['head_statuses']['vision']['state'], 'BLOCKED')
        self.assertEqual(result['head_statuses']['vision']['issues'][0]['frame_index'], 18)
        self.assertEqual(completed_heads(plan, result), {'occ': self.frames})
        self.assertEqual(load_json(self.root/'run/progress.json')['state'], 'PARTIAL')

    def test_all_labels_ready_executes_both_heads_with_frozen_scope(self):
        for _, f in self.frames:self.label(f)
        plan, result, invoke = self._run()
        self.assertEqual([c.args[1] for c in invoke.call_args_list], ['occ', 'vision'])
        self.assertEqual(result['incomplete_heads'], [])
        self.assertEqual(completed_heads(plan, result), {'occ': self.frames, 'vision': self.frames})
        self.assertEqual(load_json(self.root/'run/progress.json')['state'], 'SUCCEEDED')

    def test_vision_native_failure_keeps_OCC_reportable_without_collecting_failed_head(self):
        for _, f in self.frames:self.label(f)
        plan, result, _ = self._run(vision_failure=True)
        self.assertEqual(result['state'], 'succeeded')
        self.assertEqual(result['head_statuses']['vision']['state'], 'NEEDS_REVIEW')
        self.assertEqual(result['head_result_files'], [])
        self.assertEqual(completed_heads(plan, result), {'occ': self.frames})

    def test_collection_rejects_out_of_scope_and_partial_claims(self):
        plan, result, _ = self._run()
        for frames in ([['scene', 18]], [['scene', 18], ['scene', 20]]):
            changed = copy.deepcopy(result);changed['executed_frames_by_head']['vision'] = frames
            with self.assertRaises(ValueError):completed_heads(plan, changed)
        changed = copy.deepcopy(result);changed['plan_sha256'] = 'other'
        with self.assertRaises(ValueError):completed_heads(plan, changed)

    def test_native_launch_cannot_expand_the_frozen_candidate_scope(self):
        with self.assertRaises(BindingRejected) as caught:
            launch_batch(self.plan(), 'vision', [['scene', 20]], self.root/'outside-plan')
        self.assertEqual(caught.exception.code, 'VISION_SCOPE_DIFFERS_FROM_PLAN')


if __name__ == '__main__':
    unittest.main()
