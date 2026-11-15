import copy,json,shutil,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from pcm_eval.common import digest,save_json,load_json,json_digest
from pcm_eval.vision_gt_binding import prepare,verify,attach,BindingRejected
from pcm_eval.lpai_native_model import build_plan,launch_batch


class VisionBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'submitted';self.source=self.root/'original-task';self.source.mkdir()
        self.image=self.data/'scene/dataset/18/18_f_ca_120.jpg';self.image.parent.mkdir(parents=True)
        Image.new('RGB',(16,12),(30,45,60)).save(self.image)
        self.old=self.source/'18_f_ca_120_123.jpg';self.old.write_bytes(self.image.read_bytes()+b'\nmetadata-only-difference')
        self.label=save_json(self.source/'img_label.json',{self.old.name:{'label':{'rainfall_intensity':0,'basic_road_surface':1}}})
        self.index=save_json(self.data/'data_link.json',{'train':{},'eval':{'scene':{'scene_name':'scene','frames':[{'frame_index':18,'time_stamp':123000}]}},'vision_eval':{'scene':{'frames':[]}}})
        self.candidate={'scene':'scene','frame_index':18,'image':str(self.image),'image_sha256':digest(self.image),
            'source_image':str(self.old),'source_image_sha256':digest(self.old),'source_label':str(self.label),'source_label_sha256':digest(self.label),
            'origin':{'task_uid':'original-task','dataset_ref':'original-label-reference'}}
        self.dataset={'index_path':str(self.index),'index_sha256':digest(self.index),'version':'submitted-version',
                      'mount_receipt':{'state':'MOUNT_VERIFIED','mount_path':str(self.data)},'frames_by_head':{'occ':[['scene',18]],'vision':[]}}
        self.profile={'heads':['occ','vision'],'source_commit':'commit','source_root':str(self.root),'py_config':'config.py','version':'model','gpu_index':0,
                      'environment':{},'main_checkpoint':{'sha256':'main','path':'main.pth'},'classifier_checkpoint':{'sha256':'classifier','path':'classifier.pth'},
                      'binding':{'code':{'evaluation_commit':'commit'},'main_checkpoint':{'sha256':'main'},'classifier_checkpoint':{'sha256':'classifier'}}}

    def prepare(self,candidates=None,output='binding'):
        return prepare(dataset_root=self.data,index_path=self.index,index_sha256=digest(self.index),
            candidates=candidates or [self.candidate],approved_GT_roots=[self.source],required_tasks=['rainfall_intensity','basic_road_surface'],output=self.root/output)

    def binding(self):
        self.prepare();p=self.root/'binding/binding.json';return {'path':str(p),'sha256':digest(p)}

    def test_verified_same_pixels_preserve_labels_and_launch_current_dataset(self):
        before=digest(self.index);binding=self.binding();dataset=attach(self.dataset,binding)
        self.assertEqual(digest(self.index),before)
        self.assertNotEqual(digest(self.image),digest(self.old))
        plan=build_plan(self.profile,dataset,'small-validation',selection={'scene_count':1,'frames_per_scene':1})
        folder=self.root/'launch';folder.mkdir();launch=launch_batch(plan,'vision',[['scene',18]],folder)
        self.assertEqual(launch['environment']['DATASET_PATH'],str(self.data))
        index=load_json(launch['environment']['DATA_LINK_FILE']);frame=index['vision_eval']['scene']['frames'][0]
        self.assertEqual(frame['img_file'],self.image.name)
        self.assertEqual(load_json(frame['img_label_file']),{self.image.name:{'label':{'rainfall_intensity':0,'basic_road_surface':1}}})
        self.assertEqual(launch['expected_frames'],1)
        self.assertFalse(load_json(binding['path'])['scene_tags_used'])

    def test_same_scene_and_frame_number_with_different_image_is_rejected(self):
        Image.new('RGB',(16,12),(31,45,60)).save(self.old)
        self.candidate['source_image_sha256']=digest(self.old)
        with self.assertRaises(BindingRejected) as caught:self.prepare()
        self.assertEqual(caught.exception.code,'SOURCE_IMAGE_DIFFERS')
        self.assertFalse((self.root/'binding').exists())

    def test_controller_planning_does_not_read_LPAI_files_but_native_launch_rechecks(self):
        binding=self.binding();dataset=attach(self.dataset,binding)
        with patch('pcm_eval.vision_gt_binding.verify',side_effect=AssertionError('Controller cannot read LPAI files')):
            plan=build_plan(self.profile,dataset,'controller-plan',selection={'scene_count':1,'frames_per_scene':1})
        self.old.write_bytes(b'changed after planning')
        folder=self.root/'launch';folder.mkdir()
        with self.assertRaises(BindingRejected):launch_batch(plan,'vision',[['scene',18]],folder)

    def test_small_binding_cannot_be_reported_as_full_coverage(self):
        dataset=attach(self.dataset,self.binding())
        with self.assertRaises(ValueError):build_plan(self.profile,dataset,'incorrect-full')

    def test_bound_frames_are_not_lost_when_other_heads_use_fresh_frame_offsets(self):
        dataset=attach(self.dataset,self.binding())
        dataset['frames_by_head']['occ']=[['s',1],['s',2]]
        plan=build_plan(self.profile,dataset,'fresh-small',selection={'scene_count':1,'frames_per_scene':1,'frame_offset':1})
        self.assertEqual(plan['heads']['occ'],[['s',2]])
        self.assertEqual(plan['heads']['vision'],[['scene',18]])

    def test_scene_tags_and_histograms_are_not_per_image_annotations(self):
        self.label.write_text(json.dumps({'vlm_tags_list':{'rainfall_intensity':0},'label_stats':{'label_counts':{'rainfall_intensity':{'0':20}}}}))
        self.candidate['source_label_sha256']=digest(self.label)
        with self.assertRaises(BindingRejected) as caught:self.prepare()
        self.assertEqual(caught.exception.code,'NO_EXACT_IMAGE_LABEL')

    def test_changed_label_or_image_after_binding_fails_before_native_launch(self):
        binding=self.binding();self.label.write_text('{}')
        with self.assertRaises(BindingRejected):attach(self.dataset,binding)

    def test_original_training_frame_cannot_be_added_to_submitted_eval(self):
        bad=copy.deepcopy(self.candidate);bad['frame_index']=99
        with self.assertRaises(BindingRejected) as caught:self.prepare([bad])
        self.assertEqual(caught.exception.code,'NOT_IN_SUBMITTED_VALIDATION')

    def test_other_dataset_binding_is_rejected(self):
        binding=self.binding();dataset=copy.deepcopy(self.dataset);dataset['index_sha256']='other-index'
        with self.assertRaises(BindingRejected) as caught:verify(dataset,binding)
        self.assertEqual(caught.exception.code,'DIFFERENT_SUBMITTED_DATASET')

    def test_extra_unverified_frame_in_derived_index_is_rejected(self):
        binding=self.binding();receipt=load_json(binding['path']);link=Path(receipt['derived_index']);data=load_json(link)
        data['vision_eval']['scene']['frames'].append({'frame_index':19});link.write_text(json.dumps(data))
        with self.assertRaises(BindingRejected):verify(self.dataset,binding)

    def test_incomplete_task_set_and_duplicate_frames_are_rejected(self):
        label={self.old.name:{'label':{'rainfall_intensity':0}}};self.label.write_text(json.dumps(label));self.candidate['source_label_sha256']=digest(self.label)
        with self.assertRaises(BindingRejected) as caught:self.prepare()
        self.assertEqual(caught.exception.code,'REQUIRED_TASK_LABELS_MISSING')
        self.label.write_text(json.dumps({self.old.name:{'label':{'rainfall_intensity':0,'basic_road_surface':1}}}));self.candidate['source_label_sha256']=digest(self.label)
        with self.assertRaises(BindingRejected) as caught:self.prepare([self.candidate,self.candidate])
        self.assertEqual(caught.exception.code,'DUPLICATE_FRAME')


if __name__=='__main__':unittest.main()
