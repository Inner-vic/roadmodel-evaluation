import copy
import tempfile
import unittest
from pathlib import Path

from pcm_eval.common import digest, json_digest, load_json, save_json
from pcm_eval.evaluation_inputs import validate_inputs
from pcm_eval.lpai_native_model import build_plan, launch_batch
from pcm_eval.vision_preparation import prepare

BASE = 'https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/roads?version='
INPUTS = {'model_delivery_url': 'https://li.feishu.cn/record/test',
          'occ_dataset_url': BASE+'occ', 'vision_dataset_url': BASE+'vision'}


class ThreeInputTest(unittest.TestCase):
    def test_single_dataset_source_never_calls_historical_visual_asset_adapter(self):
        from unittest.mock import patch, Mock
        from pcm_eval.evaluation_source import prepare as prepare_source
        dataset={'source_ref':BASE+'current','version':'current','index_sha256':'current-index'}
        workflow={'inputs':{'model_delivery_url':INPUTS['model_delivery_url'],'dataset_ref':BASE+'current'},
                  'dataset_source':dataset,'jobs':[{'ready':True}],'workflow_sha256':'before'}
        historical=Mock(side_effect=AssertionError('Must not access a historical visual source'))
        with patch('pcm_eval.evaluation_source._prepare_one',return_value=workflow) as prep:
            result=prepare_source(workflow['inputs'],'rm-test',profile={},query=Mock(),probe=Mock(),vision_source=historical)
        historical.assert_not_called()
        self.assertEqual(prep.call_count,1)
        self.assertEqual(result['head_datasets'],{'occ':dataset,'vision':dataset})
        self.assertNotIn('vision_dataset_ref',result['inputs'])

    def test_public_card_accepts_one_dataset_and_rejects_split_inputs(self):
        form = {'model_delivery_url': INPUTS['model_delivery_url'], 'dataset_url': BASE+'occ'}
        self.assertEqual(validate_inputs(form), form)
        with self.assertRaises(ValueError):validate_inputs(INPUTS)
        with self.assertRaises(ValueError):validate_inputs({'model_delivery_url': form['model_delivery_url']})
        with self.assertRaises(ValueError):validate_inputs(dict(form,dataset_url=BASE.removesuffix('?version=')))

    def test_each_head_freezes_its_own_frames_version_and_index(self):
        profile = {'heads':['occ','vision'],'source_commit':'same', 'gpu_index':0,
            'source_root':'source','py_config':'config.py','version':'model', 'environment':{},
            'main_checkpoint':{'sha256':'main','path':'main.pth'},
            'classifier_checkpoint':{'sha256':'cls','path':'cls.pth'},
            'binding':{'code':{'evaluation_commit':'same'},'main_checkpoint':{'sha256':'main'},
                       'classifier_checkpoint':{'sha256':'cls'}},
            'dataset_classes_by_head':{'occ':'SmartChassisDatasetV7','vision':'SmartChassisDatasetV6'}}
        occ = {'version':'occ','index_sha256':'occ-index','frames_by_head':{'occ':[['a',1],['a',2]],'vision':[]},
               'mount_receipt':{'state':'MOUNT_VERIFIED'}}
        vision = {'version':'vision','index_sha256':'vision-index','frames_by_head':{'occ':[],
                  'vision':[['b',7],['b',8]]},'mount_receipt':{'state':'MOUNT_VERIFIED'}}
        plan=build_plan(profile,occ,'test',selection={'scene_count':1,'frames_per_scene':1},vision_dataset=vision)
        self.assertEqual(plan['heads'],{'occ':[['a',1]],'vision':[['b',7]]})
        self.assertEqual(plan['datasets_by_head']['vision']['index_sha256'],'vision-index')
        empty=copy.deepcopy(vision);empty['frames_by_head']['vision']=[]
        missing=build_plan(profile,occ,'empty',selection={'scene_count':1,'frames_per_scene':1},vision_dataset=empty)
        self.assertEqual(missing['vision_candidates'],[])
        self.assertEqual(missing['heads']['occ'],[['a',1]])

    def test_legacy_visual_roots_bind_original_image_to_exact_labeled_GT(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'mount';data=root/'VisionHead';scene='dataset-old/scenes/scene'
            image=data/scene/'original/7/7_f_ca_120_capture.jpg'
            image.parent.mkdir(parents=True);image.write_bytes(b'original-image')
            label=save_json(data/scene/'labeled/7/img_label.json',
                            {image.name:{'label':{'road':2}}})
            index=save_json(data/'data_link.json',{'vision_eval':{scene:{'frames':[{'frame_index':7}]}}})
            dataset={'index_path':str(index),'index_sha256':digest(index),'data_roots':[str(data)],
                     'mount_receipt':{'mount_path':str(root)}}
            result=prepare(dataset,[[scene,7]],Path(folder)/'run')
            self.assertEqual(result['state'],'READY')
            self.assertEqual(result['frames'][0]['label_sha256'],digest(label))
            label.write_text('{"other.jpg":{"label":{"road":2}}}',encoding='utf-8')
            rejected=prepare(dataset,[[scene,7]],Path(folder)/'other')
            self.assertEqual(rejected['state'],'BLOCKED')

    def test_lifecycle_journals_are_separate_and_reject_a_third_source(self):
        from pcm_eval.evaluation import EvaluationRuntime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=save_json(root/'cfg.json',{'owner':'ou_owner','agent_id':'a'})
            rt=EvaluationRuntime(root/'state',config_file=cfg)
            task=rt.open('ou_owner','feishu:oc_test:ou_owner')
            with rt.connect() as db:
                import json
                db.execute('update tasks set inputs=? where id=?',(json.dumps(INPUTS),task['id']))
            for ref in (INPUTS['occ_dataset_url'],INPUTS['vision_dataset_url']):
                rt.save_space_journal(task['id'],{'source_ref':ref},ref)
            self.assertNotEqual(rt.space_journal_path(task['id'],INPUTS['occ_dataset_url']),
                                rt.space_journal_path(task['id'],INPUTS['vision_dataset_url']))
            with self.assertRaises(ValueError):rt.space_journal_path(task['id'],BASE+'other')

    def test_report_displays_separate_versions_and_actual_scope(self):
        from tests.test_delivery_combined import CombinedTest
        fixture=CombinedTest();fixture.setUp();self.addCleanup(fixture.tearDown)
        fixture.workflow['inputs']['vision_dataset_ref']=BASE+'vision'
        fixture.workflow['head_datasets']={kind:{'version':kind,'source_ref':BASE+kind,'index_sha256':kind+'-index'}
                                           for kind in ('occ','vision')}
        execution=fixture.result['chains'][0]['execution']
        execution['executed_frames_by_head']={'occ':[['o',1],['o',2]],'vision':[['v',7]]}
        execution['head_datasets']={kind:{'version':kind,'index_sha256':kind+'-index'} for kind in ('occ','vision')}
        fixture.ref=fixture.ref.replace('<h1>开发空间模型评测</h1>',
                                       '<h1>开发空间模型评测</h1><h2>occ头</h2><h2>视觉头</h2>')
        xml,data=fixture.build()
        self.assertEqual(data['dataset_coverage']['occ']['frames'],2)
        self.assertEqual(data['dataset_coverage']['vision']['frames'],1)
        self.assertIn('评测数据集：occ；实际范围：1场景 / 2帧',xml)
        self.assertIn('评测数据集：vision；实际范围：1场景 / 1帧',xml)
        execution['head_datasets']['vision']['index_sha256']='other'
        with self.assertRaises(ValueError):fixture.build()

    def test_asset_receiver_refuses_more_than_two_frames_before_any_IO(self):
        from pcm_eval.visual_asset_input import stage
        with self.assertRaises(ValueError):
            stage({'selected_frames':[['a',1],['a',2],['a',3]],'max_frames':3})

    def test_background_RPC_is_executable_python(self):
        import ast
        from tools.evaluation_agent import RPC
        ast.parse(RPC)

    def test_environment_retry_preserves_original_event_and_never_retries_a_plan(self):
        from pcm_eval.evaluation import EvaluationRuntime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=save_json(root/'cfg.json',{'owner':'ou_owner','agent_id':'a'})
            rt=EvaluationRuntime(root/'state',config_file=cfg);session='feishu:oc_test:ou_owner'
            task=rt.open('ou_owner',session);rt.bind_message(task['id'],'ou_owner','om_card',session)
            event={'action':'roadmodel_submit','event_id':'real-fixture-event','operator_id':'ou_owner',
                   'chat_id':'oc_test','message_id':'om_card','form_value':{'model_delivery_url':INPUTS['model_delivery_url'],'dataset_url':INPUTS['occ_dataset_url']},
                   'value':{'task_id':task['id'],'nonce':task['nonce']}}
            rt.submit('ou_owner',session,event);rt.tick();row=rt.claim_materials('a','ou_owner')
            rt.complete_materials('a',task['id'],row['digest'],{'error':'SDK unavailable',
                                  'failure_kind':'preparation_environment'})
            rt.tick();self.assertEqual(rt.get(task['id'],'ou_owner')['state'],'NEEDS_REVIEW')
            repaired=rt.retry_pre_execution_materials(task['id'],'ou_owner')
            self.assertEqual(repaired['event_id'],'real-fixture-event')
            self.assertEqual(repaired['state'],'PREPARING')
            with self.assertRaises(ValueError):rt.retry_pre_execution_materials(task['id'],'ou_owner')


if __name__=='__main__':
    unittest.main()
