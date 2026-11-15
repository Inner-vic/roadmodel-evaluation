import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from pcm_eval.common import Result
from pcm_eval.parsers import parse_model_log
from pcm_eval.model_heads import trajectory_unavailable
from pcm_eval.model_report import build_model_report

def result(kind,frames):
    return {'schema':'pcm.single-eval/v1','evaluation_type':'model_accuracy','version_name':'v',
        'execution':{'state':'succeeded'},'collection':{'status':'complete'},'coverage':{'observed':{'model_samples':len(frames)}},
        'provenance':{'native_run':{'kind':kind,'run_id':kind,'checkpoint_sha256':{'a':'hash'},'source_commit':'source','selected_frames':frames}},
        'metrics':[{'name':kind+'.accuracy','value':50.,'unit':'percent','dimensions':{}}]}

class ModelHeadTests(unittest.TestCase):
    def test_scopes_remain_independent_and_weights_must_match(self):
        base=result('occ',[['occ',1],['occ',2]]);vision=result('vision',[['vision',7]])
        report=build_model_report(base,head_results=[vision])
        self.assertEqual(report['coverage']['observed']['model_samples'],2)
        self.assertEqual(report['head_results']['vision']['coverage']['observed']['model_samples'],1)
        vision['provenance']['native_run']['checkpoint_sha256']['a']='other'
        with self.assertRaisesRegex(ValueError,'same model'):build_model_report(base,head_results=[vision])
    def test_scope_count_mismatch_rejected(self):
        base=result('occ',[['o',1]]);vision=result('vision',[['v',1]])
        vision['coverage']['observed']['model_samples']=4
        with self.assertRaisesRegex(ValueError,'sample scope'):build_model_report(base,head_results=[vision])
    def test_empty_event_metrics_are_unavailable_not_zero_scores(self):
        data={'metrics':{'front_gt_events':0,'front_pred_events':0,'front_matched_events':0,
            'front_mae_cm':0.,'front_miss_rate':0.,'front_false_event_rate':0.,'front_p90_cm':None,
            'boundary_matched_wheels':0,'boundary_mae_cm':0.},'raw_counts':{}}
        unavailable=trajectory_unavailable(data)
        self.assertEqual(unavailable['front_mae_cm'],'no_matched_events')
        self.assertEqual(unavailable['front_miss_rate'],'no_gt_events')
        self.assertNotIn('front_gt_events',unavailable)
    def test_native_vision_counts_are_not_map(self):
        text='Vision Head mAP Results\n  road_scene_type: 0.80\nTotal samples used: 4\n(bump_type): 2/4 = 50.00%\n'
        with patch.object(Path,'read_text',return_value=text),patch.object(Result,'artifact'):
            root=Path.cwd();log=root/'eval.log'
            r=Result('model_accuracy','v',root,4);parse_model_log(log,r)
            metrics=r.finish()['metrics']
            self.assertFalse(any(m.get('dimensions',{}).get('task')=='used' for m in metrics))
            self.assertEqual(next(m['value'] for m in metrics if m['name']=='vision.correct'),2)
            self.assertEqual(next(m['value'] for m in metrics if m['name']=='vision.accuracy'),50.)
