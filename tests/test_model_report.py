import unittest
from copy import deepcopy
from pcm_eval.model_report import build_model_report,render_model_xml

def sample():
 metrics=[]
 for cid,s,c,p in [('01',10,5,5),('02',0,0,0)]:
  for key,value in [('seen',s),('correct',c),('positive',p),('iou',50. if s else 0.),('precision',100. if p else 0.),('recall',50. if s else 0.)]:
   metrics.append({'name':'occ.'+key,'value':value,'unit':'count' if key in ['seen','correct','positive'] else 'percent',
       'dimensions':{'scope':'EVAL','class_id':cid,'class_name':cid},'source':{'locator':'line:1'}})
 metrics.append({'name':'occ.miou','value':75.,'unit':'percent','dimensions':{}})
 return {'schema':'pcm.single-eval/v1','evaluation_type':'model_accuracy','version_name':'test','metrics':metrics,
   'execution':{'state':'succeeded'},'coverage':{},'provenance':{},'collection':{}}

class ModelReportTests(unittest.TestCase):
 def test_head_statuses_are_preserved_without_inventing_metrics(self):
  result=sample();statuses={'vision':{'state':'BLOCKED','completed_frames':0,'code':'PER_IMAGE_GT_NOT_FOUND'}}
  report=build_model_report(result,head_statuses=statuses)
  self.assertEqual(report['head_statuses'],statuses)
  self.assertEqual(report['vision_metrics'],[])
  self.assertEqual(report['all_metrics'],result['metrics'])
  statuses['vision']['state']='SUCCEEDED'
  self.assertEqual(report['head_statuses']['vision']['state'],'BLOCKED')
 def test_absent_class_does_not_inflate_present_mean(self):
  report=build_model_report(sample())
  self.assertEqual(report['occ_sections'][0]['present_miou'],50.)
  self.assertEqual(report['occ_sections'][0]['without_gt'],1)
  self.assertEqual(report['all_metrics'][-1]['value'],75.)
  self.assertEqual(report['generation']['new_inference_frames'],0)
 def test_wrong_evaluation_type_is_rejected(self):
  result=sample();result['evaluation_type']='compile_alignment'
  with self.assertRaises(ValueError):build_model_report(result)
 def test_counter_inconsistency_is_rejected(self):
  result=sample();result['metrics'][1]['value']=11
  with self.assertRaises(ValueError):build_model_report(result)
 def test_gt_scope_cannot_be_borrowed(self):
  result=sample();result['provenance']['native_run']={'selected_frames':[['a',1]]}
  with self.assertRaises(ValueError):build_model_report(result,gt_preparation={'state':'validated','frames':[{'scene':'b','frame_index':1}]})
