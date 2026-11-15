from copy import deepcopy
from pathlib import Path
import unittest
from pcm_eval.common import load_json
from pcm_eval.model_aggregate import sum_counters,aggregate_occ
from test_retained_runtime import directory

ROOT=Path(__file__).resolve().parents[1]
class CounterMergeTests(unittest.TestCase):
    def result(self,s,c,p):
        return {'metrics':[{'name':'occ.'+k,'value':v,'dimensions':{'scope':'EVAL','class_id':'01'}} for k,v in [('seen',s),('correct',c),('positive',p)]]}

    def test_unequal_class_support_is_merged_before_iou(self):
        row=sum_counters([self.result(10,10,10),self.result(90,0,90)])[0]
        self.assertEqual((100,10,100),(row['seen'],row['correct'],row['positive']))
        self.assertAlmostEqual(10/190*100,row['correct']/(row['seen']+row['positive']-row['correct'])*100)

    def test_incomplete_class_counts_are_rejected(self):
        r=self.result(10,2,3);r['metrics'].pop()
        with self.assertRaises(ValueError):sum_counters([r])

    def test_duplicate_completed_batch_is_not_double_counted(self):
        p=ROOT/'evidence/formal-v322-20261029-01/model/batch-0000-a02/normalized-result.json'
        selected=load_json(p)['provenance']['native_run']['selected_frames']
        with directory() as d:
            with self.assertRaises(ValueError):aggregate_occ([p,p],selected,d/'duplicate',run_id='duplicate')

    def test_real_completed_200_frames_merge_and_build_report(self):
        paths=[ROOT/'evidence/formal-v322-20261029-01/model'/n/'normalized-result.json' for n in ('batch-0000-a02','batch-0001-a03')]
        frames=sum([load_json(p)['provenance']['native_run']['selected_frames'] for p in paths],[])
        with directory() as d:
            result=aggregate_occ(paths,frames,d/'merged',run_id='verified-200')
            from pcm_eval.model_report import build_model_report
            report=build_model_report(result)
            self.assertEqual(200,report['coverage']['observed']['model_samples'])
            self.assertEqual(2,len(report['occ_sections']))
            self.assertEqual('full',result['provenance']['scope'])

if __name__=='__main__':unittest.main()
