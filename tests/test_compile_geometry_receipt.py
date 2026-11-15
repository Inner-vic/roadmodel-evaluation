from copy import deepcopy
from pathlib import Path
import unittest
from pcm_eval.common import load_json
from pcm_eval.compile_report import render_sections

ROOT=Path(__file__).resolve().parents[1]
class GeometryReceiptTests(unittest.TestCase):
    def report(self):return load_json(ROOT/'evidence/coverage-supplement-20261029/compile-report/report.json')
    def test_voxel_receipt_without_height_field_check_is_reported_without_invention(self):
        r=self.report()
        for f in r['ground_truth']['receipt']['frames']:f.pop('geometry_check',None)
        text=next(s['xml'] for s in render_sections(r) if s['id']=='definitions')
        self.assertIn('未包含逐帧height_field交叉核验比例',text)
        self.assertNotIn('比例最低为',text)

    def test_partial_height_field_proof_is_not_described_as_all_frames(self):
        r=self.report();r['ground_truth']['receipt']['frames'][0].pop('geometry_check',None)
        text=next(s['xml'] for s in render_sections(r) if s['id']=='definitions')
        self.assertIn('已提供交叉核验的',text)

if __name__=='__main__':unittest.main()
