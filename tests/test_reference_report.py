import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from pcm_eval.common import load_json
from pcm_eval.reference_report import organize, render

ROOT = Path(__file__).resolve().parents[1]


class ReferenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_json(ROOT / 'evidence/batch-upgrade-20261029/reports-01/model-v322/generated/report.json')
        cls.compiled = load_json(ROOT / 'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/reviewed-delivery/report/report.json')
        cls.reference = load_json(ROOT / 'evidence/source-docs/release-template-review-20261029.json')

    def build(self, model=None, compiled=None):
        return organize(model or self.model, compiled or self.compiled, self.reference)

    def test_real_counts_distinguish_missing_gt_from_measured_zero(self):
        d = self.build()
        rows = {r[0]['value']: r for r in d['tables'][0]['rows']}
        self.assertEqual(rows['水泥路面'][1]['state'], 'no_gt')
        self.assertEqual(rows['草地'][1]['state'], 'measured')
        m = copy.deepcopy(self.model)
        r = m['occ_sections'][0]['classes'][0]
        r.update(iou=0.0, precision=0.0, recall=0.0, correct=0)
        self.assertEqual(self.build(model=m)['tables'][0]['rows'][0][1]['state'], 'measured')

    def test_gt_not_computed_is_not_no_gt_and_old_version_is_rejected(self):
        d = self.build()
        rows = d['tables'][3]['rows']
        self.assertTrue(all(c['state'] == 'not_computed' for r in rows[:-1] for c in r[1:]))
        c = copy.deepcopy(self.compiled)
        c['identity']['model_version'] = 'v3.2.3'
        with self.assertRaisesRegex(ValueError, 'versions differ'):
            self.build(compiled=c)

    def test_unknown_vision_support_not_fabricated_as_zero(self):
        d = self.build()
        asphalt = d['tables'][2]['rows'][1]
        self.assertEqual(asphalt[1]['value'], 'class_0 沥青')
        self.assertEqual(asphalt[2]['state'], 'missing')
        self.assertIsNone(asphalt[2]['value'])
        self.assertEqual(d['sample_scopes']['vision']['dataset_version'], '26-08-14-0')
        self.assertEqual(d['sample_scopes']['occ']['dataset_version'], '26-10-20-1')

    def test_exact_original_table_inventory_without_diagnostic_chapters(self):
        d = self.build()
        tree = ET.fromstring('<root>' + render(d, 'data.json') + '</root>')
        self.assertEqual(len(list(tree.iter('table'))), 5)
        self.assertEqual([''.join(n.itertext()) for n in tree.findall('h1')], ['模型精度评测', '编译精度评测'])
        original = ET.fromstring('<root>' + self.reference['data']['document']['content'] + '</root>')
        for index in (1, 2, 3, 4):
            labels = [''.join(r[0].itertext()) for r in list(original.iter('table'))[index].findall('.//tr')][1:]
            self.assertEqual(labels, [r[0]['value'] for r in d['tables'][index]['rows']])
        self.assertEqual(d['generation']['new_inference_frames'], 0)
        self.assertIsNone(d['generation']['combined_score'])

    def test_validated_gt_summary_maps_metric_ids_and_keeps_measured_zero(self):
        c = load_json(ROOT / 'evidence/coverage-supplement-20261029/compile-report/report.json')
        d = self.build(compiled=c)
        rows = {r[0]['value']: r for r in d['tables'][3]['rows']}
        for label, key in [('可行驶区域iou', 'drivable_iou'), ('present_miou', 'present_miou')]:
            source = next(r for r in c['ground_truth']['summary'] if r['metric_id'] == key)
            self.assertEqual(rows[label][1]['value'], source['onnx'])
            self.assertEqual(rows[label][2]['value'], source['board'])
        self.assertEqual(rows['水面'][1]['state'], 'no_gt')
        self.assertEqual(rows['井盖'][1]['state'], 'measured')
        self.assertEqual(rows['井盖'][1]['value'], 0.0)
        self.assertNotIn('本次GT未计算', render(d, 'data.json'))

    def test_coverage_vision_has_all_fourteen_original_subclasses(self):
        from pcm_eval.model_report import build_model_report
        extra = load_json(ROOT / 'evidence/coverage-supplement-20261029/vision-collected/normalized-result.json')
        model = build_model_report(self.model['source_result'], binding=self.model['binding'],
                                   head_results=[extra])
        d = self.build(model=model)
        self.assertEqual(len(d['tables'][2]['rows']), 14)
        for row in d['tables'][2]['rows']:
            self.assertTrue(all(c['state'] == 'measured' for c in row))
            self.assertGreater(row[2]['value'], 0)


if __name__ == '__main__':
    unittest.main()
