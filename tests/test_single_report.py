import copy
import unittest
import xml.etree.ElementTree as ET

from pcm_eval.common import load_json
from pcm_eval.single_report import DEFAULT_TEMPLATE, build_single_report, render_xml


class SingleReportTests(unittest.TestCase):
    def result(self):
        contract = load_json(DEFAULT_TEMPLATE)
        metrics = [{'name': 'tensor.frame_mean_cosine.mean', 'value': 0.9969402983784678,
                    'unit': 'ratio', 'dimensions': {}, 'source': {'path': 'fixture', 'locator': '/mean'}}]
        for row in contract['compile_alignment']['outputs']:
            for key, value in [('mean_cosine', 0.9940238535), ('min_cosine', 0.9427010417)]:
                metrics.append({'name': 'tensor.'+key, 'value': value, 'unit': 'ratio',
                                'dimensions': {'output': row['key']}, 'source': {'path': 'fixture', 'locator': '/tensor'}})
        return {'schema': 'pcm.single-eval/v1', 'evaluation_type': 'compile_alignment', 'version_name': 'fixture-v1',
                'metrics': metrics, 'execution': {'state': 'succeeded'}, 'collection': {'status': 'complete'},
                'coverage': {'observed': {'tensor_comparison': 20},
                             'tensor_frames': {x['key']:20 for x in contract['compile_alignment']['outputs']}},
                'provenance': {'input_validation': {'ground_truth': {'state': 'not_requested'}}}}

    def test_compile_report_matches_template_order_semantics_and_percent_units(self):
        report = build_single_report('compile_alignment', self.result())
        outputs = report['tables'][1]['rows']
        self.assertEqual(len(outputs), 18)
        self.assertEqual(outputs[0]['meaning'], 'means3D')
        semantics = next(x for x in outputs if x['key']=='/Concat_28_output_0')
        self.assertEqual(semantics['meaning'], 'semantics')
        self.assertAlmostEqual(semantics['min']['value'], 94.27010417)
        xml = render_xml(report)
        ET.fromstring('<root>'+xml+'</root>')
        self.assertIn('99.69', xml)
        self.assertIn('94.27', xml)
        self.assertNotIn('0.942701', xml)

    def test_unrequested_gt_stays_empty_and_no_historical_scores_are_copied(self):
        report = build_single_report('compile_alignment', self.result())
        self.assertEqual(report['report_status'], 'partial')
        for row in report['tables'][0]['rows'][:-1]:
            self.assertIsNone(row['onnx']['value'])
            self.assertIsNone(row['board']['value'])
        self.assertEqual(report['filled_metric_cells'], 37)
        self.assertEqual(report['comparison']['state'], 'not_requested')

    def test_model_template_preserves_occ_vision_and_major_rows_without_results(self):
        report = build_single_report('model_accuracy')
        self.assertEqual(report['report_status'], 'template_only')
        self.assertEqual(report['filled_metric_cells'], 0)
        self.assertEqual([len(t['rows']) for t in report['tables']], [12,9,14])
        xml = render_xml(report)
        for label in ('occ头','轨迹头','视觉头','参与 mAP 的子类明细'):
            self.assertIn(label, xml)
        self.assertNotIn('0.9983', xml)
        self.assertNotIn('编译精度', xml)

    def test_two_evaluation_types_cannot_be_substituted_for_two_comparison_runs(self):
        with self.assertRaises(ValueError):
            build_single_report('model_accuracy', self.result())

    def test_nonempty_iou_is_not_mislabeled_as_drivable_iou(self):
        result = self.result()
        result['metrics'].append({'name':'occ_iou','value':88.8,'unit':'percent','dimensions':{'backend':'board'}})
        report = build_single_report('compile_alignment', result)
        drivable = next(r for r in report['tables'][0]['rows'] if r['key']=='drivable')
        self.assertIsNone(drivable['board']['value'])

    def test_invalid_provenance_and_partial_collection_do_not_become_complete(self):
        result = self.result()
        for m in result['metrics']: m['validity']='invalid_output_provenance'
        report = build_single_report('compile_alignment', result)
        self.assertEqual(report['filled_metric_cells'], 0)
        self.assertNotIn('99.69', render_xml(report))

    def test_source_location_and_exact_values_are_retained_for_future_version_comparison(self):
        report = build_single_report('compile_alignment', self.result())
        cell = report['tables'][1]['rows'][0]['min']
        self.assertEqual(cell['source'], {'path':'fixture','locator':'/tensor'})
        self.assertEqual(cell['source_unit'], 'ratio')


if __name__ == '__main__':
    unittest.main()
