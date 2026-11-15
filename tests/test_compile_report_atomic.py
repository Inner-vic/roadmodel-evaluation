import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from pcm_eval.compile_report import build_compile_report, render_sections
from pcm_eval.common import load_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evidence/actual-runs/pcm-v7-multiscene-001'


class AtomicCompileReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=load_json(RUN/'gt-adapted-20261027/result.json')
        cls.cosine=load_json(RUN/'collected/cosine_summary.json')
        cls.backend={b:load_json(RUN/f'gt-adapted-20261027/metrics_{b}.json') for b in ['onnx','board']}
        cls.request=load_json(ROOT/'examples/reference_compile_multiscene.json')

    def build(self,**changes):
        return build_compile_report(changes.get('result',copy.deepcopy(self.result)),
                                    changes.get('cosine',copy.deepcopy(self.cosine)),
                                    backend_metrics=changes.get('backend',copy.deepcopy(self.backend)),
                                    request=changes.get('request',copy.deepcopy(self.request)))

    def test_real_existing_inventory_is_preserved_without_mixed_score_or_release_template(self):
        report=self.build()
        self.assertEqual(report['schema'],'roadmodel.compile-single-report/v3')
        self.assertEqual(len(report['tensor_alignment']['records']),360)
        self.assertEqual(len(report['tensor_alignment']['outputs']),18)
        self.assertEqual(len(report['ground_truth']['classes']),24)
        self.assertEqual(len(report['ground_truth']['counts']),48)
        self.assertEqual(report['coverage']['classes_with_gt'],8)
        self.assertEqual(report['coverage']['classes_without_gt'],16)
        self.assertEqual(report['generation']['new_inference_frames'],0)
        self.assertEqual(report['coverage']['finite_output_checks'],720)
        self.assertEqual(len(report['source_normalized_result']['metrics']),108)
        self.assertNotIn('template',report)
        self.assertNotIn('comparison',report)
        self.assertNotIn('total_score',report)
        self.assertEqual(report['quality_assessment']['state'],'not_assessed')

    def test_mae_is_mean_of_frames_not_element_weighted_and_counts_are_derived_explicitly(self):
        report=self.build()
        first=report['tensor_alignment']['outputs'][0]
        raw=[r['mean_abs_diff'] for f in self.cosine['frames'] for r in f['outputs'] if r['onnx_output_name']==first['name']]
        self.assertAlmostEqual(first['mean_frame_mae'],sum(raw)/20)
        self.assertAlmostEqual(first['max_abs_diff'],max(r['max_abs_diff'] for f in self.cosine['frames'] for r in f['outputs'] if r['onnx_output_name']==first['name']))
        for r in report['ground_truth']['counts']:
            self.assertEqual(r['fp'],r['positive']-r['correct'])
            self.assertEqual(r['fn'],r['seen']-r['correct'])

    def test_native_sections_keep_tensor_and_gt_separate_and_all_classes_covered(self):
        sections=render_sections(self.build())
        self.assertEqual(len(sections),8)
        xml=''.join(s['xml'] for s in sections)
        self.assertLess(len(xml),60000,'Raw validation records must stay in attachments')
        ET.fromstring('<root>'+xml+'</root>')
        self.assertIn('张量一致性',sections[3]['xml'])
        self.assertIn('相对GT的表现',sections[4]['xml'])
        for row in self.backend['board']['class_support']:
            self.assertIn(row['name'],sections[4]['xml'])
        self.assertIn('0.942701',xml)
        self.assertIn('47.83',sections[4]['xml'])
        self.assertNotIn('PCM发版',xml)

    def test_changed_order_duplicate_output_and_silent_missing_output_fail(self):
        for mutate in [lambda c:c['frames'].reverse(),
                       lambda c:c['frames'][0]['outputs'].append(c['frames'][0]['outputs'][0]),
                       lambda c:c['frames'][0]['outputs'].pop()]:
            cosine=copy.deepcopy(self.cosine);mutate(cosine)
            with self.assertRaises(ValueError):self.build(cosine=cosine)

    def test_gt_scope_or_counts_cannot_be_substituted(self):
        result=copy.deepcopy(self.result)
        result['provenance']['gt_adaptation']['selected_frames'][0][1]=999
        with self.assertRaisesRegex(ValueError,'GT result scope'):self.build(result=result)
        backend=copy.deepcopy(self.backend)
        backend['onnx']['class_support'][0]['correct']+=1000
        with self.assertRaisesRegex(ValueError,'IoU differs'):self.build(backend=backend)

    def test_unknown_outputs_remain_visible(self):
        cosine=copy.deepcopy(self.cosine)
        old=next(iter(cosine['tensors']));cosine['tensors']['new_output']=cosine['tensors'].pop(old)
        for frame in cosine['frames']:
            for r in frame['outputs']:
                if r['onnx_output_name']==old:r['onnx_output_name']='new_output'
        report=self.build(cosine=cosine)
        self.assertTrue(any(r['name']=='new_output' and r['group']=='其他输出' for r in report['tensor_alignment']['outputs']))

    def test_tensor_only_report_does_not_fill_gt_zeroes(self):
        result=copy.deepcopy(self.result)
        result['provenance'].pop('gt_adaptation')
        report=self.build(result=result,backend={})
        self.assertEqual(report['ground_truth']['summary'],[])
        xml=next(s['xml'] for s in render_sections(report) if s['id']=='gt')
        self.assertIn('未评测',xml)
        self.assertNotIn('<table>',xml)

    def test_nan_and_unrelated_model_request_fail(self):
        cosine=copy.deepcopy(self.cosine)
        cosine['frames'][0]['outputs'][0]['cosine']=float('nan')
        with self.assertRaises(ValueError):self.build(cosine=cosine)
        request=copy.deepcopy(self.request);request['model']['onnx_sha256']='different'
        with self.assertRaises(ValueError):self.build(request=request)

if __name__=='__main__':unittest.main()
