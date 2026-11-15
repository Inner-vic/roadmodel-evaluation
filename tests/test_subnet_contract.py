import copy
from pathlib import Path
import unittest
from pcm_eval.common import load_json
from pcm_eval.subnets import chain_contract,stage_info
from pcm_eval.planning import build_plan,verify_plan
from pcm_eval.compile_report import build_compile_report

ROOT=Path(__file__).resolve().parents[1]


class SubnetContractTests(unittest.TestCase):
    def infos(self):
        audit=load_json(ROOT/'evidence/feishu-entry-20261029/v322-inventory.json')
        import re
        return {int(re.search('_sub([0-4])_',x['path'])[1]):x['metadata']['./model_info.json'] for x in audit['archives']}

    def test_actual_five_packages_form_32_valid_edges_and_11_external_inputs(self):
        value=chain_contract(self.infos())
        self.assertEqual(32,len(value['edges']))
        self.assertEqual(11,len(value['external_inputs']))
        self.assertEqual([15,15,9,8,2],[len(x['outputs']) for x in value['stages']])
        self.assertEqual(3,len([p for p in value['external_inputs'] if p['external_filename'].startswith('@lidar_backbone')]))

    def test_shape_mismatch_is_rejected_before_board_execution(self):
        infos=self.infos()
        infos[2]['input'][5]['data_paths'][0]=infos[2]['input'][5]['data_paths'][0].replace('96x200','96x201')
        with self.assertRaises(ValueError):chain_contract(infos)

    def test_sub1_voxel_order_comes_from_package_not_whole_network_order(self):
        infos=self.infos();contract=chain_contract(infos)
        result=stage_info(infos[1],contract['stages'][1],'/userdata/benchEval/models/test-chain')
        self.assertIn('output_1.14400x3.int32.bin',result['input'][0]['data_paths'][0])
        self.assertIn('output_0.14400x32x4.float16.bin',result['input'][1]['data_paths'][0])
        self.assertEqual(1,result['infer_num']);self.assertFalse(result['do_compare'])

    def test_internal_ports_never_use_shipped_golden_data(self):
        infos=self.infos();contract=chain_contract(infos)
        result=stage_info(infos[2],contract['stages'][2],'/userdata/benchEval/models/test-chain')
        self.assertIn('/sub0/export/',result['input'][5]['data_paths'][0])
        self.assertTrue(all(not x['golden_paths'] for x in result['output']))

    def test_runtime_plan_uses_five_archives_without_fake_merged_package(self):
        request=load_json(ROOT/'examples/compile_v322_subnets_small.json')
        plan=build_plan(request,'rm-subnet-unit');verify_plan(plan)
        self.assertTrue(plan['ready'])
        self.assertIn('--request-json',plan['steps'][0]['argv'])
        self.assertEqual(1,plan['scope']['max_frames'])
        self.assertNotIn('packup_tar',request['model'])
        for bad in (20,True):
            changed=copy.deepcopy(request);changed['evaluation']['max_frames']=bad
            with self.assertRaises(ValueError):build_plan(changed,'rm-subnet-unit')

    def test_report_preserves_source_reference_and_all_five_archive_identities(self):
        base=ROOT/'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/report-delivery-02/collected'
        result=load_json(base/'result.json');cosine=load_json(base/'cosine_summary.json')
        request=load_json(ROOT/'examples/compile_v322_subnets_small.json')
        report=build_compile_report(result,cosine,request=request)
        self.assertEqual(18,report['coverage']['tensor_outputs'])
        self.assertEqual('not_available',report['ground_truth']['state'])
        changed=copy.deepcopy(request);changed['model']['subnet_archives'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):build_compile_report(result,cosine,request=changed)
