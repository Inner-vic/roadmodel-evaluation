"""Freeze current known materials; never bind the v3.2.3 compiler to v3.2.2."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import digest, save_json, load_json

ROOT = Path(__file__).resolve().parents[1]
model_url = 'https://li.feishu.cn/record/PUBREF000000000000000000001'
dataset_url = 'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/components/detail/project-bf8f6b06-f0e7-4d72-aa58-0c2731157df1/record/81e922d6-ee86-483c-9366-d5e3f6aca940'
request = 'examples/two_link_model_small.json'
native = load_json(ROOT / 'examples/model_request.json')
native['execution']['ssh_options'] = {'port':31017,'identity_file':'C:/Users/developer/.ssh/bmad_roadmodel_ed25519'}
native['evaluation']['selected_frames'] = load_json(ROOT/'evidence/model-evaluation/model-reference-20261027/model-input-binding.json')['dataset']['selected_frames']
if __name__ == '__main__':
    save_json(ROOT / request,native,overwrite=(ROOT/request).exists())
value = {'schema': 'roadmodel.two-link-catalog/v1', 'models': [{
    'url': model_url, 'version': 'v3.2.2',
    'evidence': 'evidence/input-contract-review-20261027/release-record.json',
    'chains': {
        'model_accuracy': {'version': 'v3.2.2', 'dataset_id': 'smartchassisdata-v3/26-10-20-1',
            'heads': ['occ'], 'request_file': request, 'request_sha256': digest(ROOT / request),
            'binding_file':'evidence/model-evaluation/model-reference-20261027/model-input-binding.json',
            'gt_preparation_file':'evidence/model-evaluation/model-reference-20261027/native-gt-preparation.json',
            'warnings': ['本数据集现有已核验小场景只绑定OCC；视觉和轨迹原验证使用其他GT版本，不自动混入本次。']},
        'compile_alignment': {'version': 'v3.2.2', 'dataset_id': 'smartchassisdata-v3/26-10-20-1',
            'heads': ['tensor_alignment'], 'issues': ['五分网适配或共享台架互斥尚无验证收据，禁止代用其他版本产物。']}
    }}], 'datasets': [{'url': dataset_url, 'id': 'smartchassisdata-v3/26-10-20-1',
        'class_name': 'SmartChassisDatasetV7',
        'data_link_file': 'data_link_26-10-20-1_train859_eval135_v7.json',
        'data_link_sha256': 'e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c',
        'workstation_cache': '/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1',
        'receipt': 'evidence/requested-dataset/compile-tensor-sync-receipt.json'}]}
subnet_request=ROOT/'examples/compile_v322_subnets_small.json'
subnet_receipt=ROOT/'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/execution.json'
if subnet_request.is_file() and subnet_receipt.is_file():
    measured=load_json(subnet_receipt)
    if measured.get('state')=='SUCCEEDED' and measured.get('public_outputs')==18 and measured.get('release',{}).get('state')=='RELEASED':
        value['models'][0]['chains']['compile_alignment']={'version':'v3.2.2','dataset_id':'smartchassisdata-v3/26-10-20-1',
            'heads':['tensor_alignment'],'request_file':'examples/compile_v322_subnets_small.json','request_sha256':digest(subnet_request),
            'shared_bench_guard_verified':True,'warnings':['本次仅验证一帧分网串接和18个张量输出；未补算GT，不作发版质量判定。']}
if __name__ == '__main__':
    save_json(ROOT / 'configs/two_link_catalog.json', value,overwrite=(ROOT/'configs/two_link_catalog.json').exists())
    save_json(ROOT / 'examples/two_link_submission.json', {'model_delivery_url': model_url, 'dataset_url': dataset_url},overwrite=(ROOT/'examples/two_link_submission.json').exists())
    print('Two-link catalog created; no evaluation executed.')
