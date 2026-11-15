"""Bind the already audited v3.2.2 adapters to a fresh online material inventory.

This does not invent support for new architectures or claim a small-scope
adapter covers a complete validation set.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import digest, load_json, save_json
from pcm_eval.delivery_source import inspect_record
from pcm_eval.delivery_workflow import material_identity


def build(materials):
    baseline = inspect_record(load_json(ROOT/'evidence/input-contract-review-20261027/release-record.json'), materials['source_url'])
    identity, known = material_identity(materials), material_identity(baseline)
    def without_tokens(value):
        return {k: [{a: b for a, b in r.items() if a != 'token'} for r in v] if isinstance(v, list) else v
                for k, v in value.items()}
    if without_tokens(identity) != without_tokens(known):
        raise ValueError('线上交付与已审计v3.2.2材料不一致，需要重新核验实际产物')
    def record(path):
        return {'path': path, 'sha256': digest(ROOT/path)}
    old = load_json(ROOT/'configs/two_link_catalog.json')['datasets'][0]
    model = {'materials': identity, 'chains': {
        'model_accuracy': {'dataset_id': old['id'], 'heads': ['occ', 'vision'], 'validation_heads': ['occ'],
            'request': record('examples/two_link_model_small.json'), 'validation_only': True,
            'binding': record('evidence/model-evaluation/model-reference-20261027/model-input-binding.json'),
            'gt_preparation': record('evidence/model-evaluation/model-reference-20261027/native-gt-preparation.json')},
        'compile_alignment': {'dataset_id': old['id'], 'heads': [],
            'request': record('examples/compile_v322_subnets_small.json'), 'validation_only': True}}}
    main = old['workstation_cache']
    dataset = {'id': old['id'], 'version': '26-10-20-1',
               'aliases': [old['url'],
                   'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1',
                   main, '工作站:'+main],
               'index': record('evidence/requested-dataset/data_link_26-10-20-1_train859_eval135_v7.json'),
               'gt_heads': ['occ'], 'locations': {'workstation': main}}
    return {'schema': 'roadmodel.delivery-catalog/v1', 'models': [model], 'datasets': [dataset]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--materials', required=True)
    p.add_argument('--output', default=str(ROOT/'configs/delivery_catalog.json'))
    args = p.parse_args()
    value = build(load_json(args.materials))
    save_json(args.output, value)
    print('Verified delivery catalog written; no inference or production deployment.')
