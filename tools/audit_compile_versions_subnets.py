"""Audit supplied versions and subnet tensor contracts, without executing payloads."""
import json
from pathlib import Path
import re
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/compile-subnet-review-20261029'


def port(path):
    name=Path(path).name
    match=re.fullmatch(r'(.+)\.([0-9]+(?:x[0-9]+)*)\.(float16|float32|int32|int64|uint8)\.bin',name)
    if not match:raise ValueError('Unknown tensor contract: '+name)
    return {'tensor':match[1],'shape':[int(x) for x in match[2].split('x')], 'dtype':match[3], 'filename':name}


def main():
    OUT.mkdir(exist_ok=True)
    old=load_json(ROOT/'evidence/actual-runs/pcm-v7-multiscene-001/collected/request.json')
    old_receipt=load_json(ROOT/'evidence/actual-runs/pcm-v7-multiscene-001/collected/execution_receipt.json')
    stages=[]
    for item in load_json(ROOT/'evidence/feishu-entry-20261029/v322-inventory.json')['archives']:
        p=Path(item['path']);info=item['metadata']['./model_info.json']
        stage=int(re.search(r'_sub([0-4])_',p.name)[1])
        inputs=[port(x['data_paths'][0]) for x in info['input']]
        outputs=[{**port(x['golden_paths'][0]),'data_path':x['data_path']} for x in info['output']]
        stages.append({'stage':stage,'archive':str(p),'sha256':digest(p),'elf_path':info['elf_path'],
                       'inputs':inputs,'outputs':outputs})
    stages.sort(key=lambda x:x['stage'])
    producers={}
    for stage in stages:
        for output in stage['outputs']:
            producers.setdefault(output['tensor'],[]).append((stage['stage'],output))
    external=[];edges=[]
    for stage in stages:
        for p in stage['inputs']:
            candidates=[x for x in producers.get(p['tensor'],[]) if x[0]<stage['stage']]
            if len(candidates)==1:
                s,o=candidates[0]
                edges.append({'from':s,'to':stage['stage'],'tensor':p['tensor'],
                    'same_contract':all(p[k]==o[k] for k in ('shape','dtype'))})
            else:external.append({'stage':stage['stage'],**p,'producer_candidates':len(candidates)})
    value={'previous':{'version':old['version_name'],'model':old['model'],
        'receipt_version':old_receipt['version_name'],'receipt_onnx_sha256':old_receipt['onnx_sha256'],
        'receipt_packup_sha256':old_receipt['packup_sha256'],'observed_frames':old_receipt['requested_frames']},
        'current':{'version':'v3.2.2','onnx_sha256':digest(ROOT/'evidence/model-evaluation/model-reference-20261027/v322.onnx'),
                   'stages':stages},'edges':edges,'external_inputs':external,
        'new_inference_frames':0,'dataset_downloads':0}
    save_json(OUT/'version-and-port-audit.json',value,overwrite=(OUT/'version-and-port-audit.json').exists())
    print(json.dumps({'previous':value['previous'],'current_onnx_sha256':value['current']['onnx_sha256'],
       'stages':[{'stage':s['stage'],'inputs':[p['filename'] for p in s['inputs']],
                  'outputs':[p['filename'] for p in s['outputs']]} for s in stages],
       'edges':len(edges),'contract_mismatches':[e for e in edges if not e['same_contract']],
       'external_inputs':external},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
