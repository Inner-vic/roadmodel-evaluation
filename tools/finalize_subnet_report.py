"""Adapt and report retained successful subnet outputs; never invokes inference."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,digest
from pcm_eval.runner import collect_remote
from pcm_eval.compile_report import write_compile_report
from pcm_eval.publication import publish_report
from prepare_subnet_remote import remote,REMOTE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/compile-subnet-review-20261029'
ADAPTER='/mnt/data/developer/pcm-roadmodel-eval/subnet-adapter-20261029'
RUN='pcm-v322-subnets-one-02'


if __name__=='__main__':
    # Freeze the exact adapter that performed the retained run before deploying postprocessing updates.
    code=r'''import pathlib,shutil,json,hashlib
base=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval');run=base/'subnet-runs/pcm-v322-subnets-one-02';adapter=base/'subnet-adapter-20261029'
for src,dest in [('subnet_chain_remote.py','executed-adapter.py'),('subnet_contract.py','executed-contract.py')]:
 p=run/dest
 if not p.exists():shutil.copyfile(adapter/src,p)
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'executed-adapter.py',run/'executed-contract.py']}))'''
    frozen=json.loads(remote(code))
    if (OUT/'retained-executed-code.json').exists():assert load_json(OUT/'retained-executed-code.json')==frozen
    else:save_json(OUT/'retained-executed-code.json',frozen)
    src=ROOT/'tools/subnet_chain_remote.py'
    subprocess.run(['scp.exe','-q','-o','BatchMode=yes',src.name,'ubuntu-workstation:'+ADAPTER+'/subnet_chain_remote.py'],cwd=src.parent,check=True,timeout=40)
    code=r'''import pathlib,sys,json
sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/subnet-adapter-20261029')
import subnet_chain_remote as adapter
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/pcm-v322-subnets-one-02')
record=json.loads((root/'execution.json').read_text());assert record['state']=='SUCCEEDED' and record['release']['state']=='RELEASED'
adapter.write_compatible_results(root,record)
print(json.dumps({'state':'collected_existing_results','new_inference_frames':0}))'''
    print(remote(code))
    inputs=load_json(OUT/RUN/'input-reuse.json')
    audit=load_json(OUT/'version-and-port-audit.json')
    request={'schema':'pcm.eval-request/v1','kind':'compile_alignment','version_name':'v3.2.2',
        'source':{'model_reference_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
                  'note':'One-frame five-original-subnet chain validation; no ELF merging or full evaluation'},
        'execution':{'transport':'ssh','ssh_alias':'ubuntu-workstation','python':'/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python',
            'repo_dir':ADAPTER,'output_root':'/mnt/data/developer/pcm-roadmodel-eval/subnet-runs',
            'adapter':'subnet_chain_workstation','driver_script':ADAPTER+'/subnet_chain_remote.py','driver_sha256':digest(src),
            'contract_script':ADAPTER+'/subnet_contract.py','contract_sha256':digest(ROOT/'pcm_eval/subnets.py'),'env':{}},
        'dataset':{'class_name':'SmartChassisDatasetV7','path':'/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1',
            'data_link_file':'/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1/data_link_26-10-20-1_train859_eval135_v7.json',
            'data_link_sha256':'e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c',
            'input_dir':'/mnt/data/developer/pcm-roadmodel-eval/runs/pcm-v7-multiscene-001/frame_000000','cached_inputs':inputs['inputs']},
        'model':{'onnx':REMOTE+'/v322.onnx','onnx_sha256':audit['current']['onnx_sha256'],
            'source_audit':REMOTE+'/source-audit.json','source_audit_sha256':digest(OUT/'version-and-port-audit.json'),
            'subnet_archives':[{'stage':x['stage'],'path':REMOTE+'/'+Path(x['archive']).name,'sha256':x['sha256']} for x in audit['current']['stages']]},
        'evaluation':{'frame_fraction':1.0,'max_frames':1,'expected_frames':1,'compute_gt_metrics':False,
            'selection':{'frames':[['2026-06-30_18-39-31_HLX34B167T1300224',18]]}}}
    save_json(ROOT/'examples/compile_v322_subnets_small.json',request,overwrite=(ROOT/'examples/compile_v322_subnets_small.json').exists())
    target=OUT/RUN/'report-delivery-02'
    result=collect_remote('ubuntu-workstation','/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/'+RUN,
                          'compile_alignment',target/'collected',version_name='v3.2.2',expected_frames=1)
    assert result['collection']['status']=='complete',result['collection']
    report=write_compile_report(target/'report/report.xml',result_file=target/'collected/result.json',
                               request_file=ROOT/'examples/compile_v322_subnets_small.json',supplemental_dir=target/'collected')
    publication=publish_report(report['xml'],target/'publication')
    receipt={'state':'SUCCEEDED','run_id':RUN,'report':report,'publication':publication,'new_inference_frames':0,
        'measured_frames_reused':1,'gt_metrics':'not_computed_for_this_tensor_adapter_validation',
        'previous_v323_report_unchanged':True}
    save_json(OUT/'report-delivery.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
