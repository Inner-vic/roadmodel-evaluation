"""Bind the verified subnet adapter for two-link intake without dispatching work."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,digest
from pcm_eval.planning import build_plan,verify_plan,ssh_command
from pcm_eval.runner import PREFLIGHT
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/compile-subnet-review-20261029'
request=load_json(ROOT/'examples/compile_v322_subnets_small.json')
request['execution']['driver_sha256']=digest(ROOT/'tools/subnet_chain_remote.py')
request['execution']['contract_sha256']=digest(ROOT/'pcm_eval/subnets.py')
request['source']['reference_preparation']='原convert_fp32_to_fp16_keep_fp8.py；交付源ONNX与派生FP16参考分别记录SHA256'
save_json(ROOT/'examples/compile_v322_subnets_small.json',request,overwrite=True)
catalog=load_json(ROOT/'configs/two_link_catalog.json')
spec=catalog['models'][0]['chains']['compile_alignment']
spec.clear();spec.update(version='v3.2.2',dataset_id='smartchassisdata-v3/26-10-20-1',heads=['tensor_alignment'],
    request_file='examples/compile_v322_subnets_small.json',request_sha256=digest(ROOT/'examples/compile_v322_subnets_small.json'),
    shared_bench_guard_verified=True,
    verification_receipt='evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/execution.json',
    warnings=['本次仅验证一帧分网串接和18个张量输出；未补算GT，不作发版质量判定。'])
save_json(ROOT/'evidence/feishu-entry-20261029/catalog.pending.json',catalog)
plan=build_plan(request,'rm-subnet-contract-check');verify_plan(plan)
save_json(OUT/'runtime-plan.json',plan)
first=plan['steps'][0]
identity=plan['input_identity']
pre=[first['argv'][0],'-c',PREFLIGHT,'new',plan['output_dir'],json.dumps(identity),*plan['input_paths']]
p=subprocess.run(ssh_command(plan['ssh_alias'],pre),capture_output=True,timeout=90)
(OUT/'runtime-preflight.log').write_bytes(p.stdout+p.stderr)
if p.returncode:raise RuntimeError('Runtime plan preflight failed; catalog remains pending, no inference dispatched')
p=subprocess.run(ssh_command(plan['ssh_alias'],first['argv']+['--validate-request-only'],first['cwd']),capture_output=True,timeout=30)
(OUT/'runtime-parser-validation.json').write_bytes(p.stdout)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1000:])
save_json(OUT/'runtime-validation.json',{'state':'PASSED','plan_hash':plan['plan_sha256'],'model_version':'v3.2.2',
    'expected_frames':1,'preflight_exit_code':0,'request_parser':json.loads(p.stdout),'new_inference_frames':0})
print(json.dumps({'state':'VERIFIED_PENDING_SERVICE_DRAIN','new_inference_frames':0,'compile_adapter':'five_subnet_chain'}))
