"""Seal the inactive candidate after exact GT and original report validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE = r'''import pathlib,json,hashlib,sys
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-memory-gt-20261030-r1')
deployment=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-memory-gt-20261030-r1')
validation=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-memory-gt-20261030-r1')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
sys.path.insert(0,str(release));from pcm_eval.common import digest,load_json,save_json,utc_now
staging=load_json(deployment/'staging.json')
for name,h in staging['source'].items():assert digest(release/name)==h,'Candidate source changed'
assert digest(release/'gt-runtime.json')==staging['config_sha256']
proofs={name:{'path':str(validation/name),'sha256':digest(validation/name)} for name in ('verification.json','report-consumer-verification.json')}
assert all(load_json(x['path'])['state']=='VERIFIED' for x in proofs.values())
assert load_json(full/'compile/progress.json')['completed_frames']==2151 and (full/'STOP_AFTER_FRAME').exists()
assert digest(full/'compile/checkpoint.npz')=='2c737cb749fb6268e4c1f8c630e86f60ebc83e9be490f19b6b8c1b4fca3f0bed'
assert not (full/'compile/controller.lock').exists()
receipt={'state':'VERIFIED_INACTIVE_CANDIDATE','release':str(release),'entrypoint':str(release/'formal_compile_remote.py'),
 'entrypoint_sha256':digest(release/'formal_compile_remote.py'),'config_sha256':staging['config_sha256'],'proofs':proofs,
 'native_inference_frames':0,'full_evaluation_resumed':False,'production_prefix':2151,
 'activation_requirement':'Explicit user instruction to resume; preserve checkpoint and reconcile prepared/uncommitted frames before dispatch',
 'performance_scope':'GT-only throughput measured; live ONNX/board/GT pipeline throughput not measured for this candidate',
 'report_coordinator':'Remains stopped under user pause; restart separately only when explicitly resuming',
 'finished_at':utc_now()}
save_json(deployment/'candidate-ready.json',receipt);print(json.dumps(receipt))
'''

if __name__ == '__main__':print(call(CODE))
