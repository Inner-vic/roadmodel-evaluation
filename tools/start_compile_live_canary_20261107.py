"""Authorize exactly two unattempted native frames; leave the full run paused."""
import json
from pathlib import Path
from deploy_environment_flow import remote

WORKER='''import pathlib,json,sys,time,traceback
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
sys.path[:0]=[str(release),str(release/'tools')]
from pcm_eval.compile_scope import execute
root=pathlib.Path(sys.argv[1]);cfg=json.loads((release/'configs/evaluation.json').read_text())
plan=json.loads((root/'plan.json').read_text());started=time.monotonic()
try:
 result=execute(plan,root/'execution',cfg['compile_profile'],bench_available=True)
 (root/'result.json').write_text(json.dumps(result,indent=2))
 (root/'status.json').write_text(json.dumps({'state':'SUCCEEDED','expected_frames':2,'new_frames':plan['frames'],'elapsed_seconds':time.monotonic()-started,'native_result':result},indent=2))
except Exception as error:
 (root/'status.json').write_text(json.dumps({'state':'NEEDS_REVIEW','error':str(error),'elapsed_seconds':time.monotonic()-started},indent=2));traceback.print_exc();raise
'''

START='''import pathlib,json,sys,copy,subprocess,hashlib,numpy as np
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4');sys.path.insert(0,str(release))
from pcm_eval.common import json_digest,digest
root=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-live-20261107-r4')
assert not root.exists(),'Prior canary must be reconciled, never automatically repeated'
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
with np.load(full/'compile/checkpoint.npz',allow_pickle=False) as cp:assert int(cp['completed'])==2151
old_progress=json.loads((full/'compile/progress.json').read_text())
assert old_progress['state'] in ('NEEDS_REVIEW','STOPPED_AT_BOUNDARY')
assert not (full/'compile/controller.lock').exists()
old_process=pathlib.Path('/proc')/str(old_progress['pid'])/'cmdline'
assert not old_process.exists() or str(full).encode() not in old_process.read_bytes()
request=json.loads((full/'request.json').read_text());indices=[2176,2177]
assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in indices),'Selected frame already attempted'
queue=subprocess.run(['/home/lixiang/bench-eval/bin/bench-eval','queue'],capture_output=True,text=True,check=True,timeout=20)
q=json.loads(queue.stdout);assert not(q.get('active') or q.get('queued') or q.get('waiting'))
assert not pathlib.Path('/home/lixiang/bench-eval/locks/bench.lock').exists()
workflow=json.loads(pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-997e5dda6e8546a4a935/workflow.json').read_text())
plan=copy.deepcopy(workflow['jobs'][1]['plan']);frames=[request['frames'][i] for i in indices]
plan.update(run_id='rm-997e5dda6e8546a4a935-canary-20261107',frames=frames,expected_frames=2)
plan['request']['evaluation'].update(expected_frames=2,max_frames=2,selection={'frames':frames})
plan.pop('plan_sha256');plan['plan_sha256']=json_digest(plan)
root.mkdir(parents=True);(root/'plan.json').write_text(json.dumps(plan,indent=2))
(root/'worker.py').write_bytes(sys.stdin.buffer.read())
proof={'indices':indices,'frames':frames,'old_checkpoint_sha256':digest(full/'compile/checkpoint.npz'),'new_inference_limit':2,'no_completed_frame_selected':True,'full_run_remains_paused':True}
(root/'reservation.json').write_text(json.dumps(proof,indent=2))
(root/'status.json').write_text(json.dumps({'state':'DISPATCHING','frames':frames}))
with (root/'worker.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,str(root/'worker.py'),str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'dispatch.json').write_text(json.dumps({'state':'STARTED','pid':child.pid,'plan_sha256':plan['plan_sha256']},indent=2))
print(json.dumps({'state':'STARTED','pid':child.pid,'root':str(root),**proof}))
'''

if __name__=='__main__':
    result=remote(START,WORKER.encode(),timeout=45)
    root=Path('evidence/compile-live-20261107');root.mkdir(exist_ok=True)
    (root/'dispatch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
