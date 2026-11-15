"""Seal the tested inactive release and register a passive resume pointer only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE = r'''import pathlib,sys,json,sqlite3,time,subprocess
b=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r2')
d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-batch-20261030-r2')
v=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-batch-20261030-r2')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
sys.path.insert(0,str(b))
from pcm_eval.common import load_json,save_json,digest,utc_now
from pcm_eval.active_execution import process_alive
import formal_compile_remote as c
staged=load_json(d/'staging.json')
for name,h in staged['source'].items():assert digest(b/name)==h,'Candidate source/config changed'
verified=load_json(v/'verification.json');assert verified['state']=='VERIFIED' and verified['reference_workers']==2
assert verified['report_validity']=='validated' and verified['report_tensor_records']==108
assert verified['all49_checks_and18_comparisons_preserved'] and verified['metric_definitions_unchanged']
assert load_json(full/'compile/progress.json')['state']=='STOPPED_AT_BOUNDARY' and load_json(full/'compile/progress.json')['completed_frames']==2151
reservation=load_json(v/'reservation.json')
for name,key in [('compile/checkpoint.npz','checkpoint_sha256'),('compile/progress.json','progress_sha256'),('STOP_AFTER_FRAME','stop_marker_sha256')]:assert digest(full/name)==reservation[key]
assert not (full/'compile/controller.lock').exists()
try:c.main()
except ValueError as error:assert 'paused by user' in str(error)
else:raise AssertionError('Candidate ignored user pause')
dispatch=load_json(d/'dispatch.json');p=pathlib.Path('/proc')/str(dispatch['pid'])/'cmdline'
deadline=time.monotonic()+15
while p.exists() and p.read_bytes():
 assert time.monotonic()<deadline,'Validation process still exiting'
 time.sleep(.25)
assert not process_alive(load_json(full/'compile/progress.json')['pid'],'formal_compile_remote.py')
model=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01')
coordinator=load_json(model/'full-report-coordinator.dispatch.json');p=pathlib.Path('/proc')/str(coordinator['pid'])/'cmdline'
assert not p.exists() or not p.read_bytes()
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/runtime.sqlite3') as db:
 rows=db.execute('SELECT detail FROM chains WHERE task=?',('rm-a361dbe3df384cccb053',)).fetchall()
 assert len(rows)==2 and all(json.loads(row[0]).get('user_paused') for row in rows)
inventory=[]
for i in range(2151,2176):
 root=full/'compile/frames'/('frame_%06d'%i)
 if (root/'execution_receipt.json').exists():
  receipt=load_json(root/'execution_receipt.json');proof=load_json(root/'single-counts-receipt.json')
  assert receipt['state']=='succeeded' and receipt['selected_frames']==[c.FRAMES[i]] and receipt['release']['state']=='RELEASED'
  assert proof['counts_sha256']==digest(root/'single-counts.npz') and proof['execution_sha256']==digest(root/'execution_receipt.json')
  inventory.append({'index':i,'state':'NATIVE_AND_GT_RETAINED','counts_sha256':proof['counts_sha256']})
 elif (root/'prepare.json').exists():
  prepared=load_json(root/'prepare.json');assert prepared['state']=='READY'
  inventory.append({'index':i,'state':'REFERENCE_READY_NATIVE_NOT_DISPATCHED','prepare_sha256':digest(root/'prepare.json')})
queue=json.loads(subprocess.run([str(c.old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,timeout=20,check=True).stdout)
proofs={name:{'path':str(v/name),'sha256':digest(v/name)} for name in ('verification.json','measured-execution.json')}
receipt={'state':'VERIFIED_INACTIVE_CANDIDATE','release':str(b),'controller':str(b/'formal_compile_remote.py'),
 'controller_sha256':digest(b/'formal_compile_remote.py'),'source_receipt_sha256':digest(d/'staging.json'),'proofs':proofs,
 'checkpoint_prefix':2151,'checkpoint_sha256':reservation['checkpoint_sha256'],'pause_marker_unchanged':True,
 'Runtime_still_user_paused':True,'report_coordinator_still_stopped':True,'full_evaluation_resumed':False,
 'transport':'one input packet, one private remote orchestrator, one checked output bundle per frame',
 'reference_workers':2,'reference_threads':'original default unchanged','GT_workers':4,'metric_definitions_unchanged':True,
 'mean_sampled_completion_interval_seconds':verified['mean_board_completion_interval_seconds'],
 'sampled_later_three_intervals_seconds':verified['mean_later_three_completion_intervals_seconds'],
 'six_frame_cold_start_through_report_seconds':verified['end_to_end_seconds_including_startup_and_report'],
 'steady_full_run_not_claimed':True,'retained_slots':inventory,'original_queue':queue,
 'resume_requirement':'Only resume after an explicit user instruction; preserve checkpoint, adopt cached native/GT and reference-only slots, then restart report coordinator separately',
 'note':'Preparation start=0/end=duration in this prefetch benchmark represents worker-local elapsed duration from original prepare.json, not global monotonic timestamps. Use board/GT events and total wall time for pipeline timing.',
 'sealed_at':utc_now()}
save_json(d/'candidate-ready.json',receipt)
pointer={'state':'INACTIVE_WAITING_EXPLICIT_USER_RESUME','candidate_receipt':str(d/'candidate-ready.json'),
 'candidate_receipt_sha256':digest(d/'candidate-ready.json'),'full_evaluation_resumed':False,'checkpoint_prefix':2151}
save_json(full/'compile-next-controller.json',pointer)
print(json.dumps({k:value for k,value in receipt.items() if k not in ('retained_slots','proofs','note')}))
'''

if __name__ == '__main__':print(call(CODE))
