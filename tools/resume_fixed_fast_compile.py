"""Resume only a proven pre-inference controller-start failure at its checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,os,sys,time,hashlib,subprocess,ast,sqlite3,shutil
b=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-fast-20261030-r3');d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-fast-20261030-r3');prior=d.with_name('compile-fast-20261030-r2')
old_b=b.with_name('compile-fast-20261030-r2');v=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-fast-20261030-r2')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');root=full/'compile'
handoff=json.loads((prior/'handoff.json').read_text());failed=json.loads((root/'progress.json').read_text());cursor=handoff['boundary_cursor']
assert failed['state']=='NEEDS_REVIEW' and failed['error_type']=='NameError' and failed['error']=="name 'validation' is not defined"
assert failed['pid']==handoff['new_pid'] and failed['completed_frames']==cursor
assert not (root/'controller.lock').exists() and not (full/'STOP_AFTER_FRAME').exists()
try:assert not (pathlib.Path('/proc')/str(failed['pid'])/'cmdline').read_bytes()
except FileNotFoundError:pass
frame=root/'frames'/f'frame_{cursor:06d}';assert frame.is_dir() and not list(frame.iterdir()),'Unexpected frame preparation/execution; reconcile first'
backup=pathlib.Path(handoff['backup']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'checkpoint.npz')==sha(backup/'checkpoint.npz')==handoff['checkpoint_sha256']
sources=json.loads((d/'source-receipt.json').read_text());assert 'Ran 8 tests' in sources['unit_tests']
for name,h in sources['source'].items():assert sha(b/name)==h
for name in ('fast_subnet_driver.py','board_validate_subnet.py'):assert sha(b/name)==sha(old_b/name),'Previously validated engine changed'
def function(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
before=(old_b/'formal_compile_remote.py').read_text();after=(b/'formal_compile_remote.py').read_text()
for name in ('validate','ensure_gt','make_pipeline'):assert ast.dump(function(before,name))==ast.dump(function(after,name))
oldmain=function(before,'main');newmain=function(after,'main');replaced=0
for n in ast.walk(oldmain):
 if isinstance(n,ast.Name) and n.id=='validation':n.id='FULL';replaced+=1
assert replaced==1 and ast.dump(oldmain)==ast.dump(newmain),'Full controller differs beyond proven scope fix'
proof=json.loads((v/'verification.json').read_text());assert proof['state']=='VERIFIED' and proof['same_frozen_request_sha256']==sha(full/'request.json')
for name in ('ORT-session-reuse-preflight.json','report-consumer-verification.json'):assert json.loads((v/name).read_text())['state']=='VERIFIED'
assert not (d/'restart-dispatch.json').exists(),'Unknown or already-started restart; inspect before retry'
(d/'failed-start-progress.json').write_text(json.dumps(failed));shutil.copy2(root/'checkpoint.npz',d/'before-restart-checkpoint.npz')
record={'state':'DISPATCHING','preserved_prefix':cursor,'failed_pid':failed['pid'],'known_failure_before_input_preparation':True,
 'native_repeated_frames':0,'validation_reused_without_rerun':str(v),'only_logic_fix':'full branch staging root FULL instead of undefined validation'}
(d/'restart-dispatch.json').write_text(json.dumps(record))
with (d/'full-controller.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,str(b/'formal_compile_remote.py')],cwd=b,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record.update(state='STARTED',new_pid=child.pid);(d/'restart-dispatch.json').write_text(json.dumps(record))
deadline=time.monotonic()+90
while time.monotonic()<deadline:
 p=json.loads((root/'progress.json').read_text())
 if p['pid']==child.pid:
  assert p['state']!='NEEDS_REVIEW',p
  if p['completed_frames']>cursor:break
 if child.poll() is not None:raise ValueError('Fixed controller exited; inspect its evidence, never replay automatically')
 time.sleep(.5)
else:raise ValueError('New controller has not yet committed a frame; inspect progress, do not re-dispatch')
import numpy as np
with np.load(d/'before-restart-checkpoint.npz',allow_pickle=False) as cp:expected={k:cp[k].copy() for k in cp.files if k!='completed'}
with np.load(root/'checkpoint.npz',allow_pickle=False) as cp:new_cursor=int(cp['completed']);actual={k:cp[k].copy() for k in cp.files if k!='completed'}
for i in range(cursor,new_cursor):
 with np.load(root/'frames'/f'frame_{i:06d}'/'single-counts.npz',allow_pickle=False) as cp:
  for key in expected:expected[key]+=cp[key]
assert all(np.array_equal(expected[k],actual[k]) for k in expected)
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
from pcm_eval.runtime import Runtime
from pcm_eval.active_execution import poll
rt=Runtime('/mnt/data/bench-eval/state/roadmodel')
with rt.connect() as c:row=c.execute('SELECT id,owner FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone()
task=rt.get(row['id'],row['owner']);(d/'task-before-refresh.json').write_text(json.dumps(task,ensure_ascii=False))
for chain in task['chains']:
 if chain['kind']=='compile_alignment':
  detail=poll(chain['detail']['binding']);assert detail['state']=='RUNNING' and detail['proof']['compile_controller_alive']
  with rt.connect() as c:c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',('RUNNING',json.dumps(detail,ensure_ascii=False),row['id'],chain['kind']))
with rt.connect() as c:
 states=[x[0] for x in c.execute('SELECT state FROM chains WHERE task=?',(row['id'],))]
 if 'NEEDS_REVIEW' not in states:c.execute("UPDATE tasks SET state='RUNNING',sent_digest='',updated=? WHERE id=?",(time.time(),row['id']))
coordinator=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');report=json.loads((coordinator/'full-report-coordinator.dispatch.json').read_text());os.kill(report['pid'],0)
record.update(state='RUNNING_VERIFIED',current_prefix=new_cursor,counts_exactly_match_old_plus_new=True,report_coordinator_pid=report['pid'],report_coordinator_alive=True)
(d/'restart-dispatch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
if __name__=='__main__':print(call(CODE,timeout=140))
