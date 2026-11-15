"""Stage private optimizer, validate unused frames, then checkpoint handoff."""
import io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/compile-fast-20261030-r3'
DEPLOY='/mnt/data/bench-eval/roadmodel/deployment/compile-fast-20261030-r3'
VALID='/mnt/data/bench-eval/roadmodel/validation/compile-fast-20261030-r2'
PY='/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'


def prepare():
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        for name,remote in [('board_validate_subnet.py','board_validate_subnet.py'),('fast_subnet_driver.py','fast_subnet_driver.py'),('fast_compile_controller.py','formal_compile_remote.py')]:
            t.add(ROOT/'tools'/name,arcname=remote)
        t.add(ROOT/'tests/test_fast_compile.py',arcname='tests/test_fast_compile.py')
    code="""import pathlib,sys,tarfile,io,json,hashlib,subprocess
b=pathlib.Path(%r);d=pathlib.Path(%r);d.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(b/m.name).resolve();p.relative_to(b);p.parent.mkdir(parents=True,exist_ok=True);data=tar.extractfile(m).read();assert not p.exists() or p.read_bytes()==data
  if not p.exists():p.write_bytes(data)
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_fast_compile.py','-q'],cwd=b,capture_output=True)
(d/'unit-tests.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr).decode()
source={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in b.glob('*.py')}
(d/'source-receipt.json').write_text(json.dumps({'state':'VERIFIED','source':source,'native_inference':0,'unit_tests':(p.stdout+p.stderr).decode()},indent=2))
print(json.dumps({'state':'PROGRAM_PREPARED','release':str(b),'unit_tests':(p.stdout+p.stderr).decode()}))
"""%(RELEASE,DEPLOY)
    return call(code,buf.getvalue())


def validate():
    code="""import pathlib,subprocess,sys,os,json
b=pathlib.Path(%r);d=pathlib.Path(%r);v=pathlib.Path(%r)
assert not v.exists() and not (d/'validation-dispatch.json').exists(),'No repeated validation'
env={**os.environ,'ROADMODEL_OPT_VALIDATION':str(v),'PYTHONUNBUFFERED':'1'}
with (d/'validation.log').open('xb') as f:p=subprocess.Popen([sys.executable,str(b/'formal_compile_remote.py'),'validate'],cwd=b,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'STARTED','pid':p.pid,'validation':str(v)};(d/'validation-dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
"""%(RELEASE,DEPLOY,VALID)
    return call(code)


def status():
    code="""import pathlib,json,os
d=pathlib.Path(%r);v=pathlib.Path(%r);full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
out={}
for label,p in [('validation',v/'progress.json'),('verification',v/'verification.json'),('full',full/'compile/progress.json')]:
 if p.exists():out[label]=json.loads(p.read_text())
out['validation_log']=(d/'validation.log').read_text(errors='replace')[-4200:] if (d/'validation.log').exists() else ''
if (d/'validation-dispatch.json').exists():
 p=json.loads((d/'validation-dispatch.json').read_text())['pid'];out['validation_pid']=p
 try:os.kill(p,0);out['validation_alive']=True
 except ProcessLookupError:out['validation_alive']=False
print(json.dumps(out,ensure_ascii=False))
"""%(DEPLOY,VALID)
    return call(code)


def handoff():
    code="""import pathlib,sys,json,os,time,hashlib,shutil,subprocess,sqlite3
b=pathlib.Path(%r);d=pathlib.Path(%r);v=pathlib.Path(%r)
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');root=full/'compile'
proof=json.loads((v/'verification.json').read_text());assert proof['state']=='VERIFIED' and proof['all18_tensor_metrics_exact_match'] and proof['same_metric_definitions'] and proof['all49_hash_finite_checks_preserved']
assert json.loads((v/'ORT-session-reuse-preflight.json').read_text())['state']=='VERIFIED'
assert json.loads((v/'report-consumer-verification.json').read_text())['state']=='VERIFIED'
assert proof['same_frozen_request_sha256']==hashlib.sha256((full/'request.json').read_bytes()).hexdigest()
sources=json.loads((d/'source-receipt.json').read_text())['source']
for name,h in sources.items():assert hashlib.sha256((b/name).read_bytes()).hexdigest()==h,'Reviewed optimizer changed'
assert not (d/'handoff.json').exists(),'Handoff already attempted; inspect evidence'
before=json.loads((root/'progress.json').read_text());old_pid=before['pid'];assert old_pid==json.loads((v/'reservation.json').read_text())['full_PID'] and before['state'] in ('RUNNING','WAITING_BENCH')
assert 'formal_compile_remote.py' in (pathlib.Path('/proc')/str(old_pid)/'cmdline').read_bytes().decode(errors='replace')
marker=full/'STOP_AFTER_FRAME';assert not marker.exists()
with marker.open('x') as f:json.dump({'purpose':'authorized verified optimizer handoff','validation':str(v)},f)
(d/'handoff.json').write_text(json.dumps({'state':'WAITING_FRAME_BOUNDARY','old_pid':old_pid,'old_progress':before}))
deadline=time.monotonic()+110
while time.monotonic()<deadline:
 state=json.loads((root/'progress.json').read_text())
 if state['state']=='STOPPED_AT_BOUNDARY' and not (root/'controller.lock').exists():break
 if state['state']=='NEEDS_REVIEW':raise ValueError('Original controller needs reconciliation; no replacement dispatched')
 time.sleep(.5)
else:raise ValueError('Original frame has not stopped safely; no replacement dispatched')
exit_deadline=time.monotonic()+10
while True:
 try:cmd=(pathlib.Path('/proc')/str(old_pid)/'cmdline').read_bytes()
 except FileNotFoundError:break
 if not cmd:break
 if time.monotonic()>exit_deadline:raise ValueError('Original controller still exiting; no new controller dispatched')
 time.sleep(.05)
backup=d/('boundary-'+str(state['completed_frames']));backup.mkdir(exist_ok=False)
for name in ('progress.json','checkpoint.npz','metrics_board.partial.json','metrics_onnx.partial.json'):shutil.copy2(root/name,backup/name)
import numpy as np
sys.path[:0]=[str(full/'code')];from gt_adapter import Counts
with np.load(backup/'checkpoint.npz',allow_pickle=False) as cp:
 cursor=int(cp['completed']);assert cursor==state['completed_frames']
 for backend in ('board','onnx'):
  acc=Counts();acc.confusion=cp[backend+'_confusion'].copy();acc.ds=cp[backend+'_ds'].copy();acc.num_frames=cursor
  assert acc.summary()==json.loads((backup/('metrics_'+backend+'.partial.json')).read_text()),'Checkpoint metrics differ'
assert cursor<min(proof['indices'])
coordinator=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');dispatch=json.loads((coordinator/'full-report-coordinator.dispatch.json').read_text());os.kill(dispatch['pid'],0)
rtroot=pathlib.Path('/mnt/data/bench-eval/state/roadmodel')
with sqlite3.connect(rtroot/'runtime.sqlite3') as src:
 with sqlite3.connect(backup/'runtime.sqlite3') as dst:src.backup(dst)
marker.unlink()
with (d/'full-controller.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,str(b/'formal_compile_remote.py')],cwd=b,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
transition={'state':'STARTED','old_pid':old_pid,'new_pid':child.pid,'boundary_cursor':cursor,'checkpoint_sha256':hashlib.sha256((backup/'checkpoint.npz').read_bytes()).hexdigest(),
 'backup':str(backup),'validation':str(v),'native_restarted_frames':0,'frozen_request_unchanged':True,'report_coordinator_pid':dispatch['pid']}
(d/'handoff.json').write_text(json.dumps(transition,indent=2))
deadline=time.monotonic()+40
while time.monotonic()<deadline:
 current=json.loads((root/'progress.json').read_text())
 if current['pid']==child.pid:
  assert current['state']!='NEEDS_REVIEW',current
  break
 if child.poll() is not None:raise ValueError('New controller stopped; inspect its log before any retry')
 time.sleep(.5)
else:raise ValueError('New controller startup outcome needs review')
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
from pcm_eval.runtime import Runtime
from pcm_eval.active_execution import poll
rt=Runtime(rtroot)
with rt.connect() as c:row=c.execute('SELECT id,owner FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone()
task=rt.get(row['id'],row['owner']);(backup/'task-before-refresh.json').write_text(json.dumps(task,ensure_ascii=False))
for chain in task['chains']:
 if chain['kind']=='compile_alignment':
  detail=poll(chain['detail']['binding']);assert detail['state']=='RUNNING' and detail['proof']['compile_controller_alive']
  with rt.connect() as c:c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',('RUNNING',json.dumps(detail,ensure_ascii=False),row['id'],chain['kind']))
with rt.connect() as c:
 states=[x[0] for x in c.execute('SELECT state FROM chains WHERE task=?',(row['id'],))]
 if 'NEEDS_REVIEW' not in states:c.execute("UPDATE tasks SET state='RUNNING',sent_digest='',updated=? WHERE id=?",(time.time(),row['id']))
transition.update(state='HANDED_OFF',new_progress=current,report_coordinator_alive=True);(d/'handoff.json').write_text(json.dumps(transition,indent=2));print(json.dumps(transition))
"""%(RELEASE,DEPLOY,VALID)
    return call(code,timeout=180)

if __name__=='__main__':print({'prepare':prepare,'validate':validate,'status':status,'handoff':handoff}[sys.argv[1]]())
