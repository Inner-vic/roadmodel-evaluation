"""User-requested pause: stop at native frame boundary and freeze report delivery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,time,os,signal,sqlite3,hashlib,datetime,sys
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');model=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');root=full/'compile'
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');audit=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment')/('user-pause-'+now);audit.mkdir(exist_ok=False)
p=json.loads((root/'progress.json').read_text());(audit/'before-progress.json').write_text(json.dumps(p,indent=2));marker=full/'STOP_AFTER_FRAME'
if not marker.exists():
 with marker.open('x') as f:json.dump({'purpose':'explicit user pause; do not resume automatically','requested_at':now,'audit':str(audit)},f)
report=json.loads((model/'full-report-coordinator.dispatch.json').read_text());report_state=json.loads((model/'full-report/coordinator.json').read_text())
(audit/'report-before-pause.json').write_text(json.dumps({'dispatch':report,'progress':report_state},indent=2))
def command(pid):
 try:return (pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
cmd=command(report['pid']);report_paused=not bool(cmd)
if cmd:
 assert 'run_entry_pipeline.py' in cmd and 'full-report-coordinator.json' in cmd,'Report process identity differs'
 assert report_state['state']=='WAITING_NATIVE_EXECUTION','Report no longer idle; inspect publication before stopping'
 os.kill(report['pid'],signal.SIGTERM);report_paused=True
# Do not stop the shared gateway or another user's board work.
deadline=time.monotonic()+55
while time.monotonic()<deadline:
 p=json.loads((root/'progress.json').read_text());cmd=command(p['pid'])
 if not cmd and not (root/'controller.lock').exists():break
 assert 'formal_compile_remote.py' in cmd,'Native process identity differs'
 if p['state']=='NEEDS_REVIEW':break
 time.sleep(.25)
native_paused=not bool(command(p['pid'])) and not (root/'controller.lock').exists()
if not native_paused:
 value={'state':'PAUSE_REQUESTED_WAITING_SAFE_BOUNDARY','progress':p,'report_paused':report_paused,'audit':str(audit)}
 (audit/'pause.json').write_text(json.dumps(value,indent=2));print(json.dumps(value));raise SystemExit(0)
assert p['state'] in ('STOPPED_AT_BOUNDARY','NEEDS_REVIEW','SUCCEEDED'),p
import numpy as np
with np.load(root/'checkpoint.npz',allow_pickle=False) as cp:completed=int(cp['completed']);assert completed==p['completed_frames']
(audit/'checkpoint.npz').write_bytes((root/'checkpoint.npz').read_bytes())
if completed:
 receipt=json.loads((root/'frames'/f'frame_{completed-1:06d}'/'execution_receipt.json').read_text());assert receipt['release']['state']=='RELEASED'
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
from pcm_eval.runtime import Runtime
rt=Runtime('/mnt/data/bench-eval/state/roadmodel')
with rt.connect() as src:
 with sqlite3.connect(audit/'runtime.sqlite3') as dst:src.backup(dst)
with rt.connect() as c:row=c.execute('SELECT id,owner FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone()
task=rt.get(row['id'],row['owner']);(audit/'task-before-pause.json').write_text(json.dumps(task,ensure_ascii=False,indent=2))
notice='已按用户要求暂停评测及自动回填；保留'+str(completed)+'/8337帧编译检查点。模型原生评测已完成，结果保留。等待明确恢复指令，不自动重启。'
with rt.connect() as c:
 for chain in task['chains']:
  detail=chain['detail'];detail['mode_before_user_pause']=detail.get('mode');detail['mode']='user_paused_native_execution';detail['user_paused']=True;detail['issues']=[notice];detail['state']='NEEDS_REVIEW';detail['report_url']=None
  c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',('NEEDS_REVIEW',json.dumps(detail,ensure_ascii=False),row['id'],chain['kind']))
 c.execute("UPDATE tasks SET state='NEEDS_REVIEW',notice=?,sent_digest='',updated=? WHERE id=?",(notice,time.time(),row['id']))
value={'state':'PAUSED_BY_USER','completed_frames':completed,'expected_frames':8337,'native_controller_stopped':True,'report_coordinator_stopped':not bool(command(report['pid'])),
 'last_frame_lease_released':True,'checkpoint_sha256':hashlib.sha256((audit/'checkpoint.npz').read_bytes()).hexdigest(),
 'STOP_AFTER_FRAME_retained':True,'automatic_resume':False,'new_inference_frames':0,'audit':str(audit),
 'model_OCC':json.loads((model/'model/progress.json').read_text())['state'],'model_vision':json.loads((model/'vision/progress.json').read_text())['state']}
(audit/'pause.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value,ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE,timeout=100))
