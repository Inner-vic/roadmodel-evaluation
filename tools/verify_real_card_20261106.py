"""Read-only verification for the authorized two-frame production card."""
import json
from pathlib import Path

from deploy_environment_flow import remote

IDLE = '''import os,pathlib,json,subprocess
own={int(x.name) for x in pathlib.Path('/proc').iterdir() if x.name.isdigit()}
unknown=[]
for pid in own:
 if pid==os.getpid():continue
 try:
  args=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().split(bytes([0]))
  names=[pathlib.PurePosixPath(a.decode(errors='ignore')).name for a in args[:3]]
  if names and names[0].startswith('python') and not any(n in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server','supervisord','update_pubkey.py') for n in names[1:]):
   unknown.append({'pid':pid,'executables':names[:2]})
 except OSError:pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
busy=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) in own]
print(json.dumps({'idle':not unknown and not busy and gpu.returncode==0,'unknown_python':unknown,'container_gpu_pids':busy}))
'''

VERIFY = '''import pathlib,json,sys,subprocess,shlex
sys.path.insert(0,NEW)
from pcm_eval.evaluation import EvaluationRuntime
payload=json.load(sys.stdin)
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation')
cfg=json.loads(r.config_file.read_text());task='rm-997e5dda6e8546a4a935';run=r.root/'runs'/task
t=r.get(task,cfg['owner'])
with r.connect() as c:
 pub=dict(c.execute('select * from publications where task=?',(task,)).fetchone())
 active=c.execute("select count(*) from materials where state='READING'").fetchone()[0]+c.execute("select count(*) from publications where state='WRITING'").fetchone()[0]
receipt=json.loads(next(run.glob('effective/**/native/occ/batch-0000/execution_receipt.json')).read_text())
proof={'task_id':task,'event_id':t['event_id'],'message_id':t['message_id'],'state':t['state'],'scope':t['options'],'selected_frames':receipt['selected_frames'],'native_duration_seconds':receipt['duration_seconds'],'native_state':receipt['state'],'execution_location':'lpai','dataset_copied':False,'new_inference_frames':receipt['expected_frames'],'publication':json.loads(pub['detail']),'bench_contacted':False,'full_run_started':False,'startup_mode':'codex_foreground','background_browser_verified':False,'active_agent_operations':active}
p=cfg['profile']
args=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','30331','-i',p['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',payload['idle_code']])]
x=subprocess.run(args,capture_output=True,text=True,timeout=35)
assert x.returncode==0,x.stderr[-500:]
proof['idle_probe']=json.loads(x.stdout)
(pathlib.Path(STAGE)/'real-card-end-to-end-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
'''

if __name__ == '__main__':
    result=remote(VERIFY,json.dumps({'idle_code':IDLE}).encode(),timeout=50)
    Path('evidence/environment-init-20261106/real-card-end-to-end-proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
