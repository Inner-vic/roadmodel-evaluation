"""Read-only proof before replacing an idle same-node evaluation space."""
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from deploy_environment_flow import remote
from pcm_eval.space_api import OfficialSpaces

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'.runtime/evaluation-environment-desktop-20261106-r3/configs/agent.json').read_text())
space=OfficialSpaces(cfg['profile'],cli=cfg['lpai_cli']).read()
probe='''import pathlib,os,json,subprocess
own=set();unknown=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 own.add(int(p.name))
 if int(p.name)==os.getpid():continue
 try:
  names=[pathlib.PurePosixPath(x.decode(errors='ignore')).name for x in (p/'cmdline').read_bytes().split(bytes([0]))[:3]]
  if names and names[0].startswith('python') and not any(n in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server','supervisord','update_pubkey.py') for n in names[1:]):unknown.append({'pid':int(p.name),'program':names[:2]})
 except OSError:pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
busy=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) in own] if gpu.returncode==0 else [-1]
print(json.dumps({'idle':not unknown and not busy,'unknown_python':unknown,'gpu_processes':busy}))
'''
code='''import json,sys,subprocess,shlex,pathlib,time
value=json.load(sys.stdin)
cfg=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6/configs/evaluation.json').read_text())
profile=cfg['profile'];access=value['access'];assert access['host']==profile['ssh_host']
cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p',str(access['port']),'-i',profile['identity_file'],'root@'+access['host'],shlex.join(['python3','-c',value['code']])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
assert p.returncode==0,'Read-only idle probe failed'
proof=json.loads(p.stdout);proof['checked_at']=time.time();print(json.dumps(proof))
'''
result=remote(code,json.dumps({'access':space['runtime_access'],'code':probe}).encode(),timeout=55)
assert result['idle'],'Space is busy; do not stop it'
result.update(space_name=space['metadata']['name'],space_uid=space['metadata']['uid'])
(ROOT/'evidence/rebuild-initialization-20261109/old-space-idle.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
