import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
code='''import pathlib,os,json,subprocess
visible=set();unknown=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 visible.add(int(p.name))
 if int(p.name)==os.getpid():continue
 try:
  names=[pathlib.PurePosixPath(x.decode(errors='ignore')).name for x in (p/'cmdline').read_bytes().split(bytes([0]))[:3]]
  if names and names[0].startswith('python') and not any(n in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server','supervisord','update_pubkey.py') for n in names[1:]):unknown.append(int(p.name))
 except OSError:pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
busy=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) in visible] if gpu.returncode==0 else [-1]
print(json.dumps({'idle':not unknown and not busy,'unknown_python':unknown,'gpu_processes':busy}))
'''
value=remote('''import json,sys,pathlib,subprocess,shlex,time
cfg=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4/configs/evaluation.json').read_text())
code=json.load(sys.stdin)['code'];cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','30331','-i',cfg['profile']['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',code])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stderr[-1000:]
value=json.loads(p.stdout);value['checked_at']=time.time()
pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/vision-gt-audit-20261107/idle-proof.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
''',json.dumps({'code':code}).encode(),timeout=60)
(Path(__file__).resolve().parents[1]/'evidence/vision-gt-audit-20261107/idle-proof.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
