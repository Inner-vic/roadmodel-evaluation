"""Read-only GPU/runtime and relevant path preflight on the authorized space."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command

parser=argparse.ArgumentParser();parser.add_argument('--port',type=int);parser.add_argument('--identity-file');args=parser.parse_args()
code=r'''
import importlib.util,json,os,pathlib,shutil,sys
r={'python':sys.executable,'python_version':sys.version.split()[0],
   'torch_available':importlib.util.find_spec('torch') is not None,
   'modules':{n:importlib.util.find_spec(n) is not None for n in ['mmengine','mmcv','mmdet3d','mmdet','spconv','cv2','PIL','numpy']},
   'paths':{p:pathlib.Path(p).exists() for p in ['/lpai','/workspace','/mnt/datasets/smartchassisdata-v3/26-10-20-1','/mnt/datasets/smartchassisdata-v3/1-0-10']}}
for p in ['/lpai','/workspace']:
 if pathlib.Path(p).is_dir():r[p+'_directories']=[x.name for x in pathlib.Path(p).iterdir() if x.is_dir()][:40]
if r['torch_available']:
 import torch
 r['torch']=torch.__version__;r['cuda_build']=torch.version.cuda;r['cuda_available']=torch.cuda.is_available()
 if r['cuda_available']:
  r['gpu_count']=torch.cuda.device_count();r['gpu_name']=torch.cuda.get_device_name(0);r['gpu_memory_free_total']=list(torch.cuda.mem_get_info(0))
print(json.dumps(r))
'''
command=ssh_command('lpai',['python3','-c',code])
if args.port:
 command[1:1]=['-p',str(args.port),'-o','StrictHostKeyChecking=accept-new']
if args.identity_file:command[1:1]=['-i',args.identity_file]
proc=subprocess.run(command,capture_output=True,timeout=50)
root=Path('evidence/model-evaluation/model-reference-20261027')
if proc.returncode:
 record={'state':'failed','exit_code':proc.returncode,'ssh_port_override':args.port,'observation':proc.stderr.decode(errors='replace')[-1200:]}
 (root/'gpu-runtime-probe.json').write_text(json.dumps(record)+'\n')
 print(json.dumps(record));raise SystemExit(1)
data=json.loads(proc.stdout);data['state']='queried';data['remote_changed']=False
data['ssh_port_override']=args.port
(root/'gpu-runtime-probe.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
