"""Fixed workstation preparation protocol used by the desktop Agent Runtime."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcm_eval.space_preparation_queue import PreparationQueue


def main(queue_root):
    packet=json.load(sys.stdin);queue=PreparationQueue(queue_root);op=packet['op'];client=packet['client']
    if op=='claim':return queue.claim(client)
    request=queue.context(packet['id'],client,packet['lease'])
    if op=='update':
        queue.update(packet['id'],client,packet['lease'],packet['state'],packet['detail']);return {'ok':True}
    profile=request['profile'];plan=packet['plan']
    if plan['space_uid']!=profile['space_uid'] or plan['asset']['source_ref']!=request['inputs']['dataset_ref']:
        raise ValueError('Probe identity differs from confirmed request')
    if op=='probe':
        code=(ROOT/'pcm_eval/dataset_mount.py').read_text()+'\nplan='+repr(plan)+'\nprint(json.dumps(probe_mounted_index(plan,index_relative=plan["asset"].get("index_relative_hint"),include_manifest=True)))'
    elif op=='idle':
        # No user commands are executed here. Report active application commands
        # as a count; don't expose process arguments, environments, or secrets.
        code='''import pathlib,json,os,subprocess
busy=0;own=set();unknown_python=0
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  own.add(int(p.name))
  if int(p.name)==os.getpid():continue
  try:
   parts=(p/'cmdline').read_bytes().split(b'\\x00');names=[pathlib.PurePosixPath(x.decode(errors='ignore')).name for x in parts[:3]]
   if any(x in ('eval.py','train.py','native-worker.py') for x in names):busy+=1
   if names and names[0].startswith('python') and not any(x in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server') for x in names[1:3]):unknown_python+=1
  except (OSError,ValueError):pass
gpu=None
try:
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
 if r.returncode==0:gpu=sum(int(x.strip()) in own for x in r.stdout.splitlines() if x.strip().isdigit())
except (OSError,subprocess.TimeoutExpired):pass
print(json.dumps({'idle':busy==0 and unknown_python==0 and gpu==0,'application_process_count':busy,'unclassified_python_count':unknown_python,'container_gpu_processes':gpu}))'''
    else:raise ValueError('Unknown preparation operation')
    argv=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12',
          '-o','StrictHostKeyChecking=accept-new','-p',str(profile['ssh_port']),'-i',profile['identity_file'],
          'root@'+profile['ssh_host'],shlex.join(['python3','-c',code])]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=100)
    if p.returncode:raise ValueError('LPAI space read-only probe failed; native receipt requires review')
    return json.loads(p.stdout)


if __name__=='__main__':
    try:print(json.dumps({'ok':True,'data':main(sys.argv[1])},ensure_ascii=False))
    except Exception as error:print(json.dumps({'ok':False,'error_type':type(error).__name__,'reason':str(error)[:300]},ensure_ascii=False));sys.exit(1)
