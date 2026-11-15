"""Reuse workstation cache; upload only absent v3.2.2 model files and inspect lock API."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command
from pcm_eval.common import load_json,digest,save_json

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/data/developer/pcm-roadmodel-eval/model-materials/v3.2.2-subnets'


def remote(code):
    p=subprocess.run(ssh_command('ubuntu-workstation',['python3','-']),input=code.encode(),capture_output=True,timeout=80)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
    return p.stdout.decode()


if __name__=='__main__':
    audit=load_json(ROOT/'evidence/compile-subnet-review-20261029/version-and-port-audit.json')
    paths=[ROOT/'evidence/model-evaluation/model-reference-20261027/v322.onnx']+[Path(x['archive']) for x in audit['current']['stages']]
    receipt={'files':[],'dataset_files_transferred':0}
    for path in paths:
        target=REMOTE+'/'+path.name;sha=digest(path)
        query="import pathlib,hashlib,json;p=pathlib.Path("+repr(target)+");p.parent.mkdir(parents=True,exist_ok=True);print(json.dumps({'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}))"
        present=json.loads(remote(query))
        if present['exists'] and present['sha256']!=sha:raise ValueError('Different existing model file; preserve and inspect')
        if not present['exists']:
            p=subprocess.run(['scp.exe','-q','-o','BatchMode=yes',path.name,'ubuntu-workstation:'+target],cwd=path.parent,capture_output=True,timeout=240)
            if p.returncode:raise RuntimeError('Model-only transfer failed: '+p.stderr.decode(errors='replace')[-1000:])
            assert json.loads(remote(query))['sha256']==sha
        receipt['files'].append({'path':target,'sha256':sha,'reused':present['exists']})
        print(json.dumps(receipt['files'][-1]),flush=True)
    query=r'''import pathlib,json,hashlib
base=pathlib.Path('/home/lixiang/bench-eval/current/automation')
out={}
for name in ('runner_state.py','bench_transaction.py'):
 lines=(base/name).read_text().splitlines();keep=set()
 for i,line in enumerate(lines):
  if 'class BenchLock' in line or 'class ResourceLock' in line or 'class Global' in line or 'def acquire' in line or 'def release' in line or 'class Lock' in line:
   keep.update(range(max(0,i-3),min(len(lines),i+85)))
 out[name]='\n'.join(f'{i+1}: {lines[i]}' for i in sorted(keep))
base=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval')
out['old_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'package/benchEval-skill-20261020/examples/assets').iterdir() if p.is_file()}
out['frame_manifest']=json.loads((base/'runs/pcm-v7-multiscene-001/frame_000000/frame_manifest.json').read_text())
print(json.dumps(out))'''
    details=json.loads(remote(query));save_json(ROOT/'evidence/compile-subnet-review-20261029/remote-preparation.json',{'transfer':receipt,**details})
    print(json.dumps(details,ensure_ascii=False,indent=2))
