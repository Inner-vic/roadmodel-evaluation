"""Deploy only reviewed adapter files; run exactly one cached frame on the workstation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command
from pcm_eval.common import digest
from prepare_subnet_remote import remote,REMOTE

ROOT=Path(__file__).resolve().parents[1]
ADAPTER='/mnt/data/developer/pcm-roadmodel-eval/subnet-adapter-20261029'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--board',action='store_true');args=p.parse_args()
    import re
    if not re.fullmatch(r'pcm-v322-subnets-[A-Za-z0-9_-]{1,40}',args.run_id):raise ValueError('Invalid run ID')
    remote('import pathlib;pathlib.Path('+repr(ADAPTER)+').mkdir(parents=True,exist_ok=True)')
    files=[(ROOT/'tools/subnet_chain_remote.py',ADAPTER+'/subnet_chain_remote.py'),
           (ROOT/'pcm_eval/subnets.py',ADAPTER+'/subnet_contract.py'),
           (ROOT/'evidence/compile-subnet-review-20261029/version-and-port-audit.json',REMOTE+'/source-audit.json')]
    for path,target in files:
        # Reviewed source may be updated before a new isolated run. Existing run directories remain frozen.
        proc=subprocess.run(['scp.exe','-q','-o','BatchMode=yes',path.name,'ubuntu-workstation:'+target],cwd=path.parent,capture_output=True,timeout=45)
        if proc.returncode:raise RuntimeError('Adapter upload failed')
        verify=remote('import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('+repr(target)+').read_bytes()).hexdigest())').strip()
        assert verify==digest(path)
    argv=['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python','-u',ADAPTER+'/subnet_chain_remote.py','--run-id',args.run_id]
    if args.board:argv.append('--board')
    out=ROOT/'evidence/compile-subnet-review-20261029'/args.run_id
    out.mkdir(exist_ok=False)
    with (out/'console.log').open('xb') as log:
        proc=subprocess.run(ssh_command('ubuntu-workstation',argv),stdout=log,stderr=subprocess.STDOUT,timeout=900)
    print(json.dumps({'run_id':args.run_id,'exit_code':proc.returncode,'log':str(out/'console.log'),'max_frames':1}))
    raise SystemExit(proc.returncode)
