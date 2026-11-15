"""Use the authorized existing space without changing the user's SSH config."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

IDENTITY=os.environ.get('ROADMODEL_IDENTITY','C:/Users/developer/.ssh/bmad_roadmodel_ed25519')
HOST='ssh-d.lpai.lixiang.com'
PORT=31017

def command(argv):
    return [shutil.which('ssh.exe') or 'ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
            '-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
            '-p',str(PORT),'-i',IDENTITY,'root@'+HOST,shlex.join(argv)]

def run(argv,payload=None,timeout=90):
    proc=subprocess.run(command(argv),input=payload,capture_output=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError('Authorized model space command failed: '+proc.stderr.decode(errors='replace')[-1800:])
    return proc.stdout

def put(local,remote):
    # The caller chooses an isolated path; never overwrite a differing file.
    writer="import pathlib,sys,hashlib;p=pathlib.Path(sys.argv[1]);b=sys.stdin.buffer.read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==b,'File content differs';p.write_bytes(b) if not p.exists() else None;print(hashlib.sha256(b).hexdigest())"
    return run(['python3','-c',writer,remote],Path(local).read_bytes(),timeout=180).decode().strip()
