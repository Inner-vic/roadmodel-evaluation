"""Collect only bounded diagnostic artifacts, preserving earlier evidence."""
import io,tarfile,json,hashlib
from pathlib import Path
from model_space_transport import run
root=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01')
code=r'''
import pathlib,sys,tarfile
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
run=base/'trajectory-diagnostic-01'
files=[p for directory in [run,base/'trajectory-amp-control-01'] for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.py']]
files.extend(base/n for n in ['trajectory-diagnostic-01-console.log','run_trajectory_diagnostic_01.py'])
assert sum(p.stat().st_size for p in files)<64*1024*1024
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for p in files:archive.add(p,arcname=p.relative_to(base).as_posix())
'''
data=run(['python3','-c',code],timeout=180);root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    for m in archive:
        assert m.isfile() and (root/m.name).resolve().is_relative_to(root.resolve())
        body=archive.extractfile(m).read();p=root/m.name;p.parent.mkdir(exist_ok=True,parents=True)
        assert not p.exists() or p.read_bytes()==body,'Existing local diagnostic evidence differs'
        if not p.exists():p.write_bytes(body)
receipt=json.loads((root/'trajectory-diagnostic-01/diagnostic-receipt.json').read_text())
for p in receipt['outputs']:
    assert hashlib.sha256((root/'trajectory-diagnostic-01'/p['path']).read_bytes()).hexdigest()==p['sha256']
component=root/'trajectory-diagnostic-01/semantic-component-replay-02'
if (component/'receipt.json').is_file():
    for p in json.loads((component/'receipt.json').read_text())['outputs']:
        assert hashlib.sha256((component/p['path']).read_bytes()).hexdigest()==p['sha256']
amp=root/'trajectory-amp-control-01'
if (amp/'receipt.json').is_file():
    for p in json.loads((amp/'receipt.json').read_text())['outputs']:
        assert hashlib.sha256((amp/p['path']).read_bytes()).hexdigest()==p['sha256']
print(json.dumps({'state':receipt['state'],'new_forward_frames':receipt['new_forward_frames'],'received_bytes':len(data),'output':str(root)}))
