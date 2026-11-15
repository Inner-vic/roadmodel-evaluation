import json,tarfile,hashlib
from pathlib import Path
from model_space_transport import run,put
root=Path('evidence/model-evaluation/remaining-heads-20261027')
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027'
for name in ['vision-subset.json','trajectory-subset.json']:
 source=root/name;target=root/'input'/name
 target.write_bytes(source.read_bytes())
archive=root/'head-inputs.tar'
with tarfile.open(archive,'w') as t:
 for p in (root/'input').rglob('*'):
  if p.is_file():t.add(p,arcname=p.relative_to(root/'input').as_posix())
print('uploaded',put(archive,base+'/head-inputs.tar'),flush=True)
code=r'''
import pathlib,tarfile,json,hashlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
root=base/'input';root.mkdir(exist_ok=True)
with tarfile.open(base/'head-inputs.tar') as t:
 for m in t.getmembers():
  assert m.isfile() and (root/m.name).resolve().is_relative_to(root.resolve())
  raw=t.extractfile(m).read();p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==raw,'Different existing input'
  if not p.exists():p.write_bytes(raw)
files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.rglob('*') if p.is_file()]
(base/'input-manifest.json').write_text(json.dumps(files,indent=2))
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
'''
print(run(['python3','-c',code],timeout=120).decode())
