import tarfile,json,hashlib
from pathlib import Path
from model_space_transport import run,put
root=Path('evidence/model-evaluation/model-reference-20261027')
remote='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
archive=root/'two-frame-input.tar'
with tarfile.open(archive,'w') as t:
 for p in (root/'two-frame-input').rglob('*'):
  if p.is_file():t.add(p,arcname=p.relative_to(root/'two-frame-input').as_posix())
print('uploaded',put(archive,remote+'/two-frame-input.tar'),flush=True)
code=r'''
import pathlib,tarfile,hashlib,json
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'input';root.mkdir(exist_ok=True)
with tarfile.open(base/'two-frame-input.tar') as t:
 for m in t.getmembers():
  assert m.isfile() and (root/m.name).resolve().is_relative_to(root.resolve())
  b=t.extractfile(m).read();p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==b,'Conflicting input file'
  if not p.exists():p.write_bytes(b)
print(json.dumps({'files':len([p for p in root.rglob('*') if p.is_file()]),'data_root':str(root)}))
'''
print(run(['python3','-c',code]).decode())
