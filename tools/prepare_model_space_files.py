import json,tarfile,hashlib
from pathlib import Path
from model_space_transport import run,put

LOCAL=Path('evidence/model-evaluation/model-reference-20261027')
REMOTE='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
archive=LOCAL/'source.tar'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if Path(m.name).name=='AGENTS.md': print('SOURCE_INSTRUCTIONS',m.name,t.extractfile(m).read().decode())
print('source_uploaded',put(archive,REMOTE+'/source.tar'),flush=True)
code=r'''
import pathlib,tarfile,json,hashlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'source'
if not root.exists():
 root.mkdir()
 with tarfile.open(base/'source.tar') as t:
  for m in t.getmembers():
   assert m.isfile() or m.isdir(), m.name
   assert (root/m.name).resolve().is_relative_to(root.resolve())
  t.extractall(root,filter='data')
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
(base/'source-manifest.json').write_text(json.dumps({'commit':'9e16e29351d17a726ce6beca6f821de79b7ade40','files':files},indent=2))
print(json.dumps({'source':str(root),'files':len(files)}))
'''
print(run(['python3','-c',code],timeout=180).decode(),flush=True)
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 p=LOCAL/'weights'/version/filename
 target=REMOTE+'/weights/'+filename
 print(json.dumps({'target':target,'sha256':put(p,target),'bytes':p.stat().st_size}),flush=True)
