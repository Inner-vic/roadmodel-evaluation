import io,tarfile,json,hashlib,re
from pathlib import Path
from model_space_transport import run
root=Path('evidence/model-evaluation/remaining-heads-20261027/collected')
code=r'''
import pathlib,sys,tarfile
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
files=[]
for name in ['vision-01','trajectory-01','trajectory-01-recovered']:
 files.extend(p for p in (base/name).rglob('*') if p.is_file() and p.suffix in ['.json','.log','.npz','.jpg','.py'])
files.extend(base/n for n in ['workflow-state.json','workflow-console.log','trajectory_head_adapter.py','head_preflight.py','run_remaining_heads.py','input-manifest.json'])
assert sum(p.stat().st_size for p in files)<100*1024*1024
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p in files:t.add(p,arcname=p.relative_to(base).as_posix())
'''
raw=run(['python3','-c',code],timeout=180);root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 for m in t.getmembers():
  assert m.isfile() and (root/m.name).resolve().is_relative_to(root.resolve())
  body=t.extractfile(m).read();p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==body,'Existing evidence differs'
  if not p.exists():p.write_bytes(body)
vision=root/'vision-01';receipt=json.loads((vision/'execution_receipt.json').read_text());pre=json.loads((vision/'model-preflight.json').read_text())
console=(vision/'eval-console.log').read_text(encoding='utf-8')
assert receipt['state']=='succeeded' and pre['state']=='passed' and pre['dataset_size']==4
assert re.search(r'100%[^\r\n]*\b4/4\b',console) and 'Total samples used: 4' in console
coverage={'schema':'roadmodel.native-coverage/v1','state':'verified','run_id':receipt['run_id'],'expected_frames':4,
 'observed_frames':4,'completed_batches':4,'batch_size':1,'selected_frames':receipt['selected_frames'],
 'console_sha256':hashlib.sha256((vision/'eval-console.log').read_bytes()).hexdigest(),
 'loader_preflight_sha256':hashlib.sha256((vision/'model-preflight.json').read_bytes()).hexdigest(),
 'evidence':'native eval completed 100% 4/4 and Total samples used: 4 with matching frozen 4-item loader'}
(vision/'coverage_receipt.json').write_text(json.dumps(coverage,indent=2)+'\n')
print(json.dumps({'bytes_received':len(raw),'files':len([p for p in root.rglob('*') if p.is_file()]),
 'trajectory_files':[p.name for p in (root/'trajectory-01-recovered').iterdir()],
 'vision_artifacts':[str(p.relative_to(vision)) for p in vision.rglob('*') if p.suffix in ['.json','.jpg']]}))
