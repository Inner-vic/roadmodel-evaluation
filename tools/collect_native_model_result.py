import io,json,tarfile,hashlib,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_space_transport import run
from pcm_eval.parsers import collect_model

root=Path('evidence/model-evaluation/native-v3.2.2-smoke-20261027')
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
code=r'''
import pathlib,sys,tarfile
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for name in ['execution_receipt.json','model-preflight.json','operator-build-state.json','import-preflight.json','source-manifest.json','native-launch.json','eval-console.log']:
  p=base/name;t.add(p,arcname=name)
 for p in (base/'run-01').rglob('*'):
  if p.is_file() and p.suffix in ['.log','.py','.json']:
   t.add(p,arcname=p.relative_to(base).as_posix())
'''
raw=run(['python3','-c',code],timeout=180)
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 for m in t.getmembers():
  assert m.isfile() and (root/m.name).resolve().is_relative_to(root.resolve())
  b=t.extractfile(m).read();p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b,'Preserve existing evidence'
  else:p.write_bytes(b)
receipt=json.loads((root/'execution_receipt.json').read_text())
preflight=json.loads((root/'model-preflight.json').read_text())
console=(root/'eval-console.log').read_text(encoding='utf-8')
assert receipt['state']=='succeeded' and receipt['exit_code']==0
assert preflight['state']=='passed' and preflight['dataset_size']==2
assert re.search(r'100%[^\r\n]*\b2/2\b',console),'No evidence of completing both native eval batches'
logs=list((root/'run-01').glob('*/*.log'));assert len(logs)==1,logs
coverage={'schema':'roadmodel.native-coverage/v1','state':'verified','run_id':receipt['run_id'],
 'expected_frames':2,'completed_batches':2,'batch_size':1,'observed_frames':2,
 'selected_frames':receipt['selected_frames'],'loader_preflight_sha256':hashlib.sha256((root/'model-preflight.json').read_bytes()).hexdigest(),
 'evidence':'native eval tqdm reached 100% 2/2, exit_code=0, frozen 2-item loader and batch size 1; no filtering flags',
 'console_sha256':hashlib.sha256((root/'eval-console.log').read_bytes()).hexdigest()}
(root/'coverage_receipt.json').write_text(json.dumps(coverage,indent=2)+'\n')
result=collect_model(root,'v3.2.2',expected_frames=2,log_file=logs[0])
result['coverage']['observed']['model_samples']=2
result['collection']['issues']=[i for i in result['collection']['issues'] if i['code']!='unknown_model_coverage']
result['provenance'].update(native_run=receipt,native_coverage=coverage,model_preflight=preflight)
result['collection']['status']='complete' if not any(i.get('severity')=='error' for i in result['collection']['issues']) else 'partial'
(root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':receipt['state'],'seconds':receipt['duration_seconds'],'frames':2,
 'metrics':len(result['metrics']),'summary':[m for m in result['metrics'] if m['name'] in ['occ.miou','occ.iou2','pillar.miou','pillar.iou2']]}))
