"""Test the strict binder, then try only the two real candidates in LPAI."""
import io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-frame-binding-20261107'

def stage():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name in ('pcm_eval/vision_gt_binding.py','pcm_eval/lpai_native_model.py','tests/test_vision_gt_binding.py','tests/test_lpai_native_model.py'):
            tar.add(ROOT/name,arcname=name)
    proof=remote('''import pathlib,sys,shutil,io,tarfile,json,ast,subprocess
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/vision-frame-binding-20261107/candidate-v3')
assert not stage.exists(),'Preserve prior test candidate'
stage.mkdir(parents=True)
for folder in ('pcm_eval','schemas','configs','tests'):
 shutil.copytree(release/folder,stage/folder,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();target=(stage/member.name).resolve();target.relative_to(stage);raw=tar.extractfile(member).read();ast.parse(raw);target.write_bytes(raw)
p=subprocess.run([sys.executable,'-m','unittest','tests.test_vision_gt_binding','tests.test_lpai_native_model','-q'],cwd=stage,capture_output=True,text=True,timeout=60)
(stage.parent/'tests-v3.log').write_text(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr)[-2000:]
print(json.dumps({'state':'TESTED','candidate':str(stage),'tests':(p.stdout+p.stderr)[-400:],'production_changed':False,'new_inference_frames':0}))
''',stream.getvalue(),timeout=80)
    (OUT/'candidate-tests-v3.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

def validate():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name in ('__init__.py','common.py','vision_gt_binding.py'):
            tar.add(ROOT/'pcm_eval'/name,arcname='pcm_eval/'+name)
    code='''import pathlib,sys,tarfile,io,json,hashlib
target=pathlib.Path('/lpai/roadmodel-evaluation/GT-binding-validation-20261107')
assert not (target/'validation.json').exists(),'Do not overwrite completed verification'
target.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(target/m.name).resolve();p.relative_to(target);assert m.isfile();raw=tar.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==raw or (m.name=='pcm_eval/__init__.py' and p.read_bytes()==b''),'Unexpected candidate code change'
  p.write_bytes(raw)
print(json.dumps({'state':'INSTALLED','root':str(target)}))
'''
    print(run(['python3','-c',code],stream.getvalue(),timeout=30).decode())
    local=json.loads((OUT/'candidate-binding.json').read_text());probe=json.loads((OUT/'original-GT-probe.json').read_text())
    required=[r['key'] for r in json.loads((ROOT/'configs/report_template.json').read_text(encoding='utf-8'))['model_accuracy']['vision_tasks']]
    candidates=[]
    for row,p in zip(local['selected'],probe['rows']):
        original=next(x for x in p['historical_candidates'] if x['label_exists'] and x['image_exists'])
        candidates.append({'scene':row['scene'],'frame_index':row['frame']['frame_index'],
            'image':p['submitted_image']['path'],'image_sha256':p['submitted_image']['sha256'],
            'source_image':original['image']['path'],'source_image_sha256':original['image']['sha256'],
            'source_label':original['label']['path'],'source_label_sha256':original['label']['sha256'],
            'origin':{'task_uid':local['original_training_task'],'dataset_ref':'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/smartchassisdata-v3?version=26-08-14-0'}})
    code='''import pathlib,sys,json
root=pathlib.Path('/lpai/roadmodel-evaluation/GT-binding-validation-20261107');sys.path.insert(0,str(root))
from pcm_eval.vision_gt_binding import prepare,BindingRejected
from pcm_eval.common import digest
payload=json.load(sys.stdin);data=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1');index=data/'data_link_26-10-20-1_train859_eval135_v7.json'
rows=[]
for i,candidate in enumerate(payload['candidates']):
 output=root/('frame-'+str(candidate['frame_index']))
 try:
  receipt=prepare(dataset_root=data,index_path=index,index_sha256=digest(index),candidates=[candidate],
     approved_GT_roots=['/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029/vision-input'],required_tasks=payload['required'],output=output)
  rows.append({'scene':candidate['scene'],'frame_index':candidate['frame_index'],'state':receipt['state'],'receipt':receipt})
 except BindingRejected as error:
  rows.append({'scene':candidate['scene'],'frame_index':candidate['frame_index'],'state':'BLOCKED','code':error.code,'detail':error.detail,'derived_index_created':output.exists()})
proof={'dataset_version':'26-10-20-1','checked_frames':len(rows),'rows':rows,'new_inference_frames':0,'scene_tags_used':False,'dataset_replaced':False}
(root/'validation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
    result=json.loads(run(['python3','-c',code],json.dumps({'candidates':candidates,'required':required}).encode(),timeout=60))
    (OUT/'real-small-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':stage() if len(sys.argv)<2 else validate()
