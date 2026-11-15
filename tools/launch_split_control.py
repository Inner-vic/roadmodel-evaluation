import hashlib,io,json,sys,zipfile
from pathlib import Path
from model_space_transport import run,put
root=Path('evidence/model-evaluation/split-gt-audit-20261027');remote='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/split-gt-control-20261027'
inputs=json.loads((root/'two-frame-input-receipt.json').read_text(encoding='utf-8'))
launch=json.loads(Path('evidence/model-evaluation/remaining-heads-20261027/trajectory-launch.json').read_text(encoding='utf-8'))
launch['environment']['DATASET_PATH']=remote+'/input/DataReorganized-V151';launch['environment']['DATA_LINK_FILE']=remote+'/input/two-training-frames.json'
launch.update(run_id='model-v3.2.2-split-control-20261027',selected_frames=inputs['selected_frames'])
manifest=[]
with zipfile.ZipFile(root/'two-frame-transfer.zip','x',zipfile.ZIP_DEFLATED) as z:
    for f in inputs['files']:
        raw=Path(f['local_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256']
        n='input/'+f['path'];z.writestr(n,raw);manifest.append({'path':n,'sha256':f['sha256']})
    raw=(root/'two-training-frames.json').read_bytes();z.writestr('input/two-training-frames.json',raw);manifest.append({'path':'input/two-training-frames.json','sha256':hashlib.sha256(raw).hexdigest()})
    z.writestr('launch.json',json.dumps(launch));z.writestr('inputs-manifest.json',json.dumps(manifest))
put(root/'two-frame-transfer.zip',remote+'/two-frame-transfer.zip');put('tools/run_split_control_remote.py',remote+'/run_split_control.py')
code="""import pathlib,zipfile,sys,subprocess,json
b=pathlib.Path(sys.argv[1]);z=zipfile.ZipFile(b/'two-frame-transfer.zip')
for n in z.namelist():
 p=(b/n).resolve();assert p.is_relative_to(b.resolve());assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
assert not (b/'workflow-console.log').exists()
f=(b/'workflow-console.log').open('w');p=subprocess.Popen([sys.executable,str(b/'run_split_control.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid,'root':str(b)}))
"""
print(run(['python3','-c',code,remote],timeout=60).decode())
