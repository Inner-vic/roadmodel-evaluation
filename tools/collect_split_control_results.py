import hashlib,io,json,zipfile
from pathlib import Path
from model_space_transport import run
root=Path('evidence/model-evaluation/split-gt-audit-20261027');dest=root/'model-control-01';assert not dest.exists();dest.mkdir()
remote='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/split-gt-control-20261027'
code="""import pathlib,zipfile,io,sys
b=pathlib.Path(sys.argv[1]);out=io.BytesIO()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in (b/'run-01').iterdir():
  if p.is_file():z.write(p,'run-01/'+p.name)
 for n in ['workflow-console.log','inputs-manifest.json','launch.json','run_split_control.py']:z.write(b/n,n)
sys.stdout.buffer.write(out.getvalue())
"""
raw=run(['python3','-c',code,remote],timeout=90)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for n in z.namelist():
        target=(dest/n).resolve();assert target.is_relative_to(dest.resolve()) and not target.exists()
        assert z.getinfo(n).file_size<16*1024*1024;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
r=json.loads((dest/'run-01/receipt.json').read_text());assert r['state']=='succeeded' and r['new_forward_frames']==2 and r['new_training_runs']==0
for f in r['outputs']:assert hashlib.sha256((dest/'run-01'/f['path']).read_bytes()).hexdigest()==f['sha256']
assert (dest/'run_split_control.py').read_bytes()==Path('tools/run_split_control_remote.py').read_bytes()
(root/'model-control-collection.json').write_text(json.dumps({'result':'hash_verified','files':len(z.namelist()),'new_forward_frames':2,'new_training_runs':0,'receipt':str(dest/'run-01/receipt.json')},indent=2)+'\n')
print(json.dumps({'state':'collected','files':len(z.namelist()),'new_forward_frames':2}))
