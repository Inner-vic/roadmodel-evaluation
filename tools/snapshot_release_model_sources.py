"""Read pinned model source files without changing an existing checkout."""
import hashlib
import json
from pathlib import Path
import subprocess

COMMITS=['9e16e29351d17a726ce6beca6f821de79b7ade40','597278d0f86c6f22548f62fa31b43e1ac16c49f7']
FILES=['eval.py','config/3-0-0-temporal-dit.py','config/2-1-0.py','misc/ckpt_load_utils.py',
       'misc/checkpoint_util.py','misc/cfg_utils.py','dataset/dataset.py','dataset/transform_3d.py','model/head/multi_head_wrapper.py']
for commit in COMMITS:
 root=Path('inspection/release-model-source')/commit
 tree=subprocess.run(['git','-C','inspection/roadmodel-dev-v41','ls-tree','-r','--name-only',commit],capture_output=True,check=True).stdout.decode().splitlines()
 rows=[]
 for name in FILES:
  if name not in tree:continue
  p=root/name
  if not p.exists():
   data=subprocess.run(['git','-C','inspection/roadmodel-dev-v41','show',commit+':'+name],capture_output=True,check=True).stdout
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  rows.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 root.mkdir(parents=True,exist_ok=True)
 (root/'manifest.json').write_text(json.dumps({'commit':commit,'files':rows},indent=2)+'\n')
 print(json.dumps({'commit':commit,'files':len(rows),'root':str(root)}),flush=True)
