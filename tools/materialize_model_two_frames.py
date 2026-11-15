"""Reuse only two cached workstation frames; never fetch the full dataset."""
import json,hashlib
from pathlib import Path
from sync_requested_eval import ssh,REMOTE_ROOT

root=Path('evidence/model-evaluation/model-reference-20261027')
link=json.loads((root/'two-frame-data-link.json').read_text())
inventory=json.loads(Path('evidence/requested-dataset/eval135-file-inventory.json').read_text())['files']
scene=next(iter(link['eval']))
rows=[]
for row in inventory:
 p=row['path'];parts=p.split('/')
 if parts[0]!=scene:continue
 if Path(p).name=='sensor_param.json':rows.append(row)
 elif len(parts)==4 and parts[2] in ['18','19']:
  name=parts[-1]
  if name.endswith('_f_ca_120.jpg') or name.startswith(('lidar_original_','T_frame_','occ_frame_')): rows.append(row)
assert len(rows)==9,[(r['path'],r['size']) for r in rows]
receipt={'source':'existing_workstation_cache','source_root':REMOTE_ROOT,'files':[]}
for row in rows:
 target=root/'two-frame-input'/row['path']
 if not target.exists():
  data=ssh(['python3','-c','import sys,pathlib;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',REMOTE_ROOT+'/'+row['path']])
  assert len(data)==row['size']
  # These bounded files all have a single-part object ETag.
  assert hashlib.md5(data).hexdigest()==row['etag'].strip('"'),row['path']
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 data=target.read_bytes()
 assert len(data)==row['size'] and hashlib.md5(data).hexdigest()==row['etag'].strip('"')
 receipt['files'].append({'path':row['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'etag':row['etag']})
receipt['bytes']=sum(r['bytes'] for r in receipt['files'])
(root/'two-frame-cache-reuse-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
