"""Download only frozen small-sample files absent from the audited caches."""
import json,hashlib
from pathlib import Path
from head_data_asset import client_for_reference,read_object
root=Path('evidence/model-evaluation/remaining-heads-20261027')
scope=json.loads((root/'scope-inventory.json').read_text(encoding='utf-8'))
gpu=json.loads((root/'gpu-cache.json').read_text());work=json.loads((root/'workstation-cache.json').read_text())
assert not gpu['existing_sources'] and '26-08-14-0' not in work['versions'],'Review available cache before cloud transfer'
client,st=client_for_reference();prefix=st['path'].rstrip('/')+'/'
files={}
for listing in scope['listings']:
 for row in listing.get('frame_files',[]):
  name=Path(row['path']).name
  if name.endswith(('.jpg','.npy','.json')):files[row['path']]=row
 if listing['kind']=='trajectory':
  rel=listing['root']+'/'+listing['scene']+'/dataset/sensor_param.json'
  h=client.head_object(Bucket=st['bucket'],Key=prefix+rel)
  files[rel]={'path':rel,'bytes':h['ContentLength'],'etag':h['ETag'].strip('"')}
assert len(files)==19,len(files)
assert sum(x['bytes'] for x in files.values())<64*1024*1024
receipt={'asset':'datasets/smartchassisdata-v3/versions/26-08-14-0','files':[],'new_bytes':0,'reused_bytes':0}
for rel,row in files.items():
 p=root/'input'/rel
 existed=p.exists()
 if existed:raw=p.read_bytes()
 else:
  raw,meta=read_object(client,st,rel,limit=12*1024*1024)
  assert meta['bytes']==row['bytes'] and meta['etag'].lower()==row['etag'].lower()
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 assert len(raw)==row['bytes'] and hashlib.md5(raw).hexdigest().lower()==row['etag'].lower(),rel
 receipt['files'].append(dict(row,sha256=hashlib.sha256(raw).hexdigest(),reused=existed))
 receipt['reused_bytes' if existed else 'new_bytes']+=len(raw)
(root/'input-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'files':len(receipt['files']),'new_bytes':receipt['new_bytes'],'reused_bytes':receipt['reused_bytes']}))
