"""Download only the two explicitly referenced checkpoints, with reuse and hashes."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime,timezone

root=Path('evidence/model-evaluation/model-reference-20261027')
integration=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
connection=next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 import asset
 cfg=asset.config(env='prod',bearer_file=connection['settings']['asset_bearer_file'])
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 inventory=json.loads((root/('files-'+version+'.json')).read_text())
 item=next(f for f in inventory['files'] if f['path']==filename)
 local=root/'weights'/version/filename;local.parent.mkdir(parents=True,exist_ok=True)
 receipt_path=local.with_suffix('.receipt.json')
 if local.exists():
  h=hashlib.file_digest(local.open('rb'),'sha256').hexdigest()
  if local.stat().st_size!=item['bytes'] or not receipt_path.exists() or json.loads(receipt_path.read_text())['sha256']!=h:
   raise ValueError('Existing checkpoint is not verified; refusing overwrite')
  print(json.dumps({'version':version,'state':'reused','bytes':item['bytes'],'sha256':h}),flush=True);continue
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  resource=asset.resource('models/roadmodel-version-release/versions/'+version,config=cfg)
  storage=resource._get_s3_path(use='download');prefix=storage['path'].rstrip('/')+'/'
  client=resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
  response=client.get_object(Bucket=storage['bucket'],Key=prefix+filename)
  if response['ContentLength']!=item['bytes'] or response['ETag'].strip('"')!=item['etag']:
   raise ValueError('Model object changed after inventory')
  temporary=local.with_suffix('.partial');h=hashlib.sha256();size=0
  with temporary.open('xb') as stream:
   for chunk in iter(lambda:response['Body'].read(1024*1024),b''):
    stream.write(chunk);h.update(chunk);size+=len(chunk)
  if size!=item['bytes']:raise ValueError('Checkpoint download incomplete')
  temporary.rename(local)
 record={'version':version,'filename':filename,'local_path':str(local.resolve()),'bytes':size,'sha256':h.hexdigest(),
         'etag':item['etag'],'state':'downloaded_and_hashed','created_at':datetime.now(timezone.utc).isoformat()}
 receipt_path.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
