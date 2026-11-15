"""Inventory only referenced model directories; fetch small text configs only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path,PurePosixPath

ROOT=Path('evidence/model-evaluation/model-reference-20261027')
integration=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
connection=next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 import asset
 cfg=asset.config(env='prod',bearer_file=connection['settings']['asset_bearer_file'])
for version in ['26-09-08-2','26-09-08-1']:
 output={'model_set':'roadmodel-version-release','version':version,'checkpoint_downloaded':False,'files':[],'text_downloads':[]}
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   resource=asset.resource('models/roadmodel-version-release/versions/'+version,config=cfg)
   storage=resource._get_s3_path(use='download');prefix=storage['path'].rstrip('/')+'/'
   expected=json.loads((ROOT/('model-version-2.json' if version.endswith('-2') else 'model-version-1.json')).read_text())['data']['spec']['dir']['path']
   if not expected.endswith(prefix):raise ValueError('Unexpected model storage prefix')
   client=resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
   for page in client.get_paginator('list_objects_v2').paginate(Bucket=storage['bucket'],Prefix=prefix):
    for obj in page.get('Contents',[]):
     relative=obj['Key'][len(prefix):]
     if not relative or relative.endswith('/'):continue
     path=PurePosixPath(relative)
     if path.is_absolute() or '..' in path.parts:raise ValueError('Unexpected model object path')
     row={'path':relative,'bytes':obj['Size'],'etag':obj.get('ETag','').strip('"')}
     output['files'].append(row)
     if obj['Size']<=2*1024*1024 and path.suffix.lower() in ['.py','.json','.yaml','.yml','.txt','.sh','.md']:
      local=ROOT/'asset-text'/version/relative;local.parent.mkdir(parents=True,exist_ok=True)
      response=client.get_object(Bucket=storage['bucket'],Key=obj['Key']);body=response['Body'].read(2*1024*1024+1)
      if len(body)!=obj['Size']:raise ValueError('Model text object size mismatch')
      # Keep exact config bytes but never print config bodies from an uninspected artifact.
      if local.exists() and local.read_bytes()!=body:raise ValueError('Existing model config differs from source')
      if not local.exists():local.write_bytes(body)
      output['text_downloads'].append({'path':relative,'local_path':str(local),'sha256':hashlib.sha256(body).hexdigest()})
     if len(output['files'])>5000:raise ValueError('Model inventory exceeds bounded scope')
  output['state']='inventoried'
 except Exception as error:
  output.update(state='failed',error_type=type(error).__name__)
 (ROOT/('files-'+version+'.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(output,ensure_ascii=True)[:16000],flush=True)
