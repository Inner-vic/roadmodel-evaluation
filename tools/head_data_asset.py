"""Access the previously configured asset identity without printing credentials."""
import contextlib,io,json
from pathlib import Path

def client_for_reference():
 integration=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
 conn=next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  import asset
  res=asset.resource('datasets/smartchassisdata-v3/versions/26-08-14-0',config=asset.config(env='prod',bearer_file=conn['settings']['asset_bearer_file']))
  storage=res._get_s3_path()
  client=res._s3_client.get_s3_client(endpoint=storage['endpoint'])
 return client,storage

def read_object(client,storage,relative,*,limit):
 prefix=storage['path'].rstrip('/')+'/'
 obj=client.get_object(Bucket=storage['bucket'],Key=prefix+relative)
 assert obj['ContentLength']<=limit,'Object exceeds scoped size limit'
 raw=obj['Body'].read(limit+1);assert len(raw)==obj['ContentLength']
 return raw,{'bytes':len(raw),'etag':obj['ETag'].strip('"')}
