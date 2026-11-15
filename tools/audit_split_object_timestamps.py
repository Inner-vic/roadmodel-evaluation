"""Read object version/timestamp metadata to qualify historical split attribution."""
import json
from pathlib import Path
from head_data_asset import client_for_reference
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
client,st=client_for_reference();prefix=st['path'].rstrip('/')+'/'
paths=['DataReorganized-V151/data_link_V151_train7771_eval138_v6.json']+[f['asset_path'] for f in json.loads((root/'gt-sample-receipt.json').read_text(encoding='utf-8'))['files']]
rows=[]
for p in paths:
    m=client.head_object(Bucket=st['bucket'],Key=prefix+p)
    rows.append({'path':p,'bytes':m['ContentLength'],'etag':m.get('ETag','').strip('"'),'last_modified':m['LastModified'].isoformat(),'version_id':m.get('VersionId')})
(root/'object-timestamps.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
