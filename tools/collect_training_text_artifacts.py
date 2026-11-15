"""Inventory the exact original-task output; fetch bounded text evidence only."""
import contextlib,hashlib,io,json
from pathlib import Path,PurePosixPath
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
integration=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
conn=next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
version='26-09-07-e7nn'
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    import asset
    res=asset.resource('models/test-gzw/versions/'+version,config=asset.config(env='prod',bearer_file=conn['settings']['asset_bearer_file']))
    st=res._get_s3_path(use='download');client=res._s3_client.get_s3_client(endpoint=st['endpoint'])
    prefix=st['path'].rstrip('/')+'/'
    rows=[];downloads=[]
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=st['bucket'],Prefix=prefix):
        for obj in page.get('Contents',[]):
            rel=obj['Key'][len(prefix):];path=PurePosixPath(rel)
            if not rel or rel.endswith('/'):continue
            assert not path.is_absolute() and '..' not in path.parts
            row={'path':rel,'bytes':obj['Size'],'etag':obj.get('ETag','').strip('"')};rows.append(row)
            assert len(rows)<=500
            text=(path.suffix.lower() in ['.log','.json','.csv','.txt','.yaml','.yml','.py','.md'] or path.name.startswith('events.out.tfevents.'))
            if text and obj['Size']<=16*1024*1024:
                target=root/'original-output'/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raw=target.read_bytes();assert len(raw)==obj['Size']
                else:
                    raw=client.get_object(Bucket=st['bucket'],Key=obj['Key'])['Body'].read(16*1024*1024+1)
                    assert len(raw)==obj['Size'];target.write_bytes(raw)
                downloads.append({**row,'local_path':str(target),'sha256':hashlib.sha256(raw).hexdigest()})
receipt={'asset':'models/test-gzw/versions/'+version,'source_task_uid':'8d226ad5-e769-411b-9c72-57f9dd822c0e','files':rows,'downloads':downloads,'weights_downloaded':False}
(root/'original-output-inventory.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(rows),'downloads':downloads,'other_files':[r for r in rows if not r['path'].endswith('.pth')]},ensure_ascii=False))
