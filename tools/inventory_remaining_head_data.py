"""Read only the two datalinks and their parent listings from the referenced asset."""
import contextlib,io,json,hashlib
from pathlib import Path
root=Path('evidence/model-evaluation/remaining-heads-20261027');root.mkdir(parents=True,exist_ok=True)
integration=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
conn=next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 import asset
 res=asset.resource('datasets/smartchassisdata-v3/versions/26-08-14-0',config=asset.config(env='prod',bearer_file=conn['settings']['asset_bearer_file']))
 st=res._get_s3_path();client=res._s3_client.get_s3_client(endpoint=st['endpoint'])
 prefix=st['path'].rstrip('/')+'/'
 out={'asset':'datasets/smartchassisdata-v3/versions/26-08-14-0','prefix':prefix,'top_level':[], 'datalinks':[]}
 top=client.list_objects_v2(Bucket=st['bucket'],Prefix=prefix,Delimiter='/',MaxKeys=1000)
 out['top_level']=[p['Prefix'][len(prefix):] for p in top.get('CommonPrefixes',[])]+[p['Key'][len(prefix):] for p in top.get('Contents',[])]
 for rel in ['DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json','DataReorganized-V151/data_link_V151_train7771_eval138_v6.json']:
  target=root/'asset-text'/rel
  if target.exists():raw=target.read_bytes()
  else:
   response=client.get_object(Bucket=st['bucket'],Key=prefix+rel)
   assert response['ContentLength']<=120*1024*1024,'Datalink exceeds bounded metadata size'
   raw=response['Body'].read(120*1024*1024+1)
   assert len(raw)==response['ContentLength']
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  data=json.loads(raw)
  summary={}
  for key,val in data.items():
   summary[key]={'type':type(val).__name__,'size':len(val)}
   if isinstance(val,dict) and val:
    name=next(iter(val));summary[key]['first_key']=name;summary[key]['first_value']=val[name] if isinstance(val[name],list) else str(val[name])[:5000]
   elif isinstance(val,list):summary[key]['first_items']=val[:2]
  out['datalinks'].append({'path':rel,'local_path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'summary':summary})
(root/'referenced-dataset-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=True)[:15000])
