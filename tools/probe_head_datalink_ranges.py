import json
from pathlib import Path
from head_data_asset import client_for_reference
client,st=client_for_reference();prefix=st['path'].rstrip('/')+'/'
out=[]
for rel in ['DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json','DataReorganized-V151/data_link_V151_train7771_eval138_v6.json']:
 meta=client.head_object(Bucket=st['bucket'],Key=prefix+rel)
 row={'path':rel,'bytes':meta['ContentLength'],'etag':meta['ETag'].strip('"')}
 for name,rng in [('head','bytes=0-4095'),('tail','bytes=-4096')]:
  obj=client.get_object(Bucket=st['bucket'],Key=prefix+rel,Range=rng)
  raw=obj['Body'].read(4097);assert len(raw)<=4096
  row[name]=raw.decode('utf-8',errors='replace')
 out.append(row)
Path('evidence/model-evaluation/remaining-heads-20261027/datalink-range-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=True)[:11000])
