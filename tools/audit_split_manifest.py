"""Collect one referenced metadata index; audit split/schema without loading dataset GT."""
import collections,hashlib,json,os,re,sys
from pathlib import Path
from head_data_asset import client_for_reference,read_object
root=Path('evidence/model-evaluation/split-gt-audit-20261027');root.mkdir(parents=True,exist_ok=True)
target=root/'data_link_V151_train7771_eval138_v6.json'
if not target.exists():
    client,st=client_for_reference()
    raw,meta=read_object(client,st,'DataReorganized-V151/'+target.name,limit=160*1024*1024)
    target.write_bytes(raw)
else:raw=target.read_bytes();meta={'bytes':len(raw),'cache_reused':True}
data=json.loads(raw);result={'source':'smartchassisdata-v3/26-08-14-0/DataReorganized-V151/'+target.name,'sha256':hashlib.sha256(raw).hexdigest(),'metadata':meta,'splits':{}}
for split,scenes in data.items():
    fields=collections.Counter();patterns=collections.Counter();fieldsamples={};frames=0
    if not isinstance(scenes,dict):continue
    for scene,info in scenes.items():
        rows=info.get('frames',[]);frames+=len(rows)
        for f in rows:
            fields.update(f.keys())
            for k in ['occ_file','height_field_file']:
                v=f.get(k)
                if v:
                    version=re.search(r'v\d[\d-]*',v);patterns[k+':'+(version[0] if version else 'unversioned')]+=1
                    fieldsamples.setdefault(k,v)
    result['splits'][split]={'scenes':len(scenes),'frames':frames,'field_counts':dict(fields),'GT_filename_versions':dict(patterns),'first_scene':next(iter(scenes),None),'field_samples':fieldsamples}
def ids(split):return {(s,int(f['frame_index'])) for s,i in data.get(split,{}).items() for f in i['frames']}
result['split_overlap']={f'{x}/{y}':{'scene_overlap':len(set(data.get(x,{}))&set(data.get(y,{}))),'frame_overlap':len(ids(x)&ids(y))} for x,y in [('train','eval'),('joint_train','eval'),('joint_train','joint_eval')]}
(root/'manifest-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
