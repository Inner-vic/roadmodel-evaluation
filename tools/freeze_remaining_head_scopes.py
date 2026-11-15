import json,re,hashlib
from pathlib import Path
from head_data_asset import client_for_reference
root=Path('evidence/model-evaluation/remaining-heads-20261027')
vision_path=root/'asset-text/DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json'
vision=json.loads(vision_path.read_text(encoding='utf-8'))
vs=list(vision['vision_eval'])[:2]
vision_subset={k:{} for k in vision}
vision_subset['vision_eval']={s:dict(vision['vision_eval'][s],frames=vision['vision_eval'][s]['frames'][:2]) for s in vs}
(root/'vision-subset.json').write_text(json.dumps(vision_subset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
client,st=client_for_reference();prefix=st['path'].rstrip('/')+'/'
rel='DataReorganized-V151/data_link_V151_train7771_eval138_v6.json'
obj=client.get_object(Bucket=st['bucket'],Key=prefix+rel,Range='bytes=0-1048575')
raw=obj['Body'].read(1048577);assert len(raw)<=1048576
text=raw.decode('utf-8');dec=json.JSONDecoder()
start=re.search(r'"eval"\s*:\s*\{',text).end()
key,end=dec.raw_decode(text,start+len(text[start:])-len(text[start:].lstrip()))
pos=text.index(':',end)+1;pos+=len(text[pos:])-len(text[pos:].lstrip())
scene,end=dec.raw_decode(text,pos)
traj_subset={k:{} for k in vision};traj_subset['eval']={key:dict(scene,frames=scene['frames'][:2])};traj_subset['joint_eval']=traj_subset['eval']
(root/'trajectory-subset.json').write_text(json.dumps(traj_subset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
out={'dataset_version':'26-08-14-0','metadata_bytes_read_this_stage':len(raw),
 'trajectory_source':{'path':rel,'etag':obj['ETag'].strip('"'),'range_sha256':hashlib.sha256(raw).hexdigest(),'range':'bytes=0-1048575'},
 'vision_source':{'path':str(vision_path),'sha256':hashlib.sha256(vision_path.read_bytes()).hexdigest()},
 'selected':{'vision':[[s,f['frame_index']] for s,info in vision_subset['vision_eval'].items() for f in info['frames']],
             'trajectory':[[key,f['frame_index']] for f in traj_subset['eval'][key]['frames']]},'listings':[]}
scopes=[('vision',s,vision_subset['vision_eval'][s]['frames']) for s in vs]+[('trajectory',key,traj_subset['eval'][key]['frames'])]
for kind,s,frames in scopes:
 roots=['DataReorganized-VisionHead','DataReorganized-VisionHead-standard'] if kind=='vision' else ['DataReorganized-V151']
 for data_root in roots:
  scene_prefix=prefix+data_root+'/'+s+'/'
  listing=client.list_objects_v2(Bucket=st['bucket'],Prefix=scene_prefix,Delimiter='/',MaxKeys=100)
  record={'kind':kind,'root':data_root,'scene':s,'directories':[x['Prefix'][len(prefix):] for x in listing.get('CommonPrefixes',[])],
          'files':[{'path':x['Key'][len(prefix):],'bytes':x['Size']} for x in listing.get('Contents',[])]}
  for frame in frames:
   for layout in ['labeled','original','dataset']:
    objects=client.list_objects_v2(Bucket=st['bucket'],Prefix=scene_prefix+layout+'/'+str(frame['frame_index'])+'/',MaxKeys=100)
    if objects.get('Contents'):record.setdefault('frame_files',[]).extend({'path':x['Key'][len(prefix):],'bytes':x['Size'],'etag':x['ETag'].strip('"')} for x in objects['Contents'])
  out['listings'].append(record)
(root/'scope-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=True)[:19000])
