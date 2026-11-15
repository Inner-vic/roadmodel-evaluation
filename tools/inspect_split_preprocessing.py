"""Static audit of the exact retained resolved training configuration and source."""
import ast,collections,hashlib,json,re
from pathlib import Path
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
source=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
cfgfile=Path('evidence/model-evaluation/training-semantic-audit-20261027/original-output/logs/3-0-0-temporal-dit.py')
tree=ast.parse(cfgfile.read_text(encoding='utf-8'))
assert all(isinstance(n,ast.Assign) for n in tree.body)
calls=collections.Counter(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call));print('config call names',dict(calls))
assert set(calls)<={'dict','range','float'}
scope={};exec(compile(tree,str(cfgfile),'exec'),{'__builtins__':{},'dict':dict,'range':range,'float':float},scope)
train=scope['train_dataset_config'];val=scope['val_dataset_config']
diff={k:{'train':train.get(k),'validation':val.get(k)} for k in set(train)|set(val) if train.get(k)!=val.get(k)}
out={'config_sha256':hashlib.sha256(cfgfile.read_bytes()).hexdigest(),'train':train,'validation':val,'differences':diff,'train_loader':scope.get('train_loader'),'val_loader':scope.get('val_loader'),'TRAJ':{k:v for k,v in scope.items() if k.startswith('TRAJ_')},'loss_input_conversion':scope.get('loss_input_convertion')}
(root/'preprocessing-config-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'differences':diff,'TRAJ':out['TRAJ']},ensure_ascii=False,indent=2))
d=json.loads((root/'data_link_V151_train7771_eval138_v6.json').read_text(encoding='utf-8'))
split={}
for kind,keys in [('train',['train','joint_train']),('validation',['eval','joint_eval'])]:
    scenes={}
    for k in keys:scenes.update(d[k])
    entries=[]
    for name,info in scenes.items():
        rows=sorted(info['frames'],key=lambda x:x['frame_index']);kept=(len(rows)//5)*5
        entries.append({'scene':name,'listed_frames':len(rows),'loader_entries':kept,'vehicle':name.split('_')[-1], 'collection_month':re.search(r'(20\d\d-\d\d)-\d\d_',name)[1] if re.search(r'(20\d\d-\d\d)-\d\d_',name) else 'unknown'})
    split[kind]={'scenes':len(scenes),'listed_frames':sum(e['listed_frames'] for e in entries),'loader_entries':sum(e['loader_entries'] for e in entries),'by_vehicle':dict(collections.Counter(e['vehicle'] for e in entries)),'by_month':dict(collections.Counter(e['collection_month'] for e in entries)),'scene_info_keys':dict(collections.Counter(k for v in scenes.values() for k in v)),'scenes_details':entries}
(root/'effective-splits.json').write_text(json.dumps(split,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='scenes_details'} for k,v in split.items()},ensure_ascii=False,indent=2))
