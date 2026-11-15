"""Freeze a small matched diagnostic sample, reuse cached GT, fetch GT only."""
import hashlib,json,re
from pathlib import Path
from head_data_asset import client_for_reference,read_object
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
d=json.loads((root/'data_link_V151_train7771_eval138_v6.json').read_text(encoding='utf-8'))
shared='HLX33B166R1997378';eval_scene=next(iter(d['eval']))
train_names=list(d['joint_train']);ev_names=list(d['eval'])
near=sorted([s for s in train_names if shared in s and '2026-07-07_' in s],key=lambda x:abs(int(x.split('2026-07-07_')[1][:8].replace('-',''))-100925))[:3]
if len(near)<3:near+=sorted(s for s in train_names if shared in s and s not in near)[:3-len(near)]
other=next(s for s in train_names if 'HLX34B168T1300197' in s)
unseen=next(s for s in ev_names if 'HLX34B167T1300224' in s)
evother=next(s for s in ev_names if shared in s and s!=eval_scene and '2026-07-07_' in s)
choices=[('train',s,'same vehicle and collection day, nearest collection time to existing failed validation scene') for s in near]+[('train',other,'one additional training vehicle control'),('validation',eval_scene,'reuse existing failed cement validation frame'),('validation',evother,'same vehicle/day validation control'),('validation',unseen,'validation vehicle absent from training split')]
plan=[]
for kind,scene,reason in choices:
    info=d['joint_train' if kind=='train' else 'eval'][scene];frames=sorted(info['frames'],key=lambda f:f['frame_index']);f=frames[0] if scene==eval_scene else frames[len(frames)//2]
    plan.append({'split':kind,'scene':scene,'frame':f,'reason':reason})
planpath=root/'gt-sample-plan.json'
if planpath.exists():assert json.loads(planpath.read_text(encoding='utf-8'))==plan
else:planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
client,st=client_for_reference();files=[];new_bytes=0
for item in plan:
    scene=item['scene'];f=item['frame'];fr=str(f['frame_index'])
    for key in ['height_field_file','occ_file']:
        rel='DataReorganized-V151/'+scene+'/dataset/'+fr+'/'+f[key]
        cache=Path('evidence/model-evaluation/remaining-heads-20261027/input')/rel
        target=root/'GT'/rel
        if cache.exists():local=cache;origin='existing_verified_input'
        elif target.exists():local=target;origin='audit_cache'
        else:
            raw,meta=read_object(client,st,rel,limit=8*1024*1024)
            new_bytes+=len(raw);assert new_bytes<=64*1024*1024
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);local=target;origin='scoped_GT_download'
        raw=local.read_bytes();files.append({'split':item['split'],'scene':scene,'frame_index':f['frame_index'],'kind':key,'asset_path':rel,'local_path':str(local),'origin':origin,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
receipt={'selection':'matched diagnostic sample; not random or representative of full class distribution','scene_count':len(plan),'frame_count':len(plan),'new_bytes':new_bytes,'files':files,'new_inference_frames':0,'new_training_runs':0}
(root/'gt-sample-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'scenes':[{'split':p['split'],'scene':p['scene'],'frame':p['frame']['frame_index']} for p in plan],'new_bytes':new_bytes,'reused_files':sum(x['origin']=='existing_verified_input' for x in files)},ensure_ascii=False,indent=2))
