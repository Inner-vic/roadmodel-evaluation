"""Materialize only two frozen training-frame inputs for a diagnostic forward."""
import hashlib,json
from pathlib import Path
from head_data_asset import client_for_reference,read_object
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
plan=json.loads((root/'gt-sample-plan-corrected.json').read_text(encoding='utf-8'))[:2]
gt=json.loads((root/'gt-sample-receipt.json').read_text(encoding='utf-8'))
client,st=client_for_reference();files=[];new=0;subset={k:{} for k in ['train','eval','joint_train','joint_eval','vision_train','vision_eval']}
for item in plan:
    scene=item['scene'];f=item['frame'];fr=str(f['frame_index']);base='DataReorganized-V151/'+scene+'/dataset/'
    subset['eval'][scene]={'scene_name':scene,'frames':[f]}
    for suffix in [fr+'/'+fr+'_f_ca_120.jpg',fr+'/lidar_original_frame_'+fr+'_'+str(f['time_stamp'])+'.npy',fr+'/T_frame_'+fr+'_to_StartVCS.npy','sensor_param.json']:
        rel=base+suffix;target=root/'model-input'/rel
        if target.exists():raw=target.read_bytes();reuse=True
        else:
            raw,meta=read_object(client,st,rel,limit=12*1024*1024);new+=len(raw);assert new<=32*1024*1024
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);reuse=False
        files.append({'path':rel,'local_path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'reused':reuse})
    for row in gt['files']:
        if row['scene']==scene and row['frame_index']==f['frame_index']:files.append({'path':row['asset_path'],'local_path':row['local_path'],'bytes':row['bytes'],'sha256':row['sha256'],'reused':True})
subset['joint_eval']=subset['eval'];(root/'two-training-frames.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt={'selected_frames':[[x['scene'],x['frame']['frame_index']] for x in plan],'files':files,'new_bytes':new,'expected_frames':2,'training':False,'mode':'model.eval, validation image preprocessing, exact retained straight XY'}
(root/'two-frame-input-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'new_bytes':new,'files':len(files),'selected_frames':receipt['selected_frames']},ensure_ascii=False))
