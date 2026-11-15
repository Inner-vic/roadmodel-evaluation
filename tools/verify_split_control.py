"""Independent local checks on the two-frame model contrast and original failure."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
root=Path('evidence/model-evaluation/split-gt-audit-20261027');folder=root/'model-control-01';r=json.loads((folder/'run-01/receipt.json').read_text())
def loadcfg(p):
    tree=ast.parse(p.read_text(encoding='utf-8'));assert all(isinstance(x,ast.Assign) for x in tree.body)
    assert all(isinstance(x.func,ast.Name) and x.func.id in ['dict','range','float'] for x in ast.walk(tree) if isinstance(x,ast.Call))
    d={};exec(compile(tree,str(p),'exec'),{'__builtins__':{},'dict':dict,'range':range,'float':float},d);return d
oldroot=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01/trajectory-diagnostic-01')
old=loadcfg(oldroot/'resolved-config.py');new=loadcfg(folder/'run-01/resolved-config.py')
assert old['model']==new['model'],'Model configuration changed'
oldpipe=old['val_dataset_config']['pipeline'];newpipe=new['val_dataset_config']['pipeline']
assert len(oldpipe)==len(newpipe)
pipeline_path_differences=[]
for i,(op,np_) in enumerate(zip(oldpipe,newpipe)):
    assert {k:v for k,v in op.items() if k!='occ_path'}=={k:v for k,v in np_.items() if k!='occ_path'},(i,op,np_)
    if op.get('occ_path')!=np_.get('occ_path'):pipeline_path_differences.append({'transform':i,'field':'occ_path','original':op.get('occ_path'),'control':np_.get('occ_path')})
assert old['val_dataset_config']['data_aug_conf']==new['val_dataset_config']['data_aug_conf']
with np.load(Path('evidence/model-evaluation/remaining-heads-20261027/collected/trajectory-01-recovered/frame-0.npz'),allow_pickle=False) as ref:xy=ref['traj_xy_used'].copy()
rows=[]
for i,row in enumerate(r['frames']):
    path=folder/'run-01'/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    with np.load(path,allow_pickle=False) as d:
        assert np.array_equal(d['traj_xy_used'],xy)
        gt=d['dp3_gt_traj_label'];pr=d['traj_label'];valid=d['traj_valid_mask'].astype(bool)&(gt!=26)
        assert valid.sum()==680
        assert sum(v['support'] for v in row['classes'].values())==680
        for k,v in row['classes'].items():
            m=valid&(gt==int(k));assert m.sum()==v['support'] and (pr[m]==gt[m]).sum()==v['correct']
        assert np.array_equal(d['pred_traj_label_logits'].argmax(-1)[valid],pr[valid])
        logits=d['pred_traj_label_logits'].astype(np.float64)[valid];targets=gt[valid].astype(int);mx=logits.max(1)
        ces=np.log(np.exp(logits-mx[:,None]).sum(1))+mx-logits[np.arange(len(targets)),targets]
        perCE={str(k):float(ces[targets==k].mean()) for k in np.unique(targets)}
        assert abs(float(ces.mean())-row['center_mean_CE'])<1e-5
    rows.append({'scene':row['scene'],'frame_index':row['frame_index'],'valid':680,'classes':row['classes'],'predictions':row['prediction_counts'],'class_mean_CE':perCE})
prior_rows=[]
for i,frame in enumerate([20,21]):
    with np.load(oldroot/f'frame-{i}-native.npz',allow_pickle=False) as d:
        gt=d['result_dp3_gt_traj_label'];pr=d['result_traj_label'];valid=d['result_traj_valid_mask'].astype(bool)&(gt!=26)
        l=d['result_pred_traj_label_logits'].astype(np.float64)[valid];target=gt[valid].astype(int);mx=l.max(1)
        ce=np.log(np.exp(l-mx[:,None]).sum(1))+mx-l[np.arange(len(l)),target]
        prior_rows.append({'frame_index':frame,'cement_support':int((target==2).sum()),'cement_correct':int((pr[valid][target==2]==2).sum()),'cement_mean_CE':float(ce[target==2].mean()),'XY_same_as_control':bool(np.array_equal(d['result_traj_xy_used'],xy))})
def flatten(x,path=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from flatten(v,path+'.'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from flatten(v,path+f'[{i}]')
    else:yield path,x
water_train=next((root/'model-input/DataReorganized-V151/2026-09-04/2026-04-23_10-16-00_HLX33B166R1997378/dataset').glob('sensor_param.json'))
water_val=Path('evidence/model-evaluation/remaining-heads-20261027/input/DataReorganized-V151/2026-09-08/2026-07-07_10-09-25_HLX33B166R1997378/dataset/sensor_param.json')
t=dict(flatten(json.loads(water_train.read_text())));v=dict(flatten(json.loads(water_val.read_text())))
diff=[{'field':k,'training':t.get(k),'validation':v.get(k)} for k in sorted(set(t)|set(v)) if t.get(k)!=v.get(k)]
traincal=json.loads(water_train.read_text())[0];valcal=json.loads(water_val.read_text())[0]
def qangle(a,b,keys):
    x=np.array([a[k] for k in keys]);y=np.array([b[k] for k in keys]);x/=np.linalg.norm(x);y/=np.linalg.norm(y)
    return float(np.degrees(2*np.arccos(np.clip(abs(x@y),0,1))))
calibration={'camera_intrinsics_equal':traincal['cam']['proj_param']==valcal['cam']['proj_param'],'camera_extrinsic_rotation_difference_deg':qangle(traincal['cam']['q_vcs_cam'],valcal['cam']['q_vcs_cam'],['w','x','y','z']),'lidar_extrinsic_rotation_difference_deg':qangle(traincal['lidar']['q_vcs_lidar'],valcal['lidar']['q_vcs_lidar'],['qw','qx','qy','qz']),'lidar_translation_difference_cm':float(np.linalg.norm(np.array(list(traincal['lidar']['t_vcs_lidar'].values()))-np.array(list(valcal['lidar']['t_vcs_lidar'].values())))*100),'interpretation':'different per-scene calibration metadata; both loaded through same source; no causal swap tested'}
out={'state':'verified','model_config_identical':True,'validation_pipeline_identical_except_input_root':True,'pipeline_path_differences':pipeline_path_differences,'crop_and_normalization_identical':True,'XY_exactly_equal':True,'new_unique_frames':2,'new_full_model_forward_passes':2,'new_training_runs':0,'frames':rows,'retained_validation_frames':prior_rows,'same_vehicle_cement_calibration_differences':diff,'calibration_summary':calibration}
(root/'model-control-verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2)[:5500])
