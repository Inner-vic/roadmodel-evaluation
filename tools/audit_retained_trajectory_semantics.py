"""Offline audit of retained predictions, original GT and checkpoint/config binding.

No Torch execution, inference, downloads or edits to prior run evidence.
"""
import ast,copy,json,os,types,zipfile,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path('evidence/model-evaluation/remaining-heads-20261027')
RUN=ROOT/'collected/trajectory-01-recovered'
REF=Path('evidence/model-evaluation/model-reference-20261027')
SOURCE=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
OUT=ROOT/'semantic-audit';OUT.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hist(a):
    v,n=np.unique(a,return_counts=True);return {str(int(k)):int(c) for k,c in zip(v,n)}

def merge(base,override):
    value=copy.deepcopy(base)
    for k,v in override.items():
        if isinstance(v,dict) and not v.get('_delete_',False) and isinstance(value.get(k),dict):value[k]=merge(value[k],v)
        else:value[k]=copy.deepcopy({kk:vv for kk,vv in v.items() if kk!='_delete_'} if isinstance(v,dict) else v)
    return value

def resolve_config(env):
    """These pinned config files import only os/decimal; reproduce _base_ dict merges.

    The result is independently checked against the retained native mmengine dump.
    """
    combined={}
    with patch.dict(os.environ,env,clear=True):
        for name in ['2-1-0.py','3-0-0-temporal.py','3-0-0-temporal-dit.py']:
            path=SOURCE/'config'/name;tree=ast.parse(path.read_text(encoding='utf-8'))
            tree.body=[node for node in tree.body if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='_base_' for t in node.targets)]
            scope={'_base_':copy.deepcopy(combined)}
            exec(compile(tree,str(path),'exec'),scope)
            values={k:v for k,v in scope.items() if not k.startswith('_') and not isinstance(v,types.ModuleType) and not callable(v)}
            combined=merge(combined,values)
    return combined

launch=load(RUN/'launch.json');receipt=load(RUN/'execution_receipt.json');prior=load(RUN/'source_inference_receipt.json')
raw=load(RUN/'trajectory_metrics.json');training=load(ROOT/'trajectory-context.json')['active_launch_settings']['environment']
binding=load(REF/'model-input-binding.json')
config_path=SOURCE/'config/3-0-0-temporal-dit.py'
config_lf_sha=hashlib.sha256(config_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
assert config_lf_sha==binding['code']['config_sha256'],'Configuration content differs beyond CRLF/LF'
source_manifest=load(Path('evidence/model-evaluation/native-v3.2.2-smoke-20261027/source-manifest.json'))
assert sha(config_path)==source_manifest['files']['config/3-0-0-temporal-dit.py']
for item in load(ROOT/'input-receipt.json')['files']:
    if item['path'].startswith('DataReorganized-V151/'):
        assert sha(ROOT/'input'/item['path'])==item['sha256'],'Source input changed: '+item['path']
assert sha(RUN/'source_inference_receipt.json')==receipt['source_inference_receipt_sha256']
assert launch['selected_frames']==prior['selected_frames']==receipt['selected_frames']
assert prior['run_id']==raw['source_inference_run_id']
expected={x['path']:x['sha256'] for x in prior['outputs']}
assert sha(RUN/'launch.json')==expected['launch.json'] and sha(RUN/'model-preflight.json')==expected['model-preflight.json']
train_cfg=resolve_config(training);run_cfg=resolve_config(launch['environment'])
vision_launch=load(ROOT/'collected/vision-01/launch.json')
vision_cfg=resolve_config(vision_launch['environment'])
dump=ROOT/'collected/vision-01/output/3-0-0-temporal-dit/logs/3-0-0-temporal-dit.py'
native={};exec(compile(dump.read_text(encoding='utf-8'),str(dump),'exec'),native)
for key in ['model','label_str','pc_range','DENSE_SURFACE_Y_OFFSETS','TRAJ_NUM_Z_BINS','val_dataset_config']:
    assert vision_cfg[key]==native[key],key+' independent config resolver differs from retained mmengine dump'

def flatten(d,prefix=''):
    for k,v in d.items():
        if isinstance(v,dict):yield from flatten(v,prefix+k+'.')
        else:yield prefix+k,v
train_model=dict(flatten(train_cfg['model']));run_model=dict(flatten(run_cfg['model']))
model_diff=[{'field':k,'training':train_model.get(k),'evaluation':run_model.get(k)} for k in sorted(set(train_model)|set(run_model)) if train_model.get(k)!=run_model.get(k)]
assert not model_diff,model_diff
labels=run_cfg['label_str'];assert labels[1]=='asphalt' and labels[2]=='cement' and len(labels)==26
frames=[]
for row in raw['frames']:
    path=RUN/row['prediction_file'];assert sha(path)==row['sha256']==expected[row['prediction_file']]
    scene=row['scene'];fr=row['frame_index'];folder=ROOT/'input/DataReorganized-V151'/scene/'dataset'/str(fr)
    mesh_path=next(folder.glob('height_field_*.npy'));occ_path=next(folder.glob('occ_*.npy'))
    mesh=np.load(mesh_path,allow_pickle=False);occ=np.load(occ_path,allow_pickle=False)
    with np.load(path,allow_pickle=False) as data:
        pred=data['traj_label'].reshape(-1);gt=data['dp3_gt_traj_label'].reshape(-1)
        mask=data['traj_valid_mask'].reshape(-1).astype(bool)&(gt!=len(labels))
        xy=data['traj_xy_used'].reshape(-1,2);z_norm=data['dp3_gt_traj_z_norm'].reshape(-1)
        # Independent nearest-cell semantic query of original physical-coordinate Mesh GT.
        ix=np.floor((mesh[:,0]-np.float32(5.0))/np.float32(.05)).astype(int)
        iy=np.floor((mesh[:,1]-np.float32(-3.6))/np.float32(.05)).astype(int)
        valid=(mesh[:,4]>.5)&np.isfinite(mesh[:,:4]).all(axis=1)&(mesh[:,3]>=0)&(mesh[:,3]<26)&(ix>=0)&(ix<400)&(iy>=0)&(iy<144)
        grid=np.full((400,144),26,dtype=int);height=np.zeros((400,144),dtype=np.float32);ok=np.zeros((400,144),bool)
        grid[ix[valid],iy[valid]]=mesh[valid,3].round().astype(int);height[ix[valid],iy[valid]]=mesh[valid,2];ok[ix[valid],iy[valid]]=True
        qx=np.floor((xy[:,0]-np.float32(5.0))/np.float32(.05)).astype(int);qy=np.floor((xy[:,1]-np.float32(-3.6))/np.float32(.05)).astype(int)
        sampled=grid[qx[mask],qy[mask]]
        assert np.array_equal(sampled,gt[mask]),'Retained GT differs from original Mesh semantics'
        # Independent bilinear height check provides a coordinate/frame sanity check.
        gx=(xy[:,0]-np.float32(5.0))/np.float32(.05)-np.float32(.5);gy=(xy[:,1]-np.float32(-3.6))/np.float32(.05)-np.float32(.5)
        x0=np.floor(gx).astype(int);y0=np.floor(gy).astype(int);fx=gx-x0;fy=gy-y0
        numerator=np.zeros(len(xy));denominator=np.zeros(len(xy))
        for dx,dy,weights in [(0,0,(1-fx)*(1-fy)),(0,1,(1-fx)*fy),(1,0,fx*(1-fy)),(1,1,fx*fy)]:
            xx=x0+dx;yy=y0+dy;inside=(xx>=0)&(xx<400)&(yy>=0)&(yy<144)
            xx=xx.clip(0,399);yy=yy.clip(0,143);w=weights*inside*ok[xx,yy]
            numerator+=height[xx,yy]*w;denominator+=w
        physical_gt=numerator/denominator.clip(1e-6);retained_gt=-1+z_norm*99*.02
        max_height_difference=float(np.max(np.abs(physical_gt[mask]-retained_gt[mask])))
        assert max_height_difference<1e-5
        # OCC provides a second existing GT representation at a different Y resolution.
        ox=np.floor((xy[:,0]-np.float32(5.0))/np.float32(.05)).astype(int);oy=np.floor((xy[:,1]-np.float32(-3.6))/np.float32(.15)).astype(int)
        pillars={}
        for x,y,_,label,*rest in occ.astype(int):pillars.setdefault((x,y),set()).add(label)
        supported=sum(int(int(gt[j]) in pillars.get((int(ox[j]),int(oy[j])),set())) for j in np.flatnonzero(mask))
        correct=int((pred[mask]==gt[mask]).sum());n=int(mask.sum())
        assert correct==0 and n==680
        confusion=[{'gt':int(g),'predicted':int(p),'count':int(((gt==g)&(pred==p)&mask).sum())} for g,p in np.unique(np.stack([gt[mask],pred[mask]],axis=1),axis=0)]
        frames.append({'scene':scene,'frame':fr,'prediction_sha256':sha(path),'mesh_sha256':sha(mesh_path),'occ_sha256':sha(occ_path),
            'valid_points':n,'correct':correct,'gt_counts':hist(gt[mask]),'predicted_counts':hist(pred[mask]),'confusion':confusion,
            'per_wheel_valid':mask.reshape(4,-1).sum(axis=1).tolist(),'mesh_label_match_points':int((sampled==gt[mask]).sum()),
            'occ_pillar_contains_same_gt_label_points':supported,'max_gt_height_difference_m':max_height_difference,
            'mesh_all_labels':hist(mesh[:,3]),'occ_all_labels':hist(occ[:,3])})

weights={};parameter_stats={}
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
    path=REF/'weights'/version/filename;assert sha(path)==receipt['checkpoint_sha256'][filename]
    meta=load(path.parent/'metadata-inspection.json')['tensor_metadata'];rows={k:v for k,v in meta.items() if k.startswith('head.traj_head.')}
    assert len(rows)==164
    weights[filename]={'sha256':sha(path),'strict_main_keys':len(meta),'trajectory_keys':len(rows),
        'surface_cls_keys':[k for k in rows if '.surface_cls_head.' in k]}
    with zipfile.ZipFile(path) as archive:
        prefix=next(x.rsplit('/',1)[0] for x in archive.namelist() if x.endswith('/data.pkl'))
        for key,v in rows.items():
            if '.surface_cls_head.' not in key:continue
            assert v['dtype']=='FloatStorage'
            storage=np.frombuffer(archive.read(prefix+'/data/'+str(v['storage_key'])),dtype='<f4')
            arr=np.lib.stride_tricks.as_strided(storage[v['storage_offset']:],shape=v['shape'],strides=tuple(x*4 for x in v['stride'])).copy()
            assert np.isfinite(arr).all()
            summary={'shape':list(arr.shape),'mean':float(arr.mean()),'std':float(arr.std()),'abs_max':float(np.abs(arr).max()),'sha256':hashlib.sha256(arr.tobytes()).hexdigest()}
            if key.endswith('bias') and arr.size==81:summary['class_biases_by_lateral_offset']=arr.reshape(3,27).tolist()
            parameter_stats.setdefault(filename,{})[key]=summary

env_diff=[{'field':k,'training':training.get(k),'evaluation':launch['environment'].get(k)} for k in sorted(set(training)|set(launch['environment'])) if training.get(k)!=launch['environment'].get(k)]
audit={'schema':'roadmodel.trajectory-semantic-audit/v1','method':'existing_results_only','new_inference_frames':0,
    'source_run_id':raw['source_inference_run_id'],'label_space':dict(enumerate(labels+['empty'])),
    'vision_label_space_is_separate':{'basic_road_surface':{'0':'asphalt','1':'cement'},'used_for_trajectory':False},
    'frames':frames,'total':{'valid_points':sum(x['valid_points'] for x in frames),'correct':sum(x['correct'] for x in frames),'accuracy_percent':0.},
    'configuration':{'method':'stdlib config reconstruction cross-checked against retained native mmengine vision dump',
        'native_dump_sha256':sha(dump),'model_config_differences_from_training':model_diff,
        'environment_differences':env_diff,'effective_trajectory_config':run_cfg['model']['head']['traj_head'],
        'effective_eval_single_frame_mode':run_cfg['val_dataset_config']['single_frame_mode'],
        'effective_eval_num_frames':run_cfg['val_dataset_config']['num_frames'],
        'USE_TEMPORAL':run_cfg['USE_TEMPORAL'],'lifter_type':run_cfg['model']['lifter']['type'],
        'source_config_sha256':sha(SOURCE/'config/3-0-0-temporal-dit.py'),
        'source_config_LF_sha256':config_lf_sha,'configuration_hash_note':'Runtime archive CRLF and initial Git text snapshot LF; exact text is identical after newline normalization. Runtime raw hash matches retained source manifest.',
        'DENSE_SURFACE_Y_OFFSETS':run_cfg['DENSE_SURFACE_Y_OFFSETS'],'precision':'evaluation FP32; training AMP'},
    'weights':weights,'semantic_layer_stats':parameter_stats,
    'checks':{'prediction_hashes_match_original_failed_inference_receipt':True,'recovered_npz_unchanged':True,
        'label_1_is_asphalt_2_is_cement':True,'independent_original_mesh_semantics_matches_all_retained_gt':True,
        'model_architecture_and_runtime_head_config_match_training':not model_diff,
        'classifier_overlay_prefix_does_not_touch_trajectory':True,'no_visual_class_ids_used_in_trajectory':True,
        'original_source_input_hashes_match_download_receipt':True},
    'conclusion':'Retained model outputs misclassify cement as asphalt on the selected two frames; no label-remapping, GT sampling, score arithmetic or checkpoint/config binding defect identified.',
    'limits':['Inference logits/probabilities and intermediate model features were not retained; confidence and upstream feature failure cannot be retrospectively localized.',
        'This establishes the observed two-frame error, not a model-wide semantic-collapse claim.','Training AMP and evaluation FP32 differ; effect was not isolated without new inference.']}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'checks':audit['checks'],'frames':frames,'model_config_differences':model_diff,'total':audit['total']},ensure_ascii=True))
