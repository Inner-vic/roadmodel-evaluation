"""Recompute native metrics from verified retained predictions; zero inference."""
import pathlib,json,os,sys,hashlib,logging,time,datetime,math,shutil
import numpy as np
import torch
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
old=base/'trajectory-01';outdir=base/'trajectory-01-recovered'
assert not outdir.exists(),'Preserve earlier recovery evidence'
outdir.mkdir()
launch=json.loads((old/'launch.json').read_text());prior=json.loads((old/'execution_receipt.json').read_text())
assert prior['state']=='failed' and prior['exit_code']==1
assert 'Out of range float values are not JSON compliant: nan' in (old/'eval-console.log').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={x['path']:x['sha256'] for x in prior['outputs']}
for i in range(launch['expected_frames']):assert sha(old/f'frame-{i}.npz')==expected[f'frame-{i}.npz']
os.environ.update(launch['environment']);os.chdir(launch['source_root']);sys.path.insert(0,launch['source_root'])
from mmengine import Config
from misc.traj_eval_utils import new_traj_eval_stats,update_traj_eval_stats,log_traj_eval_stats
cfg=Config.fromfile(launch['py_config'],lazy_import=False)
logger=logging.getLogger('retained_trajectory');logger.setLevel(logging.INFO);logger.addHandler(logging.FileHandler(outdir/'trajectory.log'))
start=time.monotonic();stats=new_traj_eval_stats();frames=[]
for i,(scene,index) in enumerate(launch['selected_frames']):
 with np.load(old/f'frame-{i}.npz',allow_pickle=False) as stored:
  result={k:torch.from_numpy(stored[k]) for k in stored.files}
 assert all(torch.isfinite(v).all() for v in result.values())
 before=stats['label_total'];update_traj_eval_stats(result,{},cfg,stats)
 assert stats['label_total']>before
 shutil.copy2(old/f'frame-{i}.npz',outdir/f'frame-{i}.npz')
 frames.append({'scene':scene,'frame_index':index,'ordinal':i,'valid_points':stats['label_total']-before,
                'prediction_file':f'frame-{i}.npz','sha256':expected[f'frame-{i}.npz'],'gt_source':'mesh'})
metrics=log_traj_eval_stats(stats,logger,label_str=cfg.label_str,mask_labels=cfg.mask_labels,head_mode=cfg.HEAD_MODE,
                           z_grid_size=cfg.TRAJ_Z_GRID_SIZE,traj_head_mode=cfg.TRAJ_HEAD_MODE)
undefined=[]
def clean(value,path=''):
 if isinstance(value,float) and not math.isfinite(value):undefined.append(path);return None
 if isinstance(value,dict):return {str(k):clean(v,path+'/'+str(k)) for k,v in value.items()}
 if isinstance(value,list):return [clean(v,path+'/'+str(i)) for i,v in enumerate(value)]
 return value
result={'schema':'roadmodel.native-trajectory/v1','source_commit':launch['source_commit'],'run_id':launch['run_id']+'-postprocess-01',
 'source_inference_run_id':launch['run_id'],'gt_source':'mesh','metrics':clean(metrics,'/metrics'),'raw_counts':stats,
 'undefined_native_fields':undefined,'frames':frames,'num_frames':len(frames),'label_str':list(cfg.label_str),
 'pc_range':list(cfg.pc_range),'z_bins':cfg.D,'voxel_size':[cfg.X_gride_size,cfg.Y_gride_size,cfg.TRAJ_Z_GRID_SIZE],
 'evaluation_helpers':['misc/traj_payload.py','misc/traj_eval_utils.py'],'fixed_noise_seed':cfg.DENSE_SURFACE_EVAL_NOISE_SEED,
 'ddim_steps':cfg.DENSE_SURFACE_DDIM_STEPS,'new_inference_frames':0}
(outdir/'trajectory_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for name in ['model-preflight.json','launch.json']:shutil.copy2(old/name,outdir/name)
shutil.copy2(old/'execution_receipt.json',outdir/'source_inference_receipt.json')
receipt={k:v for k,v in prior.items() if k!='outputs'}
receipt.update(state='succeeded',exit_code=0,run_id=result['run_id'],source_inference_run_id=launch['run_id'],
 source='retained_trajectory_postprocess',source_inference_receipt_sha256=sha(old/'execution_receipt.json'),
 source_inference_exit_code=1,source_failure_stage='JSON serialization after both saved forward outputs',
 started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),duration_seconds=time.monotonic()-start,
 new_inference_frames=0,completed_prediction_frames=len(frames),source_inference_seconds=prior['duration_seconds'])
receipt['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(outdir/'execution_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
print(json.dumps({'state':'succeeded','frames':len(frames),'new_inference_frames':0,'metrics':result['metrics'],'undefined_native_fields':undefined}))
