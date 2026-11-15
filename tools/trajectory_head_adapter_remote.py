"""Run only native inference and reuse the pinned task's trajectory evaluation helpers."""
import os,sys,pathlib,json,time,hashlib,logging
import numpy as np
import torch

run_dir=pathlib.Path(sys.argv[1]);launch=json.loads((run_dir/'launch.json').read_text())
source=pathlib.Path(launch['source_root']);os.environ.update(launch['environment']);os.environ['eval']='true'
os.chdir(source);sys.path.insert(0,str(source))
from mmengine import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor
import model
from dataset import get_dataloader
from misc.ckpt_load_utils import overlay_export_heads,_update_aggregator_cuda_kwargs
from misc.traj_payload import build_fixed_traj_payload
from misc.traj_eval_utils import new_traj_eval_stats,update_traj_eval_stats,log_traj_eval_stats

set_random_seed(42)
cfg=Config.fromfile(launch['py_config'],lazy_import=False)
assert cfg.HEAD_MODE=='traj_only' and cfg.DENSE_SURFACE_GT_SOURCE=='mesh'
my_model=build_segmentor(cfg.model)
sd=torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict']
my_model.load_state_dict(sd,strict=True)
overlay_export_heads(my_model,classifier_resume=launch['classifier_checkpoint'])
_update_aggregator_cuda_kwargs(cfg,my_model)
my_model.cuda().eval()
_,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
assert len(loader.dataset)==launch['expected_frames']==2
logger=logging.getLogger('trajectory_single_eval');logger.setLevel(logging.INFO)
fh=logging.FileHandler(run_dir/'trajectory.log');fh.setFormatter(logging.Formatter('%(message)s'));logger.addHandler(fh)
stats=new_traj_eval_stats();frames=[];start=time.monotonic()
with torch.no_grad():
 for i,data in enumerate(loader):
  before=stats['label_total']
  meshes=data.get('mesh_file');assert meshes is not None,'Mesh GT must be present; no fallback is allowed'
  payload=build_fixed_traj_payload(pc_range=cfg.pc_range,max_len_traj=cfg.MAX_TRAJ_LEN,
     traj_step=cfg.TRAJ_STEP,batch_size=1,use_vehicle_model=cfg.TRAJ_USE_VEHICLE_MODEL,
     param_file=cfg.TRAJ_PARAM_FILE,data_dict=data,speed_range=cfg.TRAJ_SPEED_RANGE,
     steer_range=(0.0,0.0),speed_unit=cfg.TRAJ_SPEED_UNIT,steer_unit=cfg.TRAJ_STEER_UNIT)
  data.update(payload)
  for k,v in list(data.items()):
   if isinstance(v,torch.Tensor):data[k]=v.cuda()
  imgs=data.pop('img');lidar=[x.cuda() if x is not None else None for x in data.pop('lidar')]
  numbers=[torch.tensor([len(x)],dtype=torch.int32) if x is not None else None for x in lidar]
  result=my_model(imgs=imgs,lidar=lidar,input_lidar_point_num=numbers,metas=data,global_iter=0,train_mode='occ',task_phase=None)
  for key in ['traj_z','traj_label','traj_xy_used','traj_valid_mask','dp3_gt_traj_z_norm','dp3_gt_traj_label']:
   assert key in result and isinstance(result[key],torch.Tensor),key
   assert torch.isfinite(result[key]).all(),key
  update_traj_eval_stats(result,data,cfg,stats)
  assert stats['label_total']>before,'No valid trajectory GT points'
  arrays={k:v.detach().cpu().numpy() for k,v in result.items() if k in ['traj_z','traj_zidx','traj_label','traj_xy_used','traj_valid_mask','dp3_gt_traj_z_norm','dp3_gt_traj_zidx','dp3_gt_traj_label']}
  np.savez_compressed(run_dir/f'frame-{i}.npz',**arrays)
  row={'ordinal':i,'scene':launch['selected_frames'][i][0],'frame_index':launch['selected_frames'][i][1],
       'valid_points':stats['label_total']-before,'gt_source':'mesh','prediction_file':f'frame-{i}.npz'}
  frames.append(row);logger.info('FRAME_COMPLETED '+json.dumps(row))
metrics=log_traj_eval_stats(stats,logger,label_str=cfg.label_str,mask_labels=cfg.mask_labels,
                         head_mode=cfg.HEAD_MODE,z_grid_size=cfg.TRAJ_Z_GRID_SIZE,traj_head_mode=cfg.TRAJ_HEAD_MODE)
undefined=[]
def json_safe(value,path=''):
 import math
 if isinstance(value,float) and not math.isfinite(value):undefined.append(path);return None
 if isinstance(value,dict):return {str(k):json_safe(v,path+'/'+str(k)) for k,v in value.items()}
 if isinstance(value,list):return [json_safe(v,path+'/'+str(i)) for i,v in enumerate(value)]
 return value
out={'schema':'roadmodel.native-trajectory/v1','source_commit':launch['source_commit'],'run_id':launch['run_id'],
     'gt_source':'mesh','metrics':json_safe(metrics,'/metrics'),'undefined_native_fields':undefined,
     'raw_counts':stats,'frames':frames,'num_frames':len(frames),
     'inference_seconds':time.monotonic()-start,'pc_range':list(cfg.pc_range),'z_bins':cfg.D,
     'voxel_size':[cfg.X_gride_size,cfg.Y_gride_size,cfg.TRAJ_Z_GRID_SIZE],
     'evaluation_helpers':['misc/traj_payload.py','misc/traj_eval_utils.py'],
     'fixed_noise_seed':cfg.DENSE_SURFACE_EVAL_NOISE_SEED,'ddim_steps':cfg.DENSE_SURFACE_DDIM_STEPS}
(run_dir/'trajectory_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'state':'succeeded','num_frames':len(frames),'valid_points':metrics['valid_points'],'metrics':metrics}))
