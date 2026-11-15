"""Same two frames only, training-style FP16 autocast control; no training."""
import pathlib,json,os,sys,time,hashlib,traceback
import numpy as np
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
old=base/'heads-20261027/trajectory-01';out=base/'heads-20261027/trajectory-amp-control-01'
assert not out.exists();out.mkdir()
launch=json.loads((old/'launch.json').read_text());os.environ.update(launch['environment']);os.environ['eval']='true'
os.chdir(launch['source_root']);sys.path.insert(0,launch['source_root'])
receipt={'schema':'roadmodel.trajectory-precision-control/v1','state':'preflight','selected_frames':launch['selected_frames'],
    'new_full_model_forward_frames':0,'precision':'torch.cuda.amp.autocast(True), matching original train.py',
    'weights_changed':False,'training_performed':False,'frames':[]}
try:
    import torch,model
    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmseg.models import build_segmentor
    from dataset import get_dataloader
    from misc.ckpt_load_utils import overlay_export_heads,_update_aggregator_cuda_kwargs
    from misc.traj_payload import build_fixed_traj_payload
    set_random_seed(42);cfg=Config.fromfile(launch['py_config'],lazy_import=False)
    net=build_segmentor(cfg.model)
    for name,h in launch['checkpoint_sha256'].items():assert hashlib.sha256((base/'weights'/name).read_bytes()).hexdigest()==h
    net.load_state_dict(torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
    overlay_export_heads(net,classifier_resume=launch['classifier_checkpoint']);_update_aggregator_cuda_kwargs(cfg,net)
    net.cuda().eval();head=net.head.traj_head
    _,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
    assert len(loader.dataset)==2
    captured={}
    def hook(name):
        def fn(module,args,output):
            if isinstance(output,torch.Tensor):captured[name]=output.detach().cpu().numpy()
        return fn
    for name in ['surface_position_pack','surface_lidar_pack','surface_gaussian_attention','surface_semantic_context']:
        head.get_submodule(name).register_forward_hook(hook(name))
    with torch.no_grad():
        for i,data in enumerate(loader):
            captured.clear()
            data.update(build_fixed_traj_payload(pc_range=cfg.pc_range,max_len_traj=cfg.MAX_TRAJ_LEN,traj_step=cfg.TRAJ_STEP,
                batch_size=1,use_vehicle_model=cfg.TRAJ_USE_VEHICLE_MODEL,param_file=cfg.TRAJ_PARAM_FILE,data_dict=data,
                speed_range=cfg.TRAJ_SPEED_RANGE,steer_range=(0.,0.),speed_unit=cfg.TRAJ_SPEED_UNIT,steer_unit=cfg.TRAJ_STEER_UNIT))
            for k,v in list(data.items()):
                if isinstance(v,torch.Tensor):data[k]=v.cuda()
            imgs=data.pop('img');lidar=[x.cuda() for x in data.pop('lidar')]
            with torch.cuda.amp.autocast(True):result=net(imgs=imgs,lidar=lidar,metas=data,global_iter=0,train_mode='occ',task_phase=None)
            for k in ['traj_label','traj_xy_used','traj_valid_mask','dp3_gt_traj_label','pred_traj_label_logits','traj_z']:
                v=result[k];assert torch.isfinite(v).all(),k;captured[k]=v.detach().cpu().numpy()
            captured['probabilities']=torch.softmax(result['pred_traj_label_logits'].float(),dim=-1).cpu().numpy()
            np.savez_compressed(out/f'frame-{i}.npz',**captured)
            mask=captured['traj_valid_mask'].astype(bool)&(captured['dp3_gt_traj_label']!=26)
            with np.load(old/f'frame-{i}.npz',allow_pickle=False) as prior:
                same_xy=np.array_equal(captured['traj_xy_used'],prior['traj_xy_used'])
                same_gt=np.array_equal(captured['dp3_gt_traj_label'],prior['dp3_gt_traj_label'])
                changed=int(((captured['traj_label']!=prior['traj_label'])&mask).sum())
            assert same_xy and same_gt
            labels,counts=np.unique(captured['traj_label'][mask],return_counts=True)
            receipt['frames'].append({'frame':launch['selected_frames'][i][1],'valid_points':int(mask.sum()),
                'prediction_counts':dict(zip(labels.astype(int).tolist(),counts.tolist())),
                'changed_labels_vs_FP32':changed,'same_trajectory_xy':bool(same_xy),'same_GT':bool(same_gt),
                'mean_asphalt_probability':float(captured['probabilities'][mask][:,1].mean()),
                'median_asphalt_probability':float(np.median(captured['probabilities'][mask][:,1]))})
            receipt['new_full_model_forward_frames']=i+1
    receipt.update(state='succeeded',exit_code=0)
except Exception:receipt.update(state='failed',exit_code=1,error=traceback.format_exc());print(receipt['error'])
receipt['outputs']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.glob('*.npz')]
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False));print(json.dumps(receipt))
raise SystemExit(receipt['exit_code'])
