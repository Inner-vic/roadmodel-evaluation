"""Two frozen training frames under the original validation preprocessing and XY."""
import os,sys,json,pathlib,hashlib,time,traceback
import numpy as np
BASE=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
ROOT=BASE/'split-gt-control-20261027';OUT=ROOT/'run-01'
assert not OUT.exists(),'Keep prior attempts';OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
launch=json.loads((ROOT/'launch.json').read_text());os.environ.update(launch['environment']);os.environ['eval']='true'
r={'schema':'roadmodel.split-control/v1','state':'preflight','expected_frames':2,'new_forward_frames':0,'new_training_runs':0,'selected_frames':launch['selected_frames'],'mode':'validation preprocessing, model.eval, FP32, exact retained validation straight XY','source_commit':launch['source_commit'],'checkpoint_sha256':launch['checkpoint_sha256'],'inputs':[]}
save('receipt.json',r)
try:
    for f in json.loads((ROOT/'inputs-manifest.json').read_text()):
        p=ROOT/f['path'];assert sha(p)==f['sha256'];r['inputs'].append({'path':f['path'],'sha256':f['sha256']})
    for name,digest in launch['checkpoint_sha256'].items():assert sha(BASE/'weights'/name)==digest
    source=pathlib.Path(launch['source_root']);os.chdir(source);sys.path.insert(0,str(source))
    manifest=json.loads((BASE/'source-manifest.json').read_text());assert all(sha(source/n)==s for n,s in manifest['files'].items())
    import torch
    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmseg.models import build_segmentor
    import model
    from dataset import get_dataloader
    from misc.ckpt_load_utils import overlay_export_heads,_update_aggregator_cuda_kwargs
    set_random_seed(42);cfg=Config.fromfile(launch['py_config'],lazy_import=False)
    assert cfg.HEAD_MODE=='traj_only' and cfg.USE_TEMPORAL is False
    (OUT/'resolved-config.py').write_text(cfg.pretty_text)
    net=build_segmentor(cfg.model);sd=torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict']
    net.load_state_dict(sd,strict=True);overlay_export_heads(net,classifier_resume=launch['classifier_checkpoint'])
    assert all(torch.equal(v.cpu(),sd[k]) for k,v in net.state_dict().items() if k.startswith('head.traj_head.'))
    _update_aggregator_cuda_kwargs(cfg,net);net.cuda().eval();head=net.head.traj_head
    _,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
    assert len(loader.dataset)==2
    reference=BASE/'heads-20261027/trajectory-01/frame-0.npz'
    with np.load(reference,allow_pickle=False) as old:
        xy=torch.from_numpy(old['traj_xy_used'].copy());oldmask=old['traj_valid_mask'].reshape(4,-1);vc=int(oldmask[0].sum())
        assert vc==340
    mask=torch.zeros((1,1600),dtype=torch.bool)
    for w in range(4):mask[:,w*400:w*400+vc]=True
    r.update(state='running',GPU=torch.cuda.get_device_name(0),reference_XY_sha256=sha(reference));save('receipt.json',r);start=time.monotonic();rows=[]
    with torch.no_grad():
        for i,data in enumerate(loader):
            data.update(traj_xy=xy.clone(),traj_valid_mask=mask.clone(),N=torch.tensor([[1200+vc]],dtype=torch.int32),vc_per_wheel=torch.tensor([[vc]*4],dtype=torch.int32))
            for k,v in list(data.items()):
                if isinstance(v,torch.Tensor):data[k]=v.cuda()
            imgs=data.pop('img');lidar=[v.cuda() if v is not None else None for v in data.pop('lidar')]
            result=net(imgs=imgs,lidar=lidar,input_lidar_point_num=[torch.tensor([len(v)],dtype=torch.int32) for v in lidar],metas=data,global_iter=0,train_mode='occ',task_phase=None)
            arrays={k:v.detach().cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor) and k.startswith(('traj_','dp3_gt_traj_','pred_traj_','surface_'))}
            assert np.array_equal(arrays['traj_xy_used'],xy.numpy())
            for k,v in arrays.items():assert np.isfinite(v).all(),k
            valid=result['traj_valid_mask'].bool()&(result['dp3_gt_traj_label']!=26);gt=result['dp3_gt_traj_label'];pred=result['traj_label']
            classes={str(int(k)):{'support':int(((gt==k)&valid).sum()),'correct':int(((gt==k)&valid&(pred==gt)).sum())} for k in gt[valid].unique()}
            for v in classes.values():v['recall_percent']=v['correct']/v['support']*100
            logits=result['pred_traj_label_logits'].float();ce=torch.nn.functional.cross_entropy(logits.reshape(-1,27),gt.long().reshape(-1),reduction='none').reshape_as(gt)
            savepath=OUT/f'frame-{i}.npz';np.savez_compressed(savepath,**arrays)
            row={'scene':launch['selected_frames'][i][0],'frame_index':launch['selected_frames'][i][1],'classes':classes,'valid_points':int(valid.sum()),'correct':int(((gt==pred)&valid).sum()),'GT_counts':{str(int(k)):int((gt[valid]==k).sum()) for k in gt[valid].unique()},'prediction_counts':{str(int(k)):int((pred[valid]==k).sum()) for k in pred[valid].unique()},'center_mean_CE':float(ce[valid].mean()),'file':savepath.name,'sha256':sha(savepath),'same_XY_as_retained_validation':True}
            rows.append(row);r.update(new_forward_frames=i+1,frames=rows);save('receipt.json',r);print(json.dumps(row),flush=True)
    r.update(state='succeeded',seconds=time.monotonic()-start,exit_code=0)
except Exception:
    r.update(state='failed',exit_code=1,error=traceback.format_exc());print(r['error'],flush=True)
finally:
    r['outputs']=[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'];save('receipt.json',r)
print(json.dumps({'state':r['state'],'new_forward_frames':r['new_forward_frames']}),flush=True)
raise SystemExit(r['exit_code'])
