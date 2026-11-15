"""Two frozen frames only: native forward plus retained-feature semantic diagnostics.

Hooks only copy tensors. Counterfactuals replay the small semantic module on saved
features; they are diagnostic interventions, not additional quality evaluations.
"""
import os,sys,json,pathlib,time,hashlib,datetime,traceback
import numpy as np

BASE=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
OLD=BASE/'heads-20261027/trajectory-01'
OUT=BASE/'heads-20261027/trajectory-diagnostic-01'
assert not OUT.exists(),'Preserve prior diagnostic attempts'
OUT.mkdir()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
launch=json.loads((OLD/'launch.json').read_text());os.environ.update(launch['environment']);os.environ['eval']='true'
receipt={'schema':'roadmodel.trajectory-diagnostic/v1','run_id':launch['run_id']+'-diagnostic-01',
    'source_inference_run_id':launch['run_id'],'state':'preflight','started_at':now(),
    'selected_frames':launch['selected_frames'],'expected_frames':2,'new_forward_frames':0,
    'source_commit':launch['source_commit'],'checkpoint_sha256':launch['checkpoint_sha256'],
    'instrumentation':'forward hooks copy intermediate outputs; pinned model source unchanged',
    'counterfactual_scope':'semantic module replays on retained features only; not model accuracy scores'}
save('diagnostic-receipt.json',receipt)
try:
    assert launch['expected_frames']==2 and [fr for _,fr in launch['selected_frames']]==[20,21]
    for name,digest in launch['checkpoint_sha256'].items():assert sha(BASE/'weights'/name)==digest
    manifest=json.loads((BASE/'source-manifest.json').read_text())
    assert all(sha(BASE/'source'/name)==digest for name,digest in manifest['files'].items())
    inputs=json.loads((BASE/'heads-20261027/input-manifest.json').read_text())
    assert all(sha(BASE/'heads-20261027/input'/x['path'])==x['sha256'] for x in inputs if x['path'].startswith('DataReorganized-V151/'))
    source=pathlib.Path(launch['source_root']);os.chdir(source);sys.path.insert(0,str(source))
    import torch
    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmseg.models import build_segmentor
    import model
    from dataset import get_dataloader
    from misc.ckpt_load_utils import overlay_export_heads,_update_aggregator_cuda_kwargs
    from misc.traj_payload import build_fixed_traj_payload
    set_random_seed(42)
    cfg=Config.fromfile(launch['py_config'],lazy_import=False)
    assert cfg.HEAD_MODE=='traj_only' and cfg.DENSE_SURFACE_GT_SOURCE=='mesh'
    (OUT/'resolved-config.py').write_text(cfg.pretty_text)
    save('launch.json',launch)
    net=build_segmentor(cfg.model)
    sd=torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict']
    loaded=net.load_state_dict(sd,strict=True)
    overlay_export_heads(net,classifier_resume=launch['classifier_checkpoint'])
    for k,v in net.state_dict().items():
        if k.startswith('head.traj_head.'):assert torch.equal(v.cpu(),sd[k].cpu()),'Trajectory weights changed by overlay'
    _update_aggregator_cuda_kwargs(cfg,net)
    net.cuda().eval();head=net.head.traj_head
    _,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
    assert len(loader.dataset)==2
    captured={};module_inputs={};handles=[]
    def copy_tensors(prefix,value):
        if isinstance(value,torch.Tensor):
            assert value.numel()<=4_000_000,(prefix,list(value.shape))
            captured[prefix]=value.detach().cpu().numpy().copy()
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):copy_tensors(prefix+'_'+str(i),v)
        elif isinstance(value,dict):
            for k,v in value.items():copy_tensors(prefix+'_'+str(k),v)
    def summarize(t):
        if not isinstance(t,torch.Tensor):return {'type':type(t).__name__}
        d=t.detach().float();return {'shape':list(t.shape),'dtype':str(t.dtype),'finite':bool(torch.isfinite(d).all()),
            'mean':float(d.mean()),'std':float(d.std()) if d.numel()>1 else 0.,'abs_max':float(d.abs().max())}
    def hook(name):
        def callback(module,args,output):
            copy_tensors(name,output)
            if name in ['surface_semantic_context','surface_cls_head']:
                for i,v in enumerate(args):copy_tensors(name+'_input'+str(i),v)
            if name=='surface_lidar_conditioner':module_inputs[name]=[summarize(v) for v in args]
        return callback
    names=['surface_position_pack','surface_lidar_pack','surface_gaussian_attention','surface_semantic_context',
        'surface_cls_head','surface_lidar_conditioner','anchor_in']
    names.extend('surface_semantic_context.blocks.'+str(i) for i in range(4))
    for name in names:handles.append(head.get_submodule(name).register_forward_hook(hook(name)))
    save('preflight.json',{'state':'passed','strict_main_keys':len(sd),'trajectory_keys':sum(k.startswith('head.traj_head.') for k in sd),
        'classifier_overlay_keys':sum(k.startswith('classifier.') for k in sd),'trajectory_weights_unchanged_by_overlay':True,
        'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'dataset_size':2,
        'class_names':list(cfg.label_str)+['empty'],'center_offset_index':head.surface_center_offset_index,
        'surface_y_offsets':head.surface_y_offsets.cpu().tolist()})
    np.savez_compressed(OUT/'semantic-weights.npz',weight=head.surface_cls_head.weight.detach().cpu().numpy(),
                        bias=head.surface_cls_head.bias.detach().cpu().numpy())
    receipt['state']='running';save('diagnostic-receipt.json',receipt)
    frame_rows=[];retained=[]
    start=time.monotonic()
    with torch.no_grad():
        for i,data in enumerate(loader):
            captured.clear();module_inputs.clear()
            payload=build_fixed_traj_payload(pc_range=cfg.pc_range,max_len_traj=cfg.MAX_TRAJ_LEN,traj_step=cfg.TRAJ_STEP,
                batch_size=1,use_vehicle_model=cfg.TRAJ_USE_VEHICLE_MODEL,param_file=cfg.TRAJ_PARAM_FILE,data_dict=data,
                speed_range=cfg.TRAJ_SPEED_RANGE,steer_range=(0.,0.),speed_unit=cfg.TRAJ_SPEED_UNIT,steer_unit=cfg.TRAJ_STEER_UNIT)
            data.update(payload)
            for k,v in list(data.items()):
                if isinstance(v,torch.Tensor):data[k]=v.cuda()
            imgs=data.pop('img');lidar=[v.cuda() if v is not None else None for v in data.pop('lidar')]
            input_summary={'image':summarize(imgs),'lidar':[summarize(v) for v in lidar]}
            result=net(imgs=imgs,lidar=lidar,input_lidar_point_num=[torch.tensor([len(v)],dtype=torch.int32) for v in lidar],
                       metas=data,global_iter=0,train_mode='occ',task_phase=None)
            for k,v in result.items():
                if k.startswith(('traj_','dp3_gt_traj_','pred_traj_','surface_')) and isinstance(v,torch.Tensor):copy_tensors('result_'+k,v)
            copy_tensors('result_traj_probabilities',torch.softmax(result['pred_traj_label_logits'].float(),dim=-1))
            for k,v in captured.items():assert np.isfinite(v).all(),k
            native_path=OUT/f'frame-{i}-native.npz';np.savez_compressed(native_path,**captured)
            mask=result['traj_valid_mask'].bool()&(result['dp3_gt_traj_label']!=26)
            with np.load(OLD/f'frame-{i}.npz',allow_pickle=False) as old:
                checks={key:bool(np.array_equal(result[key].detach().cpu().numpy(),old[key])) for key in old.files}
                height_delta=float(np.max(np.abs(result['traj_z'].cpu().numpy()-old['traj_z'])))
            row={'ordinal':i,'scene':launch['selected_frames'][i][0],'frame_index':launch['selected_frames'][i][1],
                'valid_points':int(mask.sum()),'correct':int(((result['traj_label']==result['dp3_gt_traj_label'])&mask).sum()),
                'prior_output_exact_match':checks,'max_height_difference_bins':height_delta,
                'input_summary':input_summary,'lidar_conditioner_inputs':dict(module_inputs),'arrays':{k:list(v.shape) for k,v in captured.items()},
                'file':native_path.name,'sha256':sha(native_path)}
            frame_rows.append(row);retained.append({k:v.copy() for k,v in captured.items()})
            receipt['new_forward_frames']=i+1;receipt['frames']=frame_rows;save('diagnostic-receipt.json',receipt)
            print(json.dumps({'native_frame_completed':row['frame_index'],'valid_points':row['valid_points'],'correct':row['correct'],'exact_prior':all(checks.values())}),flush=True)
    receipt['native_forward_seconds']=time.monotonic()-start
    for handle in handles:handle.remove()
    # Replay only the semantic module, preserving native spatial mask/order.
    with torch.no_grad():
        for i,arrays in enumerate(retained):
            t=lambda name:torch.from_numpy(arrays[name]).cuda()
            position=t('surface_position_pack');lidar=t('surface_lidar_pack');gaussian=t('surface_gaussian_attention').reshape_as(position)
            native_input=t('surface_semantic_context_input0');valid=t('surface_semantic_context_input1').bool()
            assert torch.allclose(position+gaussian+lidar,native_input,atol=2e-6,rtol=1e-6)
            variants={'native_replay':native_input,'no_position':gaussian+lidar,'no_gaussian':position+lidar,
                'no_lidar':position+gaussian,'position_only':position,'gaussian_only':gaussian,'lidar_only':lidar,'zero_input':torch.zeros_like(native_input)}
            outputs={}
            for name,features in variants.items():
                features=head.surface_semantic_context(features,valid);logits=head.surface_cls_head(features)
                logits=logits.reshape(1,2,400,3,27)[:,:,:,head.surface_center_offset_index,:].reshape(1,800,27)
                outputs[name+'_logits']=logits.float().cpu().numpy();outputs[name+'_probabilities']=torch.softmax(logits.float(),dim=-1).cpu().numpy()
            full=t('result_pred_traj_label_logits')[:,:800]
            assert np.array_equal(outputs['native_replay_logits'],full.cpu().numpy()),'Semantic replay differs from actual native logits'
            np.savez_compressed(OUT/f'frame-{i}-semantic-interventions.npz',**outputs)
    receipt.update(state='succeeded',finished_at=now(),exit_code=0,new_forward_frames=2,semantic_replay_variants=8,
        semantic_replay_is_full_model_inference=False)
except Exception:
    receipt.update(state='failed',finished_at=now(),exit_code=1,error=traceback.format_exc())
    print(receipt['error'],flush=True)
finally:
    receipt['outputs']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='diagnostic-receipt.json']
    save('diagnostic-receipt.json',receipt)
print(json.dumps({'state':receipt['state'],'new_forward_frames':receipt['new_forward_frames'],'directory':str(OUT)}))
raise SystemExit(receipt['exit_code'])
