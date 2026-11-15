"""Decompose retained Gaussian conditioning; no backbone or full-model forward."""
import json,pathlib,os,sys,hashlib,time
import numpy as np
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
run=base/'heads-20261027/trajectory-diagnostic-01';out=run/'semantic-component-replay-02'
assert not out.exists();out.mkdir()
launch=json.loads((run/'launch.json').read_text());os.environ.update(launch['environment']);os.environ['eval']='true'
os.chdir(launch['source_root']);sys.path.insert(0,launch['source_root'])
import torch,model
from mmengine import Config
from mmseg.models import build_segmentor
cfg=Config.fromfile(str(run/'resolved-config.py'),lazy_import=False)
net=build_segmentor(cfg.model)
sd=torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict'];net.load_state_dict(sd,strict=True)
net.cuda().eval();head=net.head.traj_head
rows=[]
with torch.no_grad():
    for i in range(2):
        with np.load(run/f'frame-{i}-native.npz',allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
        t=lambda name:torch.from_numpy(a[name]).cuda()
        pos=t('surface_position_pack');lidar=t('surface_lidar_pack');g=t('surface_gaussian_attention').reshape_as(pos)
        anchor=head.surface_gaussian_attention.anchor_norm(t('anchor_in'))
        global_g=head.surface_gaussian_attention.global_output(anchor.mean(dim=1)).reshape(1,1,1,96).expand_as(pos)
        local_g=g-global_g
        valid=t('surface_semantic_context_input1').bool();mask=t('result_traj_valid_mask')[:,:800].bool()
        native_input=t('surface_semantic_context_input0')
        features={'native_replay':native_input,'no_global_gaussian':native_input-global_g,
            'no_local_gaussian':native_input-local_g,'local_gaussian_only':local_g,
            'global_gaussian_only':global_g,'no_gaussian':pos+lidar}
        saved={'gaussian_global':global_g.cpu().numpy(),'gaussian_local':local_g.cpu().numpy()}
        def score(v):
            f=head.surface_semantic_context(v,valid);z=head.surface_cls_head(f)
            return z.reshape(1,2,400,3,27)[:,:,:,1,:].reshape(1,800,27).float()
        for name,v in features.items():saved[name+'_logits']=score(v).cpu().numpy()
        native=t('result_pred_traj_label_logits')[:,:800]
        replay_delta=float((torch.from_numpy(saved['native_replay_logits']).cuda()-native).abs().max())
        assert replay_delta<1e-4,('Semantic native replay error',replay_delta)
        for dtype,name in [(torch.float16,'fp16_semantic_only'),(torch.bfloat16,'bf16_semantic_only')]:
            with torch.autocast('cuda',dtype=dtype):saved[name+'_logits']=score(t('surface_semantic_context_input0')).cpu().numpy()
        np.savez_compressed(out/f'frame-{i}.npz',**saved)
        rows.append({'frame':launch['selected_frames'][i][1],'file':f'frame-{i}.npz','valid_points':int(mask.sum()),
            'global_gaussian_rms':float(global_g.square().mean().sqrt()),'local_gaussian_rms':float(local_g.square().mean().sqrt()),
            'native_replay_logit_max_error':replay_delta})
receipt={'state':'succeeded','selected_frames':launch['selected_frames'],'new_full_model_forward_frames':0,
    'method':'Gaussian global = global_output(mean(anchor_norm(retained anchor tokens))); local = retained full Gaussian - global',
    'frames':rows,'precision_note':'FP16/BF16 replay covers semantic module only, not the full backbone/conditioning graph',
    'outputs':[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.glob('*.npz')]}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
