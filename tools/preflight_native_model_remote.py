import pathlib,json,os,sys,traceback,hashlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
launch=json.loads((base/'native-launch.json').read_text())
os.environ.update(launch['environment']);os.chdir(base/'source');sys.path.insert(0,str(base/'source'))
result={'state':'running','expected_frames':2}
try:
 import torch,model
 from mmengine import Config
 from mmseg.models import build_segmentor
 cfg=Config.fromfile('config/3-0-0-temporal-dit.py',lazy_import=False)
 my_model=build_segmentor(cfg.model)
 weights=torch.load(base/'weights/epoch_44.pth',map_location='cpu',weights_only=True)['state_dict']
 model_state=my_model.state_dict()
 result['checkpoint']={'model_keys':len(model_state),'checkpoint_keys':len(weights),
   'missing':sorted(set(model_state)-set(weights)), 'unexpected':sorted(set(weights)-set(model_state)),
   'shape_mismatches':[k for k in weights.keys()&model_state.keys() if weights[k].shape!=model_state[k].shape]}
 assert not result['checkpoint']['missing'] and not result['checkpoint']['unexpected'] and not result['checkpoint']['shape_mismatches'],result['checkpoint']
 my_model.load_state_dict(weights,strict=True)
 from misc.ckpt_load_utils import overlay_export_heads,_update_aggregator_cuda_kwargs
 overlay_export_heads(my_model,classifier_resume=str(base/'weights/epoch_98.pth'))
 overlay=torch.load(base/'weights/epoch_98.pth',map_location='cpu',weights_only=True)['state_dict']
 final=my_model.state_dict()
 assert all(torch.equal(final[k],overlay[k]) for k in final if k.startswith('classifier.'))
 result['classifier_overlay_keys']=sum(k.startswith('classifier.') for k in final)
 _update_aggregator_cuda_kwargs(cfg,my_model)
 from dataset import get_dataloader
 _,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
 result['dataset_size']=len(loader.dataset)
 assert result['dataset_size']==2
 data=loader.dataset[0]
 result['first_frame']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data.items() if hasattr(v,'shape')}
 result['model']={'pc_range':list(cfg.pc_range),'HWD':[cfg.H,cfg.W,cfg.D],'head_mode':cfg.HEAD_MODE,'train_mode':cfg.train_mode}
 result.update(state='passed',torch=torch.__version__,cuda=torch.version.cuda)
except Exception:
 result.update(state='failed',error=traceback.format_exc())
(base/'model-preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
