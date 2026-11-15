import pathlib,json,os,sys,traceback,hashlib
run_dir=pathlib.Path(sys.argv[1]);launch=json.loads((run_dir/'launch.json').read_text())
os.environ.update(launch['environment']);os.environ['eval']='true'
source=pathlib.Path(launch['source_root']);os.chdir(source);sys.path.insert(0,str(source))
out={'state':'running','kind':launch['kind'],'samples':[]}
try:
 import torch,model
 from mmengine import Config
 from mmseg.models import build_segmentor
 from dataset import get_dataloader
 from misc.ckpt_load_utils import overlay_export_heads
 cfg=Config.fromfile(launch['py_config'],lazy_import=False)
 net=build_segmentor(cfg.model)
 sd=torch.load(launch['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict']
 net.load_state_dict(sd,strict=True);out['strict_main_keys']=len(sd)
 overlay_export_heads(net,classifier_resume=launch['classifier_checkpoint'])
 out['classifier_overlay_keys']=sum(k.startswith('classifier.') for k in net.state_dict())
 _,loader=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
 assert len(loader.dataset)==launch['expected_frames']
 for i in range(len(loader.dataset)):
  item=loader.dataset[i]
  row={'ordinal':i,'scene':launch['selected_frames'][i][0],'frame_index':launch['selected_frames'][i][1],
       'image_shape':list(item['img'].shape)}
  if launch['kind']=='vision':
   labels=item.get('img_class');assert isinstance(labels,dict) and labels,'No frame classification labels loaded'
   core=list(getattr(net,'classifier_core_names',[]));assert core and all(k in labels for k in core),(core,labels)
   row.update(gt=labels,core_tasks=core)
  else:
   mesh=item.get('mesh_file');assert mesh is not None and mesh.shape[0]>0,'No mesh GT'
   mesh_array=mesh.detach().cpu().numpy() if hasattr(mesh,'detach') else mesh
   row.update(mesh_shape=list(mesh.shape),mesh_xyz_min=mesh_array[:,:3].min(axis=0).tolist(),mesh_xyz_max=mesh_array[:,:3].max(axis=0).tolist())
  out['samples'].append(row)
 out.update(state='passed',dataset_size=len(loader.dataset),torch=torch.__version__,cuda=torch.version.cuda,
  gt_source=cfg.DENSE_SURFACE_GT_SOURCE,pc_range=list(cfg.pc_range))
except Exception:out.update(state='failed',error=traceback.format_exc())
(run_dir/'model-preflight.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
raise SystemExit(0 if out['state']=='passed' else 2)
