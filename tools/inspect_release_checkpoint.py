"""Inspect standard checkpoint metadata safely on CPU, without model execution."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import importlib.util
import json
from pathlib import Path

root=Path('evidence/model-evaluation/model-reference-20261027')
if importlib.util.find_spec('torch') is None:
 print(json.dumps({'state':'torch_not_available','new_model_execution':False}));raise SystemExit(0)
import torch
for version,name in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 path=root/'weights'/version/name
 if not path.is_file():continue
 record={'version':version,'file':name,'torch':torch.__version__,'inspection':'weights_only=True; map_location=cpu'}
 try:
  checkpoint=torch.load(path,map_location='cpu',weights_only=True)
  state=checkpoint.get('state_dict',checkpoint.get('model',checkpoint)) if isinstance(checkpoint,dict) else checkpoint
  record['top_level_keys']=list(checkpoint) if isinstance(checkpoint,dict) else []
  record['state_dict_count']=len(state)
  record['tensor_shapes']={k:list(v.shape) for k,v in state.items() if hasattr(v,'shape')}
  record['metadata']={k:checkpoint[k] for k in ['epoch','iter'] if k in checkpoint and type(checkpoint[k]) in (int,str,float)}
  meta=checkpoint.get('meta',{}) if isinstance(checkpoint,dict) else {}
  if isinstance(meta,dict):
   record['meta_keys']=list(meta)
   for key in ['epoch','iter','seed','mmcv_version','mmengine_version','git_hash']:
    if key in meta and type(meta[key]) in (str,int,float):record.setdefault('metadata',{})[key]=meta[key]
   for key in ['config','cfg']:
    if isinstance(meta.get(key),str):
     (path.parent/('embedded-'+key+'.txt')).write_text(meta[key],encoding='utf-8')
     record['embedded_config']=str(path.parent/('embedded-'+key+'.txt'))
  record['state']='inspected'
 except Exception as error:
  record.update(state='inspection_failed',error_type=type(error).__name__)
 (path.parent/'inspection.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='tensor_shapes'}),flush=True)
