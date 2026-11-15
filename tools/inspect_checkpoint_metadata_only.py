"""Inspect PyTorch ZIP checkpoint metadata with a strict, non-Torch unpickler.

No storage bytes are loaded and unknown pickle globals are rejected. The only
callables exposed to pickle are inert metadata constructors defined here.
"""
from collections import OrderedDict,Counter
import io
import json
import pickle
from pathlib import Path
import zipfile

class StorageType:
 def __init__(self,name):self.name=name

class TensorMetadata:
 def __init__(self,storage,offset,size,stride,*args):
  self.shape=list(size);self.dtype=storage.get('dtype') if isinstance(storage,dict) else None
  self.storage_key=storage.get('key') if isinstance(storage,dict) else None
  self.storage_offset=offset;self.stride=list(stride)

class InertMetadata:
 def __init__(self,*args):self.args=args
 def __setstate__(self,state):self.state=state

def rebuild_tensor(*args):return TensorMetadata(*args)
def rebuild_parameter(data,*args):return data
def inert(*args):return InertMetadata(*args)

class MetadataOnlyUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return OrderedDict
  if module=='torch' and name.endswith('Storage'):return StorageType(name)
  if module=='torch._utils' and name in ('_rebuild_tensor','_rebuild_tensor_v2','_rebuild_tensor_v3'):return rebuild_tensor
  if module=='torch._utils' and name in ('_rebuild_parameter','_rebuild_parameter_with_state'):return rebuild_parameter
  if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):return inert
  if module=='numpy' and name in ('dtype','ndarray'):return InertMetadata
  raise pickle.UnpicklingError('Unsupported global: '+module+'.'+name)
 def persistent_load(self,pid):
  if not isinstance(pid,tuple) or pid[0]!='storage':raise pickle.UnpicklingError('Unexpected storage reference')
  return {'dtype':pid[1].name,'key':pid[2],'location':pid[3],'numel':pid[4]}

root=Path('evidence/model-evaluation/model-reference-20261027/weights')
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 path=root/version/filename
 if not path.is_file():continue
 record={'version':version,'file':filename,'method':'restricted metadata-only ZIP unpickler','tensor_storage_loaded':False,'model_executed':False}
 try:
  with zipfile.ZipFile(path) as archive:
   info=next(i for i in archive.infolist() if i.filename.endswith('/data.pkl'))
   if info.file_size>16*1024*1024:raise ValueError('Metadata exceeds bounded inspection limit')
   checkpoint=MetadataOnlyUnpickler(io.BytesIO(archive.read(info))).load()
  record['top_level_keys']=list(checkpoint)
  state=checkpoint.get('state_dict',checkpoint.get('model',{}))
  record['tensor_metadata']={k:{'shape':v.shape,'dtype':v.dtype,'storage_key':v.storage_key,
                              'storage_offset':v.storage_offset,'stride':v.stride} for k,v in state.items() if isinstance(v,TensorMetadata)}
  record['tensor_count']=len(record['tensor_metadata'])
  record['prefix_counts']=dict(Counter(k.split('.')[0] for k in record['tensor_metadata']))
  meta=checkpoint.get('meta',{})
  record['meta_keys']=list(meta) if isinstance(meta,dict) else []
  record['metadata']={k:meta[k] for k in ['epoch','iter','seed','git_hash','mmcv_version','mmengine_version'] if isinstance(meta,dict) and k in meta and type(meta[k]) in (int,float,str)}
  record['metadata'].update({k:checkpoint[k] for k in ['epoch','global_iter'] if type(checkpoint.get(k)) in (int,float,str)})
  for key in ['cfg','config']:
   if isinstance(meta,dict) and isinstance(meta.get(key),str):
    p=path.parent/('embedded-'+key+'.txt');p.write_text(meta[key],encoding='utf-8');record['embedded_config']=str(p)
  record['state']='metadata_inspected'
 except Exception as error:record.update(state='metadata_inspection_failed',error_type=type(error).__name__,error=str(error))
 (path.parent/'metadata-inspection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in record.items() if k!='tensor_metadata'},ensure_ascii=True),flush=True)
