import json
from pathlib import Path
from model_space_transport import run

code=r'''
import pathlib,subprocess,json,torch
result={'gpus':[],'repositories':[],'custom_modules':{}}
for i in range(torch.cuda.device_count()):
 free,total=torch.cuda.mem_get_info(i);result['gpus'].append({'index':i,'free':free,'total':total})
for path in ['/lpai/bmad-roadmodel-source','/lpai/bmad-r3-a100-20261015']:
 p=pathlib.Path(path)
 if not p.is_dir():continue
 if (p/'.git').exists():
  head=subprocess.run(['git','-C',path,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
  has=subprocess.run(['git','-C',path,'cat-file','-e','9e16e29351d17a726ce6beca6f821de79b7ade40^{commit}'],capture_output=True).returncode==0
  result['repositories'].append({'path':path,'head':head,'has_target_commit':has})
 result[path+'_directories']=[x.name for x in p.iterdir() if x.is_dir()][:30]
import importlib.util
for name in ['deformable_aggregation','local_aggregate_prob','local_aggregate_prob_fast','pointops','iou3d_op','voxel_layer']:
 try:result['custom_modules'][name]=str(importlib.util.find_spec(name))
 except Exception as e:result['custom_modules'][name]=type(e).__name__
print(json.dumps(result))
'''
data=json.loads(run(['python3','-c',code]))
root=Path('evidence/model-evaluation/model-reference-20261027')
(root/'runtime-source-inspection.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
