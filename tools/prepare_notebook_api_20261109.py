"""Freeze the exact reviewed GPU Notebook request; never submit it here."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.dataset_mount import resolve_dataset
from pcm_eval.lpai import request
from pcm_eval.common import save_json

root=Path(__file__).resolve().parents[1]/'evidence/rebuild-initialization-20261109'
req=json.loads((root/'request.json').read_text())
detail=json.loads((root/'plan/initialization.json').read_text())
asset=resolve_dataset(req['dataset_ref'],request)
observed=json.loads((root/'sdk-readback.json').read_text(encoding='utf-8-sig'))
assert observed['state']=='OBSERVED' and observed['owner_verified']
t=req['template'];key=req['ssh']['credential_name'];path='/mnt/datasets/'+asset['name']+'/'+asset['version']
services=observed['notebook_config']['accesses'];access=[]
for x in services['required']+services['optional']:
    if x['name'] not in ('jupyter-lab','ssh'):continue
    x={k:v for k,v in x.items() if k in ('name','targetPort','protocol','address')}
    if x['name']=='ssh':x['pubkeyName']=key
    access.append(x)
body={'metadata':{'name':detail['space_name'],'namespace':t['namespace'],
    'labels':{'lpai.org/kind':'notebook','lpai.org/visible':'personal'},
    'annotations':{'lpai.org/desc':'RoadModel专属评测空间；当前账号初始化与小场景验收；不复制数据集。',
                   'lpai.org/gems-project':t['project']}},
    'spec':{'image':t['image'],'clusterName':t['cluster'],'sshSecretName':key,
        'resources':t['resources'],'autoCulling':{'enable':'true','windowMinutes':'240'},'env':{},
        'dataVols':[{'type':'dataset','id':asset['dataset_uid'],'name':asset['name'],'version':asset['version'],
            'fsType':'ofs','mountPath':path,'envKey':'LPAI_INPUT_DATASET_0','envValue':path}],
        'accesses':access}}
save_json(root/'api-asset.json',asset)
save_json(root/'api-create-body.json',body)
print(json.dumps({'state':'PREPARED','name':detail['space_name'],'dataset_uid':asset['dataset_uid'],
    'version':asset['version'],'platform_writes':0}))
