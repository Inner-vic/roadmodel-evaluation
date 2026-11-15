import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
from pcm_eval.common import utc_now
p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
data=request(['devspaces','get','--namespace','vc-chassis','--name',args.name])
meta=data.get('metadata',{});spec=data.get('spec',{})
result={'checked_at':utc_now(),'name':args.name,'metadata':{k:meta.get(k) for k in ['name','namespace','uid','labels','annotations']},
        'spec':{k:spec.get(k) for k in ['clusterName','image','resources','dataVols','accesses']},'status':data.get('status'),
        'changed_state':False}
root=Path('evidence/model-evaluation/model-reference-20261027');(root/(args.name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'name':args.name,'status':result['status'],'base_image':meta.get('annotations',{}).get('lpai.org/base-image'),
                  'gpu_type':spec.get('resources',{}).get('gpuType'),'mount_count':len(spec.get('dataVols') or []),'changed_state':False},ensure_ascii=True))
