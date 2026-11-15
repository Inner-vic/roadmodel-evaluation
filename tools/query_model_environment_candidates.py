"""Read current development-space candidates without starting/stopping any."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
from pcm_eval.common import utc_now
record={'checked_at':utc_now(),'read_only':True}
try:
 data=request(['devspaces','list','--namespace','vc-chassis','--page-size','50','--page-num','1'])
 record.update(state='queried',data=data)
except Exception as e:record.update(state='failed',error=str(e))
root=Path('evidence/model-evaluation/model-reference-20261027');root.mkdir(parents=True,exist_ok=True)
# Report operational names/status/endpoint metadata only; omit secret-bearing objects.
def select(v):
 if isinstance(v,list):return [select(x) for x in v]
 if not isinstance(v,dict):return v
 keep={'name','namespace','cluster','phase','status','state','items','data','list','total','total_count','has_more','owner','image','image_name','gpu','resources','uid','count','metadata','labels','annotations','ownerReferences','kind','user','email'}
 return {k:select(x) for k,x in v.items() if k in keep}
record['data']=select(record.get('data'))
(root/'devspace-candidates.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=True)[:18000])
