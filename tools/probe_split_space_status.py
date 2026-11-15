import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
v=request(['devspaces','get','--name','bmad-r3-a100-20261015'])
if isinstance(v,str):v=json.loads(v)
def selected(x):
    if isinstance(x,dict):
        return {k:selected(v) for k,v in x.items() if k in ['name','uid','status','state','phase','namespace','cluster','data','items','ports','ssh_port','port','host','ssh_host']}
    if isinstance(x,list):return [selected(v) for v in x]
    return x
s=selected(v);Path('evidence/model-evaluation/split-gt-audit-20261027/space-status.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(s,ensure_ascii=False))
