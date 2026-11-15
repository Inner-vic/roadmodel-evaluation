"""Read the known output modelset index, without downloading checkpoints."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
uid='8d226ad5-e769-411b-9c72-57f9dd822c0e'
t=request(['tasks','get','--uid',uid]);data=t.get('data',{})
if isinstance(data,str):data=json.loads(data)
selected={k:data.get(k) for k in ['lpai_model_register','lpai_tensorboard'] if k in data}
# Do not persist runtime credentials or environment payloads.
out=[]
for x in selected.get('lpai_model_register',[]):out.append({k:v for k,v in x.items() if k in ['name','model_set','model_version','model_version_id','model_path','path','env_key','env_value','is_register']})
(root/'output-registration.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('registration',json.dumps(out))
v=request(['assets','modelsets','versions','--name','test-gzw','--namespace','vc-chassis','--page-size','100','--page-num','1'])
if isinstance(v,str):v=json.loads(v)
def keep(x):
    if isinstance(x,dict):return {k:keep(a) for k,a in x.items() if not any(s in k.lower() for s in ['secret','token','credential','password','access_key','authorization'])}
    if isinstance(x,list):return [keep(a) for a in x]
    return x
v=keep(v);(root/'output-modelset-versions.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('structure',list(v) if isinstance(v,dict) else type(v).__name__)
def scan(x):
    if isinstance(x,dict):
        if 'version' in x or 'model_version' in x:
            print(json.dumps({k:v for k,v in x.items() if k in ['version','name','uid','model_version','create_time','create_timestamp','task_uid','source_id']},ensure_ascii=False))
        for a in x.values():scan(a)
    elif isinstance(x,list):
        for a in x:scan(a)
scan(v)
