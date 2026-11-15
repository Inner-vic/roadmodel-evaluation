"""Read nonsecret task code metadata and exact original execution pod identity."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/camera-GT-mapping-20261107';OUT.mkdir(exist_ok=True)
uid='81e922d6-ee86-483c-9366-d5e3f6aca940';cli='C:/Users/developer/AppData/Local/bin/lpai-agent.exe'
task=request(['tasks','get','--uid',uid],cli=cli);data=task.get('data',{});data=json.loads(data) if isinstance(data,str) else data
allowed={'name','branch','commit_id','commit','revision','code_version','version','uid','id','env_key','env_value','type','tag','sha','repo_name','git_version'}
git=[{'field_names':list(item),'values':{k:v for k,v in item.items() if k in allowed}} for item in data.get('lpai_git',[])]
pods=request(['tasks','pods','--uid',uid],cli=cli)
if isinstance(pods,dict):pods=pods.get('items',pods.get('pods',[]))
pod_allowed={'name','step','pod_type','create_timestamp','finish_timestamp','status','namespace','cluster','uid'}
safe_pods=[{k:v for k,v in p.items() if k in pod_allowed} for p in pods]
result={'task':{k:task.get(k) for k in ('uid','name','namespace','cluster','start_timestamp','update_timestamp')},'git_metadata':git,'data_field_names':list(data),'pods':safe_pods}
(OUT/'task-code-provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
