"""Read only the original jobs' dataset and loader provenance, never raw env."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
from pcm_eval.model_inputs import parse_launch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-frame-binding-20261107'
def collect(uid):
    try:task=request(['tasks','get','--uid',uid],cli='C:/Users/developer/AppData/Local/bin/lpai-agent.exe')
    except Exception as error:return {'task_uid':uid,'state':'READ_UNAVAILABLE','error_type':type(error).__name__}
    data=task.get('data',{});data=json.loads(data) if isinstance(data,str) else data
    runtime=data.get('lpai_task',{});launch=parse_launch(str(runtime.get('cmd','')))
    datasets=[{k:x.get(k) for k in ('name','version','env_key','env_value')} for x in data.get('lpai_input_dataset',[]) if isinstance(x,dict)]
    code=[{k:x.get(k) for k in ('branch','commit_id','env_value')} for x in data.get('lpai_git',[]) if isinstance(x,dict)]
    settings={k:v for k,v in launch.get('environment',{}).items() if k in ('DATASET_PATH','DATA_LINK_FILE','DATASET','TRAIN_MODE','HEAD_MODE')}
    return {'task_uid':uid,'name':task.get('name'),'status':task.get('status'),'datasets':datasets,'code':code,'settings':settings,'state':'READ_VERIFIED'}
values=[collect(x) for x in ('8fc0b602-344c-4f73-8fe7-f7b6f77b60aa','81e922d6-ee86-483c-9366-d5e3f6aca940')]
(OUT/'training-GT-contract.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
print(json.dumps(values))
