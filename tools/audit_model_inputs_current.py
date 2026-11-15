"""Read-only model-input audit. Persist only selected nonsecret task settings."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import utc_now
from pcm_eval.lpai import request

ROOT=Path('evidence/environment/model-input-review-20261027')
ROOT.mkdir(parents=True,exist_ok=True)
UID='81e922d6-ee86-483c-9366-d5e3f6aca940'

def inspect(task):
    out={'task_uid':UID,'top_level_fields':list(task)}
    for key in ['name','uid','namespace','cluster','status','evaluation','branch','commit_id']:
        if key in task:out[key]=task[key]
    # Task configuration is sometimes JSON serialized inside one of these fields.
    configs=[]
    def find_config(v,depth=0):
        if depth>4:return
        if isinstance(v,str) and v.lstrip().startswith('{'):
            try:find_config(json.loads(v),depth+1)
            except ValueError:pass
        elif isinstance(v,dict):
            if 'lpai_task' in v or 'lpai_task_meta' in v:configs.append(v)
            for key,child in v.items():
                if key in ('data','config','configuration','pipeline_config','pipeline_params','task_config','spec'):
                    find_config(child,depth+1)
    find_config(task)
    out['configurations']=[]
    for config in configs:
        selected={}
        for key,allowed in {
            'lpai_git':{'branch','commit_id','env_key','env_value','name','type'},
            'lpai_input_dataset':{'env_key','env_value','name','version'},
            'lpai_model_download':{'env_key','env_value','model_files','model_set','model_version','model_version_id','namespace'},
            'lpai_model_register':{'env_key','env_value','model_set','model_version','model_set_id','model_version_id','model_path','model_files','path','name','version','namespace','local_path','upload_path'}
        }.items():
            if key in config:
                value=config[key]
                selected[key]=[{k:v for k,v in item.items() if k in allowed} for item in value if isinstance(item,dict)] if isinstance(value,list) else {k:v for k,v in value.items() if k in allowed} if isinstance(value,dict) else {'structure':type(value).__name__}
        # Keep command only after withholding any credential-bearing segments.
        command=config.get('lpai_task',{}).get('cmd')
        if command:
            selected['command_recorded']=True
            # Extract only known configuration/entry-point tokens, no freeform shell.
            import re
            selected['entry_points']=re.findall(r'(?:[\w./-]+\.py)',str(command))
            safe_names={'DATASET','HEAD_MODE','USE_TEMPORAL','SEQUENTIAL_TRAIN','TRAIN_MODE','XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX','CHECKPOINT','RESUME_FROM','PY_CONFIG'}
            selected['command_model_env']={name:value.strip("\"'") for name,value in re.findall(r'\b([A-Z][A-Z0-9_]*)=(\S+)',str(command)) if name in safe_names}
        meta=config.get('lpai_task_meta',{})
        selected['task_meta']={k:meta[k] for k in ['evaluation','task_name','task_uid','task_type'] if k in meta}
        env=config.get('lpai_envs',{})
        selected['environment_structure']=type(env).__name__
        selected['environment_fields']=list(env) if isinstance(env,dict) else []
        # Only requested settings; never store credentials or unrelated env values.
        wanted={'DATASET','USE_TEMPORAL','SEQUENTIAL_TRAIN','HEAD_MODE','TRAIN_MODE','PY_CONFIG','CONFIG','XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'}
        selected['model_env']={}
        if isinstance(env,list):
            selected['environment_item_fields']=[list(item) for item in env[:2] if isinstance(item,dict)]
            for item in env:
                if isinstance(item,dict):
                    name=item.get('name',item.get('key',item.get('env_key')))
                    if name in wanted:selected['model_env'][name]=item.get('value',item.get('env_value'))
        elif isinstance(env,dict):
            selected['model_env']={k:v for k,v in env.items() if k in wanted}
        runtime=config.get('lpai_task',{})
        image=runtime.get('image_info',{})
        if isinstance(image,dict):selected['image']={k:v for k,v in image.items() if k in ('name','version','image','image_name','image_url','uid','id')}
        elif isinstance(image,str):
            try:
                image=json.loads(image)
                selected['image']={k:v for k,v in image.items() if k in ('name','version','image','image_name','image_url','uid','id')} if isinstance(image,dict) else {'structure':type(image).__name__}
            except ValueError:
                selected['image_structure']='string'
        out['configurations'].append(selected)
    return out

record={'checked_at':utc_now(),'action':'read_only_task_get','new_evaluation_started':False}
try:
    task=request(['tasks','get','--uid',UID])
    record.update(state='queried',task=inspect(task))
except Exception as error:
    record.update(state='query_unavailable',error_type=type(error).__name__)
    existing=json.loads(Path('evidence/environment/lpai-source-task-current.json').read_text(encoding='utf-8-sig'))
    record['existing_snapshot_summary']=inspect(existing.get('data',existing))
(ROOT/'source-task-review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=True))
