"""Read the exact training configuration referenced by an asset version."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
from pcm_eval.common import utc_now
from pcm_eval.model_inputs import parse_launch

p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();task=request(['tasks','get','--uid',args.task])
data=task.get('data',{})
if isinstance(data,str):data=json.loads(data)
runtime=data.get('lpai_task',{})
command=str(runtime.get('cmd',''))
env_names={'DATASET_PATH','DATA_LINK_FILE','DATASET','HEAD_MODE','TRAIN_MODE','USE_TEMPORAL','SEQUENTIAL_TRAIN',
 'XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX','TRAJ_HEAD_TYPE','IMG_BACKBONE','EMBED_DIMS','NUM_ANCHOR',
 'EVAL_SUBSAMPLE','CHECKPOINT','RESUME_FROM','TRAJ_RESUME','CLASSIFIER_RESUME','LPAI_CODE_DIR_1','SINGLE_LAYER_ROAD',
 'CONFIG','USE_DENSITY','OUTPUT_BUMPER_TOP','OCC_X_SIZE','OCC_Y_SIZE','Z_GRID_SIZE'}
env={}
for k,v in re.findall(r'\b([A-Z][A-Z0-9_]*)=("[^"]*"|\x27[^\x27]*\x27|[^\s;]+)',command):
 if k in env_names:env[k]=v.strip('"\x27')
flags={}
for key in ['py-config','config','resume-from','traj-resume','classifier-resume','work-dir']:
 match=re.search(r'--'+key+r'(?:=|\s+)("[^"]*"|\x27[^\x27]*\x27|[^\s;]+)',command)
 if match:flags[key]=match[1].strip('"\x27')
def select(key,allowed):
 value=data.get(key)
 if isinstance(value,list):return [{k:v for k,v in item.items() if k in allowed} for item in value if isinstance(item,dict)]
 if isinstance(value,dict):return {k:v for k,v in value.items() if k in allowed}
 return None
image=runtime.get('image_info')
if not isinstance(image,(str,dict,list,type(None))):image=None
roles=runtime.get('roles',[])
result={'checked_at':utc_now(),'task_uid':args.task,
 'task':{k:task.get(k) for k in ['name','status','evaluation','namespace','cluster','start_timestamp','update_timestamp']},
 'git':select('lpai_git',{'name','branch','commit_id','type','env_key','env_value'}),
 'dataset':select('lpai_input_dataset',{'name','version','env_key','env_value','path'}),
 'input_models':select('lpai_model_download',{'model_set','model_version','model_version_id','model_files','env_key','env_value'}),
 'output_models':select('lpai_model_register',{'name','model_set','model_version','model_path','path','model_files','env_key','env_value'}),
 'command_entry_points':re.findall(r'[\w./-]+\.py',command),'command_flags':flags,'model_environment':env,
 'image_reference':image,'role_structure':type(roles).__name__,'new_task_started':False}
# Preserve a safe full command with known secret assignments withheld, for manual config review.
safe_command=re.sub(r'(?i)\b([A-Z0-9_]*(?:TOKEN|PASSWORD|SECRET|API_KEY)[A-Z0-9_]*)=("[^"]*"|\x27[^\x27]*\x27|[^\s;]+)',r'\1=[redacted]',command)
result['launch_command']=safe_command
result['active_launch_settings']=parse_launch(command)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['launch_command']},ensure_ascii=True)[:18000])
