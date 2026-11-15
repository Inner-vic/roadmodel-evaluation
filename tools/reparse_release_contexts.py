import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.model_inputs import parse_launch

root=Path('evidence/model-evaluation/model-reference-20261027')
for name in ['trajectory-training-context.json','vision-training-context.json']:
 path=root/name;data=json.loads(path.read_text(encoding='utf-8'))
 parsed=parse_launch(data['launch_command']);data['active_launch_settings']=parsed
 data['command_flags']=parsed['flags']
 data['model_environment']={k:v for k,v in parsed['environment'].items() if k in data['model_environment']}
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'source':name,'flags':parsed['flags'],'head_mode':parsed['environment'].get('HEAD_MODE'),
                  'embed_dims':parsed['environment'].get('EMBED_DIMS'),'active_environment_keys':len(parsed['environment'])},ensure_ascii=True))
