"""Observe the real card's work order without dispatching or replaying work."""
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT/'evidence/core-e2e-20261107/reservation.json').read_text())['task']
CODE = '''import pathlib,json,sys,time
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text());task=r.get(TASK,cfg['owner']);run=r.root/'runs'/TASK
with r.connect() as db:
 actions=[dict(x) for x in db.execute('select id,action,state from space_actions where task=? order by updated',(TASK,))]
 material=db.execute('select state from materials where task=?',(TASK,)).fetchone()
 publication=db.execute('select state,detail from publications where task=?',(TASK,)).fetchone()
value={'checked_at':time.time(),'task':{k:task.get(k) for k in ('id','state','notice','event_id','report_url','options','message_id')},
 'chains':[{k:c.get(k) for k in ('kind','state')}|{'stage':c['detail'].get('stage'),'issues':c['detail'].get('issues')} for c in task['chains']],
 'space_actions':actions,'material_state':material['state'] if material else None,'publication_state':publication['state'] if publication else None,'progress':{}}
for name in ['workflow.json','progress.json','evaluation.json','effective/model_accuracy/execution/native-progress.json','effective/compile_alignment/shared-queue.json']:
 p=run/name
 if p.exists():
  data=json.loads(p.read_text())
  if name=='workflow.json':value['frames']={j['kind']:j['plan'].get('heads',j['plan'].get('frames')) for j in data['jobs'] if j.get('plan')}
  else:value['progress'][name]={k:data[k] for k in ('state','alive','finished','progress','hardware_job') if k in data}
print(json.dumps(value))
'''

if __name__ == '__main__':
    result = remote('TASK='+repr(TASK)+'\n'+CODE, timeout=40)
    (ROOT/'evidence/core-e2e-20261107/latest-status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
