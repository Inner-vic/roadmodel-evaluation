"""Read-only observation of the new real card on the initialized space."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/rebuild-initialization-20261109'
CODE='''import pathlib,json,sys,time,hashlib
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-initialized-20261109-r7');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text())
assert cfg['profile']['space_uid']==BASELINE['space_uid']
with r.connect() as db:
 rows=[dict(x) for x in db.execute('select id,state,event_id,message_id,created,updated from tasks where owner=? and updated>=? and event_id is not null order by updated desc',(cfg['owner'],BASELINE['checked_at']))]
 assert len(rows)<=1,'More than one new submitted card; do not guess'
 value={'checked_at':time.time(),'new_tasks':rows,'space_uid':cfg['profile']['space_uid'],'scope':cfg['validation_selection']}
 if rows:
  task=r.get(rows[0]['id'],cfg['owner']);run=r.root/'runs'/task['id']
  value['task']={k:task.get(k) for k in ('id','state','message_id','event_id','notice','report_url','options','created','updated')}
  value['chains']=[{'kind':c['kind'],'state':c['state'],'detail':c['detail']} for c in task['chains']]
  for table in ('materials','publications'):
   row=db.execute('select state,detail from '+table+' where task=?',(task['id'],)).fetchone()
   if row:
    detail=json.loads(row['detail']) if row['detail'] else {}
    value[table]={'state':row['state'],'detail':detail if not detail.get('workflow') else {'workflow_prepared':True}}
  p=run/'effective/model_accuracy/execution/native-progress.json'
  if p.exists():value['native_progress']=json.loads(p.read_text())
  p=run/'workflow.json'
  if p.exists():
   w=json.loads(p.read_text());value['frames']={j['kind']:(j.get('plan') or {}).get('heads',(j.get('plan') or {}).get('frames')) for j in w['jobs']}
  value['actions']=[dict(x) for x in db.execute('select action,state from space_actions where task=?',(task['id'],))]
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
value['old_full_checkpoint_sha256']=hashlib.sha256((full/'compile/checkpoint.npz').read_bytes()).hexdigest()
assert value['old_full_checkpoint_sha256']==BASELINE['old_full_checkpoint_sha256'] and not (full/'compile/controller.lock').exists()
print(json.dumps(value))
'''

def main():
    before=json.loads((OUT/'small-card-before.json').read_text())
    value=remote('BASELINE='+repr(before)+'\n'+CODE,timeout=45)
    agent=json.loads((ROOT/'.runtime/evaluation-initialized-agent/status.json').read_text())
    value['agent']={k:agent.get(k) for k in ('state','pid','updated','reason')}
    (OUT/'small-card-latest.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    if value.get('task'):
        path=OUT/'small-card-task.json'
        if not path.exists():path.write_text(json.dumps(value['task'],indent=2),encoding='utf-8')
        else:assert json.loads(path.read_text())['id']==value['task']['id']
    compact={k:v for k,v in value.items() if k not in ('chains','materials','publications','native_progress')}
    compact['chains']=[{'kind':x['kind'],'state':x['state'],'stage':x['detail'].get('stage'),'issues':x['detail'].get('issues')} for x in value.get('chains',[])]
    for key in ('materials','publications'):
        if key in value:compact[key]={'state':value[key]['state'],'error':value[key]['detail'].get('error')}
    if 'native_progress' in value:compact['native_progress']={k:value['native_progress'].get(k) for k in ('finished','alive','progress','error_log')}
    print(json.dumps(compact,ensure_ascii=True))

if __name__=='__main__':main()
