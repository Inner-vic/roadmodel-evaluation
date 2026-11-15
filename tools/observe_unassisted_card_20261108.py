"""Read-only observations for a new card; never dispatch, repair or mutate work."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/unassisted-card-e2e-20261108'
RELEASE='/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6'
FILES=['pcm_eval/space_api.py','pcm_eval/space_lifecycle.py','pcm_eval/evaluation.py','tools/evaluation_agent.py','tools/evaluation_space_actor.py']
CODE='''import pathlib,json,sys,hashlib,time
release=pathlib.Path(RELEASE);sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text())
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
value={'checked_at':time.time(),'config':{'execution_scope':cfg['execution_scope'],'validation_selection':cfg['validation_selection'],'space_backend':cfg['space_backend'],'bench_available':cfg['bench_available']},'source_sha256':{n:hashlib.sha256((release/n).read_bytes()).hexdigest() for n in FILES},'old_full_checkpoint_sha256':hashlib.sha256((full/'compile/checkpoint.npz').read_bytes()).hexdigest(),'old_full_controller_running':(full/'compile/controller.lock').exists()}
assert cfg['execution_scope']=='validation' and cfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}
with r.connect() as db:
 rows=[dict(x) for x in db.execute('select id,state,message_id,event_id,created from tasks where owner=? and created>? order by created',(cfg['owner'],SINCE))]
 value['new_tasks']=rows
 assert len(rows)<=1,'More than one new card; do not guess'
 if rows:
  task=r.get(rows[0]['id'],cfg['owner']);run=r.root/'runs'/task['id']
  value['task']={k:task.get(k) for k in ('id','state','notice','event_id','message_id','report_url','options','created','updated','inputs')}
  value['chains']=[{'kind':c['kind'],'state':c['state'],'stage':c['detail'].get('stage'),'issues':c['detail'].get('issues')} for c in task['chains']]
  value['actions']=[]
  for x in db.execute('select id,action,state,detail from space_actions where task=? order by updated',(task['id'],)):
   a=dict(x);d=json.loads(a.pop('detail'));a.update(provider=d.get('provider'),new_requests=d.get('new_requests'),real_api_call_completed=d.get('real_api_call_completed'),phase=d.get('observation',{}).get('space',{}).get('status',{}).get('phase'));value['actions'].append(a)
  m=db.execute('select state,detail from materials where task=?',(task['id'],)).fetchone();value['materials']={'state':m['state'],'detail':json.loads(m['detail']) if m['detail'] else {}} if m else None
  if value['materials'] and 'workflow' in value['materials']['detail']:value['materials'].pop('detail')
  publication=db.execute('select state from publications where task=?',(task['id'],)).fetchone();value['publication_state']=publication['state'] if publication else None
  if (run/'workflow.json').exists():
   workflow=json.loads((run/'workflow.json').read_text());value['frames']={j['kind']:j.get('plan',{}).get('heads',j.get('plan',{}).get('frames')) for j in workflow['jobs']}
  p=run/'effective/model_accuracy/execution/native-progress.json'
  if p.exists():
   native=json.loads(p.read_text());value['native_progress']={k:native.get(k) for k in ('alive','finished','progress')}
print(json.dumps(value))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    before=json.loads((OUT/'before.json').read_text());since=datetime.fromisoformat(before['checked_at']).timestamp()
    result=remote('RELEASE='+repr(RELEASE)+'\nFILES='+repr(FILES)+'\nSINCE='+repr(since)+'\n'+CODE,timeout=45)
    pinned=ROOT/'.runtime/evaluation-environment-desktop-20261106-r3'
    result['desktop_source_sha256']={n:hashlib.sha256((pinned/n).read_bytes()).hexdigest() for n in FILES}
    if args.baseline:
        assert not result['new_tasks'];target=OUT/'baseline.json';assert not target.exists()
    else:
        baseline=json.loads((OUT/'baseline.json').read_text())
        assert result['source_sha256']==baseline['source_sha256'] and result['desktop_source_sha256']==baseline['desktop_source_sha256']
        assert result['config']==baseline['config'] and result['old_full_checkpoint_sha256']==baseline['old_full_checkpoint_sha256']
        assert result['old_full_controller_running'] is False
        target=OUT/('status-'+str(int(result['checked_at']))+'.json')
        if result.get('task'):
            identity=OUT/'task.json'
            if identity.exists():assert json.loads(identity.read_text())['id']==result['task']['id']
            else:identity.write_text(json.dumps(result['task'],indent=2),encoding='utf-8')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'latest-status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
