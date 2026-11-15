import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
print(json.dumps(remote('''import sys,pathlib,json
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text());tid='rm-aeedd04d020548c6aa58';task=r.get(tid,cfg['owner'])
with r.connect() as db:
 actions=[dict(x) for x in db.execute('select id,action,state from space_actions where task=? order by updated',(tid,))]
 materials=db.execute('select state,detail from materials where task=?',(tid,)).fetchone()
run=r.root/'runs'/tid
value={'task':{k:task.get(k) for k in ('id','state','notice','event_id','report_url','options')},'chains':[{k:c.get(k) for k in ('kind','state')}|{'stage':c['detail'].get('stage'),'issues':c['detail'].get('issues')} for c in task['chains']],'actions':actions,'materials':{'state':materials['state']} if materials else None,'files':[p.name for p in run.iterdir()] if run.exists() else []}
(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107')/'latest-status.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
'''),ensure_ascii=False))
