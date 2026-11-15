"""Freeze another two-frame validation scope before submitting the fresh card."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/card-acceptance-full-20261109/normal-card'
CODE='''import pathlib,json,sys,time
root=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-card-diagnostics-20261109-r10');sys.path.insert(0,str(root))
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.lpai_native_model import select_frames
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text())
assert cfg['validation_selection']=={'scene_count':1,'frames_per_scene':2,'frame_offset':4}
with r.connect() as db:
 rows=[dict(x) for x in db.execute("select * from tasks where state='DRAFT' and owner=? and created>? order by created desc",(cfg['owner'],time.time()-300))]
 assert len(rows)==1,'No unique fresh unsubmitted card'
 task=rows[0];assert task['event_id'] is None and task['message_id']
 prior=json.loads(pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-84ad94ff8e2f4343bab3/workflow.json').read_text())
 selected=select_frames(prior['dataset']['frames_by_head']['occ'],cfg['validation_selection'])
 previous=next(j for j in prior['jobs'] if j['kind']=='compile_alignment')['plan']['frames']
 assert len(selected)==2 and not set(map(tuple,selected))&set(map(tuple,previous))
 options={'scope':'validation','selection':cfg['validation_selection']}
 db.execute('update task_options set detail=? where task=?',(json.dumps(options),task['id']))
print(json.dumps({'task':task['id'],'message_id':task['message_id'],'selected_frames':selected,
 'prior_compiled_frames_preserved':previous,'before_submission':True,'options':options}))
'''
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'reservation.json').exists()
 result=remote(CODE,timeout=40)
 (OUT/'reservation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 agent=json.loads((ROOT/'.runtime/evaluation-initialized-agent/status.json').read_text())
 before={'checked_at':time.time(),'agent_pid':agent['pid'],'agent_state':agent['state'],
  'space_uid':'8106b82c-6a38-4dcb-b962-86fb91078535','space_name':'roadmodel-eval-fa60526293a9959e3bff',
  'old_full_checkpoint_sha256':'2c737cb749fb6268e4c1f8c630e86f60ebc83e9be490f19b6b8c1b4fca3f0bed'}
 (OUT/'small-card-before.json').write_text(json.dumps(before,indent=2));print(json.dumps(result))
