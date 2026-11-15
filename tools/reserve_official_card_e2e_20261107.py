"""Freeze two unexecuted validation frames on the new, unsubmitted UI card."""
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
CODE = '''import pathlib,json,sys,time,hashlib
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text())
with r.connect() as db:
 rows=[dict(x) for x in db.execute("select * from tasks where owner=? and state='DRAFT' and created>? order by created desc",(cfg['owner'],time.time()-600))]
assert len(rows)==1,[(x['id'],x['state']) for x in rows]
t=rows[0];assert t['event_id'] is None and t['message_id']
options={'scope':'validation','selection':{'scene_count':1,'frames_per_scene':2,'scene_offset':38,'frame_offset':19}}
frames=[['2026-07-01_16-28-21_HLX33B166R1997378',33],['2026-07-01_16-28-21_HLX33B166R1997378',34]]
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
request=json.loads((full/'request.json').read_text());positions=[request['frames'].index(f) for f in frames]
assert positions==[2182,2183]
assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in positions)
assert not (full/'compile/controller.lock').exists()
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107');stage.mkdir(exist_ok=True)
assert not (stage/'reservation.json').exists()
with r.connect() as db:
 old=json.loads(db.execute('select detail from task_options where task=?',(t['id'],)).fetchone()['detail'])
 assert old=={'scope':'validation','selection':{'scene_count':1,'frames_per_scene':2}}
 assert db.execute("select state from tasks where id=?",(t['id'],)).fetchone()['state']=='DRAFT'
 db.execute('update task_options set detail=? where task=?',(json.dumps(options),t['id']))
proof={'task':t['id'],'message_id':t['message_id'],'scope':options,'frames':frames,'original_full_positions':positions,'old_checkpoint_sha256':hashlib.sha256((full/'compile/checkpoint.npz').read_bytes()).hexdigest(),'card_callback_not_yet_submitted':True}
(stage/'reservation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

if __name__ == '__main__':
    result = remote(CODE, timeout=40)
    (ROOT/'evidence/official-card-e2e-20261107/reservation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
