"""Retire an unissued old-profile intent; preserve the real card and audit trail."""
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT/'evidence/official-card-e2e-20261107/reservation.json').read_text())['task']

if __name__ == '__main__':
    assert not any(json.loads(p.read_text()).get('task') == TASK for p in
                   (ROOT/'.runtime/evaluation-flow-agent/official-space-actions').glob('*/request.json'))
    result = remote('TASK='+repr(TASK)+'\n'+'''import json,pathlib,sys,time
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text())
run=r.root/'runs'/TASK;stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107')
assert not (run/'workflow.json').exists() and not (stage/'prestart-recovery.json').exists()
path=run/'space-preparation.json';journal=json.loads(path.read_text())
with r.connect() as db:
 actions=[dict(x) for x in db.execute('select * from space_actions where task=?',(TASK,))]
 assert len(actions)==1 and actions[0]['state']=='NEEDS_REVIEW'
 detail=json.loads(actions[0]['detail']);context=json.loads(actions[0]['context'])
 assert detail.get('real_browser_call_completed') is False and 'real_api_call_completed' not in detail
 assert context['profile']['space_uid'].startswith('lpai-ui:')
 assert journal['mount_plan']['space_phase']=='stopped'
 proof={'task':TASK,'reason':'Old in-memory Agent configuration, no platform write sent','prior_journal':journal,'rejected_action':actions[0],'recovered_at':time.time(),'sdk_requests_sent':0}
 (stage/'prestart-recovery.json').write_text(json.dumps(proof,indent=2))
 journal.pop('intent');journal['prestart_recovery']='prestart-recovery.json'
 path.write_text(json.dumps(journal,indent=2))
 db.execute("update materials set state='QUEUED',updated=? where task=? and state in ('READING','WAITING_SPACE')",(time.time(),TASK))
 db.execute("update tasks set notice=?,sent_digest='' where id=?",('后台配置已更新，原工单接续官方启动',TASK))
print(json.dumps({'task':TASK,'state':'READY_TO_RESUME','sdk_requests_sent':0,'card_reused':True,'evaluations_replayed':0}))
''', timeout=40)
    (ROOT/'evidence/official-card-e2e-20261107/prestart-recovery.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
