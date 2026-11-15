"""Live source/preparation/publication validation in an isolated work-order DB."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from connect_active_entry import call
from pcm_eval.common import save_json
from evaluation_agent import prepare_materials, exchange, publish

RELEASE = '/mnt/data/bench-eval/service/roadmodel/evaluation-20261105-r1'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/evaluation-20261105-r1'


def remote(body, timeout=60):
    return json.loads(call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+body, timeout=timeout))


def main(resume_source_error=False):
    cfg = json.loads((ROOT/'evidence/evaluation-20261105/agent-config.json').read_text(encoding='utf-8'))
    cfg['state_root'] = STAGE+'/live-preparation-state'
    cfg['local_state'] = str(ROOT/'.runtime/evaluation-agent-verification')
    created = remote('RESUME_SOURCE_ERROR='+repr(resume_source_error)+'\n'+'''import pathlib,sys,json,time
sys.path.insert(0,RELEASE)
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.common import load_json
cfg=load_json(pathlib.Path(RELEASE)/'configs/evaluation.json')
stage=pathlib.Path(STAGE);journal=stage/'live-preparation.json'
rt=EvaluationRuntime(stage/'live-preparation-state')
if journal.exists():
 assert RESUME_SOURCE_ERROR,'Prior isolated verification exists; reconcile it'
 with rt.connect() as c:
  row=c.execute('select * from materials').fetchone()
  assert row and row['state']=='PREPARED' and json.loads(row['detail']).get('error') and 'workflow' not in json.loads(row['detail'])
  assert not c.execute('select 1 from publications').fetchone()
  assert not list(rt.root.rglob('inference.json'))
  save=stage/'source-query-error.json';assert not save.exists();save.write_text(row['detail'])
  c.execute("update materials set state='READING' where task=?",(row['task'],))
  print(json.dumps(dict(row)))
 raise SystemExit(0)
task=rt.open(cfg['owner'],'feishu:oc_development:'+cfg['owner'])
value={'model_delivery_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'dataset_url':'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1'}
# Development validation is explicitly operator-issued, with no callback event.
with rt.connect() as c:
 c.execute("update tasks set state='QUEUED',inputs=?,notice=? where id=?",(json.dumps(value),'隔离开发联调，未模拟用户点击',task['id']))
 for kind in ('model_accuracy','compile_alignment'):c.execute("insert into chains(task,kind,state) values(?,?,'PENDING')",(task['id'],kind))
rt.tick(spawn=lambda *a,**k:(_ for _ in ()).throw(ValueError('No inference allowed')))
row=rt.claim_materials(cfg['agent_id'],cfg['owner'])
journal.write_text(json.dumps({'state':'READING','task':task['id'],'real_submit_callback_verified':False,'new_inference_frames':0}))
print(json.dumps(row))
''')
    prepare_materials(cfg, created)
    remote('''import pathlib,sys,json
sys.path[:0]=[RELEASE,RELEASE+'/tools']
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.common import save_json
from evaluation_worker import main
stage=pathlib.Path(STAGE);rt=EvaluationRuntime(stage/'live-preparation-state')
with rt.connect() as c:row=c.execute("select * from materials where state='PREPARED'").fetchone()
assert row,'Material source was not prepared'
workflow=json.loads(row['detail'])['workflow']
assert workflow['scope']=='full' and all(not j['ready'] for j in workflow['jobs']),'This validation must not infer any frame'
run=rt.root/'runs'/row['task'];run.mkdir(parents=True);save_json(run/'workflow.json',workflow)
main(run)
print(json.dumps({'state':'WAITING_REPORT','task':row['task']}))
''', timeout=90)
    row = exchange(cfg, 'claim_publication')
    if not row:
        raise ValueError('Isolated report queue did not receive the generated report')
    detail = publish(cfg, row)
    exchange(cfg, 'complete_publication', task=row['task'], digest=row['digest'], detail=detail)
    proof = remote('''import pathlib,sys,json
sys.path.insert(0,RELEASE)
from pcm_eval.evaluation import EvaluationRuntime
stage=pathlib.Path(STAGE);rt=EvaluationRuntime(stage/'live-preparation-state')
with rt.connect() as c:row=c.execute('select * from tasks order by created desc limit 1').fetchone()
task=rt.get(row['id'],row['owner']);run=rt.root/'runs'/task['id']
workflow=json.loads((run/'workflow.json').read_text());result=json.loads((run/'progress.json').read_text())
assert task.get('report_url') and result['combined_report']['requester_read_verified']
assert not list(run.rglob('inference.json')),'Unexpected inference receipt'
proof={'state':'VERIFIED','scope':'isolated_preparation_and_publication','report_url':task['report_url'],
 'chain_states':{c['kind']:c['state'] for c in task['chains']},'table_count':result['combined_report']['table_count'],
 'all_metric_cells_match':result['combined_report']['all_metric_cells_match'],'requester_read_verified':True,
 'dataset_preparation_state':workflow['dataset_preparation']['state'],
 'new_inference_frames':0,'bench_contacted':False,'real_submit_callback_verified':False}
(stage/'live-preparation-verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False))
''')
    save_json(ROOT/'evidence/evaluation-20261105/live-preparation-verification.json', proof)
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-source-error', action='store_true')
    args = parser.parse_args()
    main(args.resume_source_error)
