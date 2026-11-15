"""Audit this real card's native results and independently read its Feishu report."""
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT/'evidence/core-e2e-20261107/reservation.json').read_text())['task']
CODE = '''import pathlib,json,sys,datetime,hashlib,xml.etree.ElementTree as ET,numpy as np
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5');sys.path[:0]=[str(release),'/mnt/data/bench-eval/service/gateway']
from pcm_eval.common import digest
from pcm_eval.validation import require_result
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.evaluation_entry import render
from pcm_eval.delivery_source import LarkSource
from pcm_eval.feishu_update import table_values
from bench_gateway.config import load
from bench_gateway.core import Store
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text());task=r.get(TASK,cfg['owner']);run=r.root/'runs'/TASK
workflow=json.loads((run/'workflow.json').read_text());result=json.loads((run/'progress.json').read_text());jobs={j['kind']:j for j in workflow['jobs']};chains={j['kind']:j for j in result['chains']}
reservation=json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/core-e2e-20261107/reservation.json').read_text());expected=reservation['frames']
assert task['event_id'] and task['message_id']==reservation['message_id'] and workflow['scope']=='validation'
assert jobs['model_accuracy']['plan']['heads']['occ']==expected and jobs['compile_alignment']['plan']['frames']==expected
model=chains['model_accuracy']['execution'];compiled=chains['compile_alignment']['execution']
assert chains['model_accuracy']['state']=='PARTIAL' and chains['compile_alignment']['state']=='SUCCEEDED'
for item in (model,compiled):
 assert item['state']=='succeeded' and digest(pathlib.Path(item['result_file']))==item['result_sha256']
 require_result(json.loads(pathlib.Path(item['result_file']).read_text()))
model_data=json.loads(pathlib.Path(model['result_file']).read_text());native=json.loads((run/'effective/model_accuracy/execution/native/result.json').read_text())
assert model_data['coverage']['observed']['model_samples']==2 and model_data['provenance']['native_run']['selected_frames']==expected
assert native['executed_frames_by_head']=={'occ':expected} and native['execution_location']=='lpai' and native['workstation_data_used'] is False
assert native['head_statuses']['vision']['state']=='BLOCKED' and native['head_statuses']['vision']['completed_frames']==0
assert [x['code'] for x in native['head_statuses']['vision']['issues']]==['PER_IMAGE_GT_NOT_FOUND']*2
batch=json.loads((run/'effective/model_accuracy/execution/native/occ/batch-0000/execution_receipt.json').read_text())
assert batch['state']=='succeeded' and batch['exit_code']==0 and batch['selected_frames']==expected
cp_root=run/'effective/compile_alignment/execution/native-scope/compile';cp=json.loads((cp_root/'progress.json').read_text())
assert cp['state']=='SUCCEEDED' and cp['completed_frames']==cp['expected_frames']==2
pointer=json.loads((run/'effective/compile_alignment/shared-queue.json').read_text());gateway=load(pathlib.Path('/mnt/data/bench-eval/service/gateway'));hardware=Store(gateway['state_root']).job(cfg['owner'],pointer['hardware_job'])
assert hardware['state']=='SUCCEEDED' and hardware['payload']['roadmodel_task']==TASK and hardware['result']['detail']==compiled
report=chains['model_accuracy']['report'];atomic=json.loads(pathlib.Path(report['report_json']).read_text())
assert atomic['all_metrics']==model_data['metrics'] and atomic['vision_metrics']==[] and atomic['head_statuses']==native['head_statuses']
publication=result['combined_report'];assert publication['state']=='VERIFIED' and publication['requester_read_verified'] and publication['table_count']==5 and publication['all_metric_cells_match']
assert publication['workflow_sha256']==workflow['workflow_sha256'] and task['report_url']==publication['report_url']
with r.connect() as db:
 payload=json.loads(db.execute('select payload from publications where task=?',(TASK,)).fetchone()['payload'])
assert payload['data']['model_head_statuses']==native['head_statuses']
binding=cfg['workstation_lark'];response=LarkSource(executable=binding['executable'],profile=binding['profile']).query(['docs','+fetch','--doc',task['report_url'],'--detail','full'])
document=response['data']['document'];actual=ET.fromstring('<root>'+document['content']+'</root>');expected_doc=ET.fromstring('<root>'+payload['xml']+'</root>')
assert len(list(actual.iter('table')))==5 and table_values(actual)==table_values(expected_doc)
assert task['sent_digest']==hashlib.sha256(json.dumps(render(task),sort_keys=True).encode()).hexdigest()
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
assert digest(full/'compile/checkpoint.npz')==reservation['old_checkpoint_sha256']
with np.load(full/'compile/checkpoint.npz',allow_pickle=False) as old:assert int(old['completed'])==2151
assert not (full/'compile/controller.lock').exists()
assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in reservation['original_full_positions'])
seconds=lambda item:(datetime.datetime.fromisoformat(item['finished_at'])-datetime.datetime.fromisoformat(item['started_at'])).total_seconds()
proof={'state':'VERIFIED_WITH_EXPLICIT_GT_GAP','task':TASK,'real_card_event_id':task['event_id'],'message_id':task['message_id'],'frames':expected,'scope':'validation',
 'model':{'state':chains['model_accuracy']['state'],'actual_inference_frames':2,'native_process_seconds':batch['duration_seconds'],'dispatch_to_collection_seconds':seconds(model),'head_statuses':native['head_statuses'],'result_sha256':model['result_sha256']},
 'compile':{'state':'SUCCEEDED','actual_board_frames':2,'dispatch_to_collection_seconds':seconds(compiled),'hardware_fifo_job':hardware['id'],'result_sha256':compiled['result_sha256'],'bench_lock_present_at_check':pathlib.Path('/home/lixiang/bench-eval/locks/bench.lock').exists()},
 'report':{'url':task['report_url'],'revision':document['revision_id'],'tables':5,'all_metric_cells_match':True,'requester_read_verified':True,'original_card_backfill_verified':True,'vision_scores_fabricated':False},
 'old_full_checkpoint_unchanged':True,'old_full_completed_frames':2151,'scope_config_unchanged':cfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}}
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/core-e2e-20261107');(stage/'end-to-end-proof.json').write_text(json.dumps(proof,indent=2));(stage/'report-readback.json').write_text(json.dumps(document,indent=2));print(json.dumps(proof))
'''

if __name__ == '__main__':
    result = remote('TASK='+repr(TASK)+'\n'+CODE, timeout=150)
    (ROOT/'evidence/core-e2e-20261107/end-to-end-proof.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
