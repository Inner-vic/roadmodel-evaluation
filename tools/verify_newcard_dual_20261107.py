"""Verify native receipts and automatic delivery; never start inference."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

proof=remote('''import pathlib,json,sys,datetime,numpy as np
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4');sys.path[:0]=[str(release),'/mnt/data/bench-eval/service/gateway']
from pcm_eval.common import digest
from pcm_eval.evaluation import EvaluationRuntime
from bench_gateway.config import load
from bench_gateway.core import Store
rt=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(rt.config_file.read_text());tid='rm-aeedd04d020548c6aa58';task=rt.get(tid,cfg['owner']);run=rt.root/'runs'/tid
workflow=json.loads((run/'workflow.json').read_text());result=json.loads((run/'progress.json').read_text())
jobs={j['kind']:j for j in workflow['jobs']};chains={j['kind']:j for j in result['chains']}
model=chains['model_accuracy']['execution'];compile=chains['compile_alignment']['execution']
expected=[['2026-07-01_16-28-21_HLX33B166R1997378',29],['2026-07-01_16-28-21_HLX33B166R1997378',30]]
assert workflow['scope']=='validation' and jobs['compile_alignment']['plan']['frames']==expected
assert jobs['model_accuracy']['plan']['heads']['occ']==expected
normalized=json.loads(pathlib.Path(model['result_file']).read_text());native=json.loads((run/'effective/model_accuracy/execution/native/result.json').read_text())
assert normalized['coverage']['observed']['model_samples']==2 and normalized['provenance']['native_run']['selected_frames']==expected
assert native['execution_location']=='lpai' and native['workstation_data_used'] is False and native['plan_sha256']==jobs['model_accuracy']['plan']['plan_sha256']
assert all(digest(pathlib.Path(x['result_file']))==x['result_sha256'] and x['state']=='succeeded' for x in (model,compile))
compiled=json.loads(pathlib.Path(compile['result_file']).read_text());cp_root=run/'effective/compile_alignment/execution/native-scope/compile'
progress=json.loads((cp_root/'progress.json').read_text());assert progress['completed_frames']==progress['expected_frames']==2 and progress['state']=='SUCCEEDED'
pointer=json.loads((run/'effective/compile_alignment/shared-queue.json').read_text())
gateway=load(pathlib.Path('/mnt/data/bench-eval/service/gateway'));hardware=Store(gateway['state_root']).job(cfg['owner'],pointer['hardware_job'])
assert hardware['state']=='SUCCEEDED' and hardware['payload']['roadmodel_task']==tid and hardware['payload']['entry_runtime']=='roadmodel-evaluation/v1'
assert hardware['result']['detail']==compile
publication=result['combined_report'];assert publication['state']=='VERIFIED' and publication['table_count']==5 and publication['all_metric_cells_match'] and publication['requester_read_verified'] and publication['workflow_sha256']==workflow['workflow_sha256']
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
prior=json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-live-20261107-r4/reservation.json').read_text())
assert digest(full/'compile/checkpoint.npz')==prior['old_checkpoint_sha256']
with np.load(full/'compile/checkpoint.npz',allow_pickle=False) as cp:assert int(cp['completed'])==2151
assert not (full/'compile/controller.lock').exists() and not pathlib.Path('/home/lixiang/bench-eval/locks/bench.lock').exists()
assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in (2178,2179))
seconds=lambda x:(datetime.datetime.fromisoformat(x['finished_at'])-datetime.datetime.fromisoformat(x['started_at'])).total_seconds()
proof={'task':tid,'event_id':task['event_id'],'message_id':task['message_id'],'scope':'validation','frames':expected,
 'model':{'state':chains['model_accuracy']['state'],'native_state':native['state'],'observed_frames':2,'elapsed_seconds':seconds(model),'execution_location':'lpai','workstation_data_used':False,'incomplete_heads':native['incomplete_heads'],'result_sha256':model['result_sha256']},
 'compile':{'state':chains['compile_alignment']['state'],'completed_frames':2,'elapsed_seconds':seconds(compile),'hardware_fifo_job':hardware['id'],'result_sha256':compile['result_sha256'],'bench_lock_released':True},
 'report':publication,'full_checkpoint_unchanged':True,'full_completed_frames':2151,
 'card_status':task['state'],'two_link_inputs':task['inputs'],'next_card_selection':cfg['validation_selection']}
pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107/end-to-end-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
''')
out=Path(__file__).resolve().parents[1]/'evidence/new-card-dual-20261107/end-to-end-proof.json'
out.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(proof))
