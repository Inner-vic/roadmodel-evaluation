"""Recover timing/report finalization from retained six-frame outputs, no inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE = r'''import pathlib,json,sys,statistics,datetime,sqlite3,os,numpy as np
out=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-pipeline-timing-20261030-r1')
deploy=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-pipeline-timing-20261030-r1')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
production=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
sys.path.insert(0,str(production))
from pcm_eval.common import digest,load_json,save_json,utc_now
from pcm_eval.compile_aggregate import aggregate_compile
from pcm_eval.compile_report import render_sections
from pcm_eval.business_layout import compile_business_sections
from pcm_eval.active_execution import process_alive
from pcm_eval.gt_adapter import Counts
assert not (out/'verification.json').exists(),'Finalization already completed; do not overwrite'
reservation=load_json(out/'reservation.json');indices=reservation['indices'];trace=load_json(out/'trace-events.json')
assert len(indices)==6 and all(e['ok'] for e in trace)
assert load_json(out/'progress.json')['native_completed']==6
dispatch=load_json(deploy/'dispatch.json');proc=pathlib.Path('/proc')/str(dispatch['pid'])/'cmdline'
assert not proc.exists() or not proc.read_bytes(),'Benchmark has not exited'
assert digest(full/'compile/checkpoint.npz')==reservation['checkpoint_sha256']
assert digest(full/'compile/progress.json')==reservation['progress_sha256']
assert digest(full/'STOP_AFTER_FRAME')==reservation['stop_marker_sha256']
counts={b+'_'+k:np.zeros(shape,np.int64) for b in ('board','onnx') for k,shape in (('confusion',(27,27)),('ds',(3,)))}
frames=[];details=[];ends=[];all_events_by_frame={};GT_ready=[]
for index in indices:
 root=full/'compile/frames'/('frame_%06d'%index);frames.append(str(root))
 events=[e for e in trace if e['index']==index];all_events_by_frame[index]=events
 sums={category:sum(e['seconds'] for e in events if e['category']==category) for category in {e['category'] for e in events}}
 assert sum(e['category']=='ONNX_session_run' for e in events)==1
 assert sum(e['category']=='InferenceExec_launch_and_execution' for e in events)==5
 assert sum(e['category']=='public_output_download' for e in events)==18
 receipt=load_json(root/'execution_receipt.json');ex=load_json(root/'execution.json');GT=load_json(root/'gt_frame_receipt.json')
 proof=load_json(root/'single-counts-receipt.json');public=load_json(root/'public-outputs.json');all_outputs=load_json(root/'all-subnet-outputs.json')
 assert receipt['state']=='succeeded' and receipt['selected_frames']==[reservation['frames'][indices.index(index)]] and receipt['release']['state']=='RELEASED'
 assert receipt['runtime']['onnx_provider']=='CPUExecutionProvider'
 assert len(all_outputs)==49 and all(v['finite'] for v in all_outputs.values()) and len(public)==18
 assert proof['execution_sha256']==digest(root/'execution_receipt.json') and proof['GT_sha256']==digest(root/'gt_frame_receipt.json') and proof['counts_sha256']==digest(root/'single-counts.npz')
 assert proof['original_GT_body_sha256']=='e86c3dfaed12b005780a23da6f8a3d3f89f909f3610f91d9cd92437022997187' and proof['GT_backend']=='memory_optimized'
 with np.load(root/'single-counts.npz',allow_pickle=False) as cp:
  for name in counts:counts[name]+=cp[name]
  for b in ('board','onnx'):
   c=Counts();c.confusion=cp[b+'_confusion'].copy();c.ds=cp[b+'_ds'].copy();c.num_frames=1
   assert c.summary()==load_json(root/('gt_metrics_'+b+'.json'))
   assert digest(root/(b+'_final_occ.npy'))==GT['predictions'][b]['sha256']
 for p in public:
  m=all_outputs[p['tensor']];path=root/('sub'+str(m['stage']))/p['filename']
  assert path.stat().st_size==p['nbytes'] and digest(path)==m['sha256']
  assert np.isfinite(np.fromfile(path,dtype=m['dtype'])).all()
 cleanup=[e for e in events if e['category']=='board_cleanup'];assert len(cleanup)==1
 ends.append(cleanup[0]['end'])
 data_event=next(e for e in events if e['category']=='data_prepare_and_export')
 prepare_event=next(e for e in events if e['category']=='prepare_including_ONNX')
 GT_ready.append((root/'single-counts-receipt.json').stat().st_mtime)
 details.append({'index':index,'frame':receipt['selected_frames'][0],'timings':sums,
  'prepare_span_seconds':prepare_event['end']-data_event['start'],
  'board_seconds_to_tensor_comparison':ex['optimized_board_seconds_before_diagnostic'],
  'board_cleanup_seconds':sums['board_cleanup'],'GT_seconds':proof['GT_seconds'],
  'downloaded_bytes':ex['public_tensor_bytes_downloaded'],'output_checks':49,'public_comparisons':18})
with np.load(out/'counts-checkpoint.npz',allow_pickle=False) as cp:
 assert int(cp['completed'])==6 and all(np.array_equal(counts[name],cp[name]) for name in counts)
modelroot=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01')
config=load_json(modelroot/'full-report-coordinator.json')
import time
report_start=time.monotonic()
report=aggregate_compile(frames,reservation['frames'],out/'report-recovered',request=config['report']['compile_execution_request'],
 run_id='bounded-pipeline-timing-6',checkpoint=out/'counts-checkpoint.npz')
xml='<document>'+''.join(s['xml'] for s in compile_business_sections(report,render_sections(report)))+'</document>'
import xml.etree.ElementTree as ET;ET.fromstring(xml);(out/'report-recovered/report.xml').write_text(xml)
report_seconds=time.monotonic()-report_start
assert report['result_validity']=='validated'
assert sum(len(x['outputs']) for x in load_json(out/'report-recovered/cosine_summary.json')['frames'])==108
progress=load_json(full/'compile/progress.json');assert progress['state']=='STOPPED_AT_BOUNDARY' and progress['completed_frames']==2151
assert not process_alive(progress['pid'],'formal_compile_remote.py') and not (full/'compile/controller.lock').exists()
coordinator=load_json(modelroot/'full-report-coordinator.dispatch.json');p=pathlib.Path('/proc')/str(coordinator['pid'])/'cmdline'
assert not p.exists() or not p.read_bytes()
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/runtime.sqlite3') as db:
 rows=db.execute('SELECT detail FROM chains WHERE task=?',('rm-a361dbe3df384cccb053',)).fetchall()
 assert len(rows)==2 and all(json.loads(row[0]).get('user_paused') for row in rows)
averages={k:statistics.mean(d['timings'].get(k,0) for d in details) for k in sorted({k for d in details for k in d['timings']})}
intervals=[b-a for a,b in zip(ends,ends[1:])]
started=datetime.datetime.fromisoformat(load_json(out/'started.json')['started_at']).timestamp()
proof={'state':'VERIFIED','indices':indices,'frames':details,'average_stage_seconds':averages,
 'board_completion_intervals_seconds':intervals,'mean_completion_interval_seconds':statistics.mean(intervals),
 'mean_later_three_completion_intervals_seconds':statistics.mean(intervals[-3:]),
 'mean_prepare_span_seconds':statistics.mean(d['prepare_span_seconds'] for d in details),
 'mean_board_seconds_to_tensor_comparison':statistics.mean(d['board_seconds_to_tensor_comparison'] for d in details),
 'mean_GT_compute_seconds':statistics.mean(d['GT_seconds'] for d in details),
 'wall_seconds_from_job_start_to_all_GT_results_ready':max(GT_ready)-started,
 'report_generation_seconds_after_schema_path_recovery':report_seconds,
 'timing_method':'Exact recorded monotonic event durations; cadence uses board-cleanup completion events; startup-to-GT-ready uses UTC/file timestamps. Initial in-memory summary lost on schema-path error, no inference repeated to reconstruct it.',
 'report_recovered_without_inference':True,'initial_schema_path_failure':str(out/'failure.json'),
 'new_ONNX_inferences':6,'new_NPU_chains':6,'InferenceExec_launches':30,'previously_completed_frames_repeated':0,
 'all49_checks_and18_comparisons_preserved':True,'metric_definitions_unchanged':True,
 'report_validity':report['result_validity'],'report_tensor_records':108,'Feishu_writes':0,
 'full_evaluation_resumed':False,'production_prefix':2151,'production_checkpoint_unchanged':True,
 'pause_marker_unchanged':True,'report_coordinator_still_stopped':True,'Runtime_still_user_paused':True,
 'new_results_retained_for_future_resume':True,'finished_at':utc_now()}
save_json(out/'verification.json',proof)
save_json(out/'recovery.json',{'state':'VERIFIED_WITHOUT_INFERENCE','initial_failure_sha256':digest(out/'failure.json'),
 'verification_sha256':digest(out/'verification.json'),'new_inference_frames':0})
save_json(out/'progress.json',{'state':'VERIFIED','native_completed':6,'maximum_native_frames':6,
 'full_evaluation_resumed':False,'initial_report_schema_failure_recovered':True},overwrite=True)
print(json.dumps(proof))
'''

if __name__ == '__main__':
    print(call("""import pathlib,sys
p=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-pipeline-timing-20261030-r1/finalize-report.py')
assert not p.exists(),'Inspect existing recovery; do not repeat'
raw=sys.stdin.buffer.read();p.write_bytes(raw);exec(compile(raw,str(p),'exec'))
""", CODE.encode()))
