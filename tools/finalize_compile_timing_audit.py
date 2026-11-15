"""Keep diagnosis evidence on the workstation; no execution changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,datetime,statistics,os
base=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/compile-timing-audit')
audit=json.loads(sorted(base.glob('frames-*.json'))[-1].read_text())
stream=json.loads(sorted(base.glob('stream-read-probe-*.json'))[-1].read_text())
prefetch=json.loads((base/'read-probe-1854.json').read_text())
stages=audit['stage_averages'];collection=sum(s['post_command_collection_seconds_approx'] for s in stages)
task=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');p=json.loads((task/'compile/progress.json').read_text());os.kill(p['pid'],0)
coordinator=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');dispatch=json.loads((coordinator/'full-report-coordinator.dispatch.json').read_text());os.kill(dispatch['pid'],0)
result={'state':'AUDIT_AND_READ_ONLY_PROBE_COMPLETE','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'frames_sampled':audit['frames_sampled'],'frame_seconds':audit['averages']['total'],
 'prepare_and_ONNX_seconds_approx':audit['averages']['export_and_prepare']+audit['averages']['admission'],
 'board_seconds_approx':audit['averages']['board_transfer_chain_compare_release'],
 'within_board_collection_seconds_approx':collection,'within_board_command_seconds_approx':sum(s['command_seconds_approx'] for s in stages),
 'logged_kernel_seconds':sum(s['kernel_seconds'] for s in stages),'GT_seconds':audit['averages']['GT'],
 'outputs_per_frame':sum(s['outputs'] for s in stages),'output_bytes_per_frame':sum(s['bytes'] for s in stages),
 'timestamp_precision_note':'Stage collection is inferred from local file mtimes. Guard claimed_at has one-second precision, so subsecond admission differences are not real negative waits.',
 'read_probe':stream,'prefetch_rejected':prefetch,
 'recommendations':[
  {'priority':1,'change':'Replace serial SFTP tensor reads with bounded binary SSH streams or one validated archive per stage; preserve all49 outputs and current size/SHA/finite checks.',
   'status':'Two real output tensors verified byte-for-byte; no measured full-frame acceleration yet.'},
  {'priority':2,'change':'Reuse the ORT session and immutable model metadata across independent temporal-disabled frames, retaining CPUExecutionProvider and ORT_DISABLE_ALL.',
   'status':'Source-confirmed per-frame session creation; acceleration and numerical equivalence not yet validated.'},
  {'priority':3,'change':'Pipeline future-frame CPU preparation and completed-frame GT with the exclusive board phase, retain per-frame Runner admission/release and exact frozen frame order.',
   'status':'Design option; not enabled or benchmarked.'}],
 'planning_estimate':{'candidate_frame_seconds':[25,30],'remaining_compile_hours':[(8337-p['completed_frames'])*25/3600,(8337-p['completed_frames'])*30/3600],
   'measured':False,'note':'Illustrative projection from removing most serial output-read latency; not an updated task ETA.'},
 'current_progress':p,'original_controller_alive':True,'original_PID':p['pid'],'report_coordinator_alive':True,
 'running_controller_modified':False,'native_inference_frames_for_audit':0,'GT_recomputations_for_audit':0,
 'transition_boundary':'An optimization needs an audited new controller and handoff at a confirmed frame checkpoint; do not overwrite loaded code, stop in-flight hardware, or replay finished frames.'}
out=base/'conclusion.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('read_probe','prefetch_rejected')},ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE))
