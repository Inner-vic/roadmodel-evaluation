"""Attribute retained timing logs; never dispatches work or changes its metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE = r'''import pathlib,json,re,statistics,sys
out=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-pipeline-timing-20261030-r1')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
from pcm_eval.common import load_json,save_json,digest
v=load_json(out/'verification.json');assert v['state']=='VERIFIED'
assert not (out/'bottleneck-audit.json').exists(),'Use the existing audit'
native=[]
for index in v['indices']:
 row={'index':index,'stages':[]}
 for stage in range(5):
  text=(full/'compile/frames'/('frame_%06d'%index)/('sub%d.stdout.log'%stage)).read_text(errors='replace')
  values={}
  for key in ('loadEngine','createExecutionContext','Infer'):
   matches=re.findall(r'\[InferenceExec\] '+key+r' time:\s*(\d+)\s*us',text)
   assert len(matches)==1,(index,stage,key,matches);values[key+'_seconds']=int(matches[0])/1e6
  row['stages'].append(values)
 row['all_five_infer_seconds']=sum(x['Infer_seconds'] for x in row['stages'])
 row['all_five_load_and_context_seconds']=sum(x['loadEngine_seconds']+x['createExecutionContext_seconds'] for x in row['stages'])
 native.append(row)
client=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020/board_infer_client.py')
assert 'time.sleep(0.05)' in client.read_text()
metrics=v['average_stage_seconds'];t=v['mean_board_seconds_to_tensor_comparison']
proof={'state':'VERIFIED','verification_sha256':digest(out/'verification.json'),'native_log_rows':native,
 'mean_all_five_NPU_infer_seconds':statistics.mean(x['all_five_infer_seconds'] for x in native),
 'mean_all_five_model_load_and_context_seconds':statistics.mean(x['all_five_load_and_context_seconds'] for x in native),
 'mean_InferenceExec_call_wall_seconds':metrics['InferenceExec_launch_and_execution'],
 'mean_board_round_to_comparison_seconds':t,'client_remote_status_poll_seconds':0.05,
 'client_source_sha256':digest(client),'remote_public_download_calls_per_frame':18,
 'download_bytes_per_frame':704080,'mean_download_seconds':metrics['public_output_download'],
 'mean_remote_output_validation_seconds':metrics['board_output_validation'],
 'mean_remote_hash_command_seconds':metrics['board_hash_check'],
 'mean_CPU_ONNX_seconds':metrics['ONNX_session_run'],
 'extra_diagnostic_queue_precheck_seconds_per_frame':metrics['validation_queue_precheck'],
 'measured_cadence_including_diagnostic_checks_seconds':v['mean_completion_interval_seconds'],
 'clock_note':'Use workstation monotonic durations and relative NPU Infer times; board printed calendar timestamps are not used.',
 'conclusion':'Remaining critical path is board-side invocation/validation and many small remote round trips; CPU ONNX is the next constraint. GT overlaps with these stages. Do not treat isolated GT throughput as total pipeline speed.',
 'next_step':'Validate one board-side batch orchestrator retaining all five original executions, 49 checks, and 18 comparisons; batch verified input/output transport. Do not reduce frames or change GT/provider.',
 'native_inference_frames_in_this_audit':0,'full_evaluation_resumed':False}
save_json(out/'bottleneck-audit.json',proof)
print(json.dumps({k:val for k,val in proof.items() if k!='native_log_rows'}))
'''

if __name__ == '__main__':print(call(CODE))
