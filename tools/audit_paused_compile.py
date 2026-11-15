"""Read retained timings and CPU source only; never evaluate or resume work."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,re,statistics,sys,inspect,os,datetime
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');root=full/'compile';p=json.loads((root/'progress.json').read_text());assert p['state']=='STOPPED_AT_BOUNDARY'
pid=p['pid'];proc=pathlib.Path('/proc')/str(pid)/'cmdline';assert not proc.exists() or not proc.read_bytes();assert not (root/'controller.lock').exists()
marker=json.loads((full/'STOP_AFTER_FRAME').read_text());audit=pathlib.Path(marker['audit']);deploy=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-fast-20261030-r3')
with (deploy/'full-controller.log').open('rb') as f:f.seek(max(0,(deploy/'full-controller.log').stat().st_size-3000000));log=f.read().decode(errors='replace')
pattern=r'\[cpu_localagg\][^\n]*time=([\d.]+)s.*?\[cpu_final_occ\] wrote [^\n]*frame_(\d+)/(board|onnx)_temporary_splat'
kernels={}
for seconds,index,backend in re.findall(pattern,log,re.S):kernels.setdefault(int(index),{})[backend]=float(seconds)
rows=[]
for i in range(max(1916,p['completed_frames']-60),p['completed_frames']):
 frame=root/'frames'/f'frame_{i:06d}';gt=json.loads((frame/'single-counts-receipt.json').read_text());ex=json.loads((frame/'execution.json').read_text());prep=json.loads((frame/'prepare.json').read_text())
 rows.append({'index':i,'GT_total':gt['GT_seconds'],'board_total':ex['optimized_board_seconds_before_diagnostic'],'prepare':prep['preparation_seconds'],
              'localagg_kernels':kernels.get(i),'GT_finished_at':(frame/'gt_frame_receipt.json').stat().st_mtime})
matched=[r for r in rows if r['localagg_kernels'] and len(r['localagg_kernels'])==2]
out={'state':'PAUSED_READ_ONLY_AUDIT','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paused_prefix':p['completed_frames'],
 'frames_sampled':len(rows),'average_prepare_seconds':statistics.mean(r['prepare'] for r in rows),'average_board_seconds':statistics.mean(r['board_total'] for r in rows),
 'average_GT_total_seconds':statistics.mean(r['GT_total'] for r in rows),'kernel_log_pairs':len(matched),
 'average_two_localagg_kernel_seconds':statistics.mean(sum(r['localagg_kernels'].values()) for r in matched),
 'average_GT_outside_kernels_seconds':statistics.mean(r['GT_total']-sum(r['localagg_kernels'].values()) for r in matched),
 'steady_seconds_per_frame':(rows[-1]['GT_finished_at']-rows[0]['GT_finished_at'])/(len(rows)-1),
 'last_samples':rows[-4:],'process_stopped':True,'automatic_report_waiter_stopped':True,'new_inference_frames':0,'GT_recomputations':0}
package=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020');sys.path.insert(0,str(package))
import run_localagg_cpu as agg
out['CPU_functions']={name:inspect.getsource(getattr(agg,name)) for name in ('built_in_occ_xyz','combine_geosem','output_single_layer_road')}
out['local_aggregate_forward_source']=inspect.getsource(agg.local_aggregate_forward)[:11000]
out['CPU_counts']={'logical':os.cpu_count(),'affinity':len(os.sched_getaffinity(0))}
saved={k:v for k,v in out.items() if k not in ('CPU_functions','local_aggregate_forward_source')};(audit/'paused-timing-audit.json').write_text(json.dumps(saved,indent=2));print(json.dumps(out,ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE))
