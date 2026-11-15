"""Synthetic array microbenchmark, not a model/GT-frame evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,sys,json,time,datetime,os,numpy as np
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');p=json.loads((full/'compile/progress.json').read_text());assert p['state']=='STOPPED_AT_BOUNDARY'
mem={line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))};assert mem['MemAvailable']>3*1024**2
sys.path[:0]=[str(full/'code'),'/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020']
import run_localagg_cpu as cpu
from gt_adapter import Counts
H,W,D=400,48,100;N=H*W*D;C=27;rng=np.random.default_rng(20261030)
logits=rng.random((N,C),dtype=np.float32);binary=rng.random(N,dtype=np.float32)
rows=[]
def measure(name,func):
 started=time.perf_counter();result=func();rows.append({'operation':name,'seconds':time.perf_counter()-started});return result
points=measure('construct_dense_xyz',lambda:cpu.built_in_occ_xyz(H,W,D));del points
geo=measure('combine_geosem_dense_multiply_concat_cast',lambda:cpu.combine_geosem(logits,binary));del logits,binary
pred=measure('semantic_argmax',lambda:geo.argmax(axis=1).astype(np.int32))
final=measure('single_layer_road_numpy_index_sum_max_masks',lambda:cpu.output_single_layer_road(geo,pred,H,W,D,use_density=False,output_bumper_top=False));del geo,pred
gt=np.full((H,W,D),26,np.int32);drive=np.zeros((H,W,D),np.uint8);indices=rng.choice(N,size=N//100,replace=False)
gt.flat[indices]=rng.integers(1,26,size=len(indices),dtype=np.int32);drive.flat[indices]=1;mask=~np.isin(gt,[0,15]);count=Counts()
measure('voxel_and_drivable_count_accumulation',lambda:count.update(final,gt,mask,drive));del gt,drive,mask,final,count
marker=json.loads((full/'STOP_AFTER_FRAME').read_text());audit=pathlib.Path(marker['audit'])
value={'state':'SYNTHETIC_ARRAY_BENCHMARK_COMPLETE','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'HWD':[H,W,D],'channels':C,
 'one_dense_float32_tensor_MiB':N*C*4/1024**2,'operations':rows,'total_measured_array_seconds':sum(x['seconds'] for x in rows),
 'actual_frame_attribution':False,'note':'Synthetic values, same array dimensions. Mechanism check only; these timings cannot replace a real-frame function profile.',
 'real_frame_inputs_used':False,'precision_scores_generated':False,'model_inference_frames':0,'GT_evaluation_frames':0,'task_resumed':False,'paused_prefix':p['completed_frames']}
(audit/'synthetic-array-benchmark.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
if __name__=='__main__':print(call(CODE,timeout=90))
