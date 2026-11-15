"""Bounded retained-output validation; full native evaluation stays paused."""
import os,sys,json,time,multiprocessing,statistics,hashlib,textwrap
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.parallel_gt import initialize_gt_worker,process_retained_gt,OrderedCheckpoint
from pcm_eval.common import load_json,save_json,digest,utc_now

FULL=Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')


def prepare_jobs(out,indices):
    source=FULL/'code/formal_compile_remote.py';s=source.read_text();a=s.index("            receipt = load(root / 'execution_receipt.json')");b=s.index('            cursor += 1',a)
    body=textwrap.dedent(s[a:b]);return [{'legacy_worker':str(source),'legacy_worker_sha256':digest(source),'frozen_request':str(FULL/'request.json'),
       'frozen_request_sha256':digest(FULL/'request.json'),'GT_body_sha256':hashlib.sha256(body.encode()).hexdigest(),
       'frame_index':i,'source_root':str(FULL/'compile/frames'/f'frame_{i:06d}'),'output':str(out/f'job-{seq:02d}-frame-{i:06d}'),'output_root':str(out)} for seq,i in enumerate(indices)]


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    p=load_json(FULL/'compile/progress.json');assert p['state']=='STOPPED_AT_BOUNDARY' and p['completed_frames']==2151
    before=digest(FULL/'compile/checkpoint.npz');indices=[19,419,1001,1499,1917,2051,2149,8333]
    inputs=prepare_jobs(out/'inputs',indices);save_json(out/'manifest.json',{'indices':indices,'frozen_request_sha256':inputs[0]['frozen_request_sha256'],'checkpoint_before':before,'native_evaluation_paused':True})
    results=[]
    for backend,workers,threads in (("reference",1,1),("memory_optimized",1,1),("memory_optimized",2,1),("memory_optimized",4,1)):
        case=out/f'{backend}-workers-{workers}-threads-{threads}';case.mkdir();jobs=prepare_jobs(case,indices);start=time.perf_counter();completed={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_gt_worker,initargs=(threads,backend)) as pool:
            futures={pool.submit(process_retained_gt,j):seq for seq,j in enumerate(jobs)}
            for future in as_completed(futures):
                completed[futures[future]]=future.result()
                save_json(case/'progress.json',{'completed':len(completed),'total':len(jobs),'updated_at':utc_now()},overwrite=(case/'progress.json').exists())
        wall=time.perf_counter()-start;ordered=[completed[i] for i in range(len(jobs))]
        checkpoint=OrderedCheckpoint(case/'counts-checkpoint.npz')
        for ordinal,r in enumerate(ordered):checkpoint.commit(ordinal,r)
        row={'backend':backend,'workers':workers,'threads_per_worker':threads,'frames':len(jobs),'wall_seconds_including_startup':wall,'wall_seconds_per_frame':wall/len(jobs),
             'mean_worker_compute_seconds':statistics.mean(r['GT_seconds'] for r in ordered),'all_voxel_labels_and_metrics_exact':True,'counts_checkpoint':str(checkpoint.path),'counts_sha256':digest(checkpoint.path)}
        results.append(row);save_json(out/'progress.json',{'state':'BENCHMARKING','cases':results},overwrite=(out/'progress.json').exists())
        assert digest(FULL/'compile/checkpoint.npz')==before and load_json(FULL/'compile/progress.json')['state']=='STOPPED_AT_BOUNDARY'
    import numpy as np
    with np.load(results[0]['counts_checkpoint'],allow_pickle=False) as original:
        for row in results[1:]:
            with np.load(row['counts_checkpoint'],allow_pickle=False) as current:
                assert set(original.files)==set(current.files) and all(np.array_equal(original[k],current[k]) for k in original.files)
    best=min(results,key=lambda r:r['wall_seconds_per_frame']);save_json(out/'verification.json',{'state':'VERIFIED','cases':results,'selected_workers':best['workers'],
         'selected_threads':best['threads_per_worker'],'speedup_including_startup':results[0]['wall_seconds_per_frame']/best['wall_seconds_per_frame'],
         'all_voxel_labels_metrics_and_counts_exact':True,'native_inference_frames':0,'production_checkpoint_unchanged':True,'full_evaluation_resumed':False,'finished_at':utc_now()})
    print(json.dumps(load_json(out/'verification.json')),flush=True)

if __name__=='__main__':main()
