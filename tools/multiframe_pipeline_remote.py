"""Small-scope pipeline: verified reference prefetch, native batching, parallel GT."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import formal_compile_remote as controller
from multiframe_subnet_driver import Engine
from pcm_eval.batch_schedule import groups
from pcm_eval.common import digest,save_json
from pcm_eval.reference_prefetch import ReferencePrefetch,settings as reference_settings,requires_reference
from pcm_eval.gt_runtime import validated_settings,initialize_native_gt


def preflight():
    """Read-only validation of the frozen small scope; no sessions or workers started."""
    out=HERE.parent/'validation';r=json.loads((out/'reservation.json').read_text())
    assert 1<=len(r['indices'])<=8 and r['batch_size'] in (1,2,4,8)
    assert r['frames']==[controller.FRAMES[i] for i in r['indices']]
    assert digest(controller.FULL/'compile/checkpoint.npz')==r['checkpoint_sha256']
    assert (controller.FULL/'STOP_AFTER_FRAME').exists() and not (controller.c.ROOT/'controller.lock').exists()
    reference=reference_settings(HERE/'reference-runtime.json',digest(controller.FULL/'request.json'))
    gt=validated_settings(HERE/'gt-runtime.json')
    assert gt['frozen_request_sha256']==digest(controller.FULL/'request.json')
    cached=[]
    for i in r['indices']:
        root=controller.c.ROOT/'frames'/f'frame_{i:06d}'
        assert not (root/'execution.json').exists() and not (root/'execution_receipt.json').exists()
        if (root/'prepare.json').exists():
            prep=json.loads((root/'prepare.json').read_text());meta=json.loads((root/'input-reuse.json').read_text())['frame']['meta']
            assert prep['state']=='READY' and [meta['scene_name'],meta['frame_idx']]==controller.FRAMES[i]
            cached.append(i)
    return {'state':'READY_OFFLINE','scope_frames':len(r['indices']),'batch_size':r['batch_size'],
        'reusable_reference_indices':cached,'reference_workers':reference['workers'],'GT_workers':gt['workers'],
        'new_inference_frames':0,'full_resumed':False}


def wait_bench(out):
    import subprocess
    sys.path.insert(0,'/mnt/data/bench-eval/service/gateway')
    from bench_gateway.config import load
    from bench_gateway.core import Store
    started=time.monotonic()
    while True:
        q=json.loads(subprocess.run([str(controller.old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,
                       text=True,check=True,timeout=20).stdout)
        with Store(load('/mnt/data/bench-eval/service/gateway')['state_root']).connect() as db:
            busy=bool(db.execute("select id from jobs where state in ('QUEUED','RUNNING')").fetchone())
        if not (busy or q.get('active') or q.get('queued') or q.get('waiting') or (controller.old.RUNNER/'locks/bench.lock').exists()):
            return time.monotonic()-started
        save_json(out/'progress.json',{'state':'WAITING_BENCH','full_resumed':False,'wait_seconds':time.monotonic()-started},overwrite=True)
        if time.monotonic()-started>900:raise ValueError('Bench remains occupied; no dispatch')
        time.sleep(3)


def main():
    out=HERE.parent/'validation';r=json.loads((out/'reservation.json').read_text());indices=r['indices']
    if not 1<=len(indices)<=8 or r['batch_size'] not in (1,2,4,8):raise ValueError('Only bounded validation allowed')
    if any((controller.c.ROOT/'frames'/f'frame_{i:06d}'/'execution.json').exists() for i in indices):
        raise ValueError('Existing hardware attempt must be reconciled, never retried')
    assert (controller.FULL/'STOP_AFTER_FRAME').exists() and not (controller.c.ROOT/'controller.lock').exists()
    assert digest(controller.FULL/'compile/checkpoint.npz')==r['checkpoint_sha256']
    assert r['frames']==[controller.FRAMES[i] for i in indices]
    save_json(out/'started.json',{'pid':os.getpid(),'max_frames':len(indices),'full_resumed':False})
    started=time.monotonic();engine=None;reference=None;pool=None;pending=[];batches=[];wait=0
    try:
        engine=Engine()
        config=reference_settings(HERE/'reference-runtime.json',digest(controller.FULL/'request.json'))
        gt=validated_settings(HERE/'gt-runtime.json');os.environ['ROADMODEL_GT_BACKEND']=gt['backend']
        fresh=[i for i in indices if requires_reference(controller.c.ROOT/'frames'/f'frame_{i:06d}')]
        exporter=controller.make_pipeline(fresh,out/'canonical') if fresh else iter(())
        if config['workers']<=1:raise ValueError('Verified parallel reference workers required')
        reference=ReferencePrefetch(engine,indices,controller.FRAMES,exporter,controller.c.ROOT/'frames',out,config)
        pool=ProcessPoolExecutor(max_workers=gt['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_native_gt,initargs=(1,gt['backend']))
        initialization=time.monotonic()-started
        for ordinal,group in enumerate(groups(indices,r['batch_size'])):
            roots=[reference.get(i) for i in group['indices']]
            wait+=wait_bench(out)
            assert digest(controller.FULL/'compile/checkpoint.npz')==r['checkpoint_sha256']
            timing=engine.run_many(roots,r['run_id']+'-b'+str(ordinal),validation=True,
                                   validate_collector=r.get('validate_collector',True))
            batches.append({'indices':group['indices'],'timing':timing})
            for i,root in zip(group['indices'],roots):pending.append((i,root,pool.submit(controller.ensure_gt,i,root,True)))
        sums={};gt_results=[]
        for i,root,future in pending:
            counts=future.result()
            for b in counts:
                for kind,array in zip(('confusion','ds'),counts[b]):sums[b+'_'+kind]=sums.get(b+'_'+kind,0)+array
            oracle=json.loads((root/'GT-consistency-validation.json').read_text());assert oracle['state']=='VERIFIED'
            gt_results.append({'index':i,'GT_oracle':'VERIFIED'})
        import numpy as np
        with (out/'counts-checkpoint.npz').open('xb') as f:np.savez(f,completed=len(indices),**sums)
        assert digest(controller.FULL/'compile/checkpoint.npz')==r['checkpoint_sha256'] and (controller.FULL/'STOP_AFTER_FRAME').exists()
        save_json(out/'verification.json',{'state':'VERIFIED_BOUNDED_MULTIFRAME_PIPELINE','indices':indices,'frames':r['frames'],
            'batch_size':r['batch_size'],'batches':batches,'GT':gt_results,'reference_workers':config['workers'],
            'GT_workers':gt['workers'],'initialization_seconds':initialization,'elapsed_seconds':time.monotonic()-started,
            'bench_wait_seconds':wait,'full_checkpoint_unchanged':True,'full_resumed':False,
            'new_reference_frames':len(fresh),'retained_reference_frames':len(indices)-len(fresh),
            'diagnostic_collection_enabled':r.get('validate_collector',True),
            'native_frame_count':len(indices),'same_frame_serial_NPU_equivalence':'NOT_TESTED_NO_REPEAT'})
        save_json(out/'progress.json',{'state':'SUCCEEDED','completed':len(indices),'full_resumed':False},overwrite=True)
    except Exception as error:
        save_json(out/'progress.json',{'state':'NEEDS_REVIEW','error':repr(error),'full_resumed':False},overwrite=True)
        raise
    finally:
        if reference is not None:reference.close()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if engine is not None:engine.close()


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:print(json.dumps(preflight()))
    elif not sys.argv[1:]:main()
    else:raise ValueError('Unsupported arguments')
