"""Checkpoint-compatible full controller; explicit user pause always takes precedence."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import formal_compile_remote as original
from multiframe_subnet_driver import Engine
from pcm_eval.batch_schedule import groups
from pcm_eval.reference_prefetch import ReferencePrefetch,requires_reference,settings as reference_settings
from pcm_eval.gt_runtime import validated_settings,initialize_native_gt
TOTAL=len(original.FRAMES)


def finalize(counts,progress):
    progress('FINALIZING')
    frames=[original.c.load(original.c.ROOT/'frames'/f'frame_{i:06d}'/'cosine_summary.json')['frames'][0] for i in range(TOTAL)]
    names=[r['onnx_output_name'] for r in frames[0]['outputs']]
    tensors={name:{'num_frames':TOTAL,
        'mean_cosine':sum(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in frames)/TOTAL,
        'min_cosine':min(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in frames)} for name in names}
    original.save(original.c.ROOT/'cosine_summary.json',{'num_frames':TOTAL,'frames':frames,'tensors':tensors,
        'mean_cosine_all':sum(f['mean_cosine'] for f in frames)/TOTAL,'min_cosine_all':min(f['mean_cosine'] for f in frames)})
    for b in counts:original.save(original.c.ROOT/('metrics_'+b+'.json'),counts[b].summary())
    progress('SUCCEEDED',version=original.REQUEST['model_version'])


def configuration():
    value=json.loads((HERE/'multiframe-runtime.json').read_text())
    if value['frozen_request_sha256']!=original.sha(original.FULL/'request.json'):
        raise ValueError('Multi-frame validation belongs to a different full request')
    width=value['batch_size']
    if type(width) is not int or width not in (2,4,8):raise ValueError('Unvalidated batch width')
    proof_path=Path(value['verification_path'])
    if original.sha(proof_path)!=value['verification_sha256']:raise ValueError('Native validation changed')
    proof=json.loads(proof_path.read_text())
    if (proof['state']!='VERIFIED_BOUNDED_MULTIFRAME_PIPELINE' or proof['batch_size']!=width
            or not proof['full_checkpoint_unchanged'] or proof['full_resumed']
            or not 0<len(proof['GT'])==proof['native_frame_count']==len(proof['indices'])
            or not all(x['GT_oracle']=='VERIFIED' for x in proof['GT'])):
        raise ValueError('Multi-frame native/GT validation is incomplete')
    request_path=Path(value['verification_request_path'])
    if original.sha(request_path)!=value['verification_request_sha256']:raise ValueError('Validation input changed')
    validation=json.loads(request_path.read_text());request=original.c.load(original.FULL/'request.json')
    if (validation['indices']!=proof['indices'] or validation['frames']!=proof['frames']
            or validation['frames']!=[request['frames'][i] for i in validation['indices']]):
        raise ValueError('Validation frames belong to different data')
    for name,expected in value['code_sha256'].items():
        path=(HERE/name).resolve();path.relative_to(HERE.resolve())
        if original.sha(path)!=expected:raise ValueError('Validated candidate code changed: '+name)
    return value


def preflight():
    value=configuration()
    reference=reference_settings(HERE/'reference-runtime.json',original.sha(original.FULL/'request.json'))
    gt=validated_settings(HERE/'gt-runtime.json')
    if gt['frozen_request_sha256']!=original.sha(original.FULL/'request.json'):raise ValueError('GT request mismatch')
    return {'state':'READY_USER_PAUSED' if (original.FULL/'STOP_AFTER_FRAME').exists() else 'READY',
        'batch_size':value['batch_size'],'reference_workers':reference['workers'],'GT_workers':gt['workers'],
        'new_inference_frames':0,'automatic_resume':False}


def main():
    if (original.FULL/'STOP_AFTER_FRAME').exists():raise ValueError('Paused by user; no native dispatch')
    value=configuration();width=value['batch_size']
    reference_config=reference_settings(HERE/'reference-runtime.json',original.sha(original.FULL/'request.json'))
    gt=validated_settings(HERE/'gt-runtime.json');os.environ['ROADMODEL_GT_BACKEND']=gt['backend']
    root=original.c.ROOT;claim=root/'controller.lock'
    with claim.open('x') as f:f.write(str(os.getpid()))
    cursor=0;pending=[];engine=None;reference=None;pool=None
    counts={b:original.c.Counts() for b in ('board','onnx')}
    def progress(state,**extra):
        original.save(root/'progress.json',{'state':state,'task_id':original.REQUEST['task_id'],
            'completed_frames':cursor,'expected_frames':TOTAL,'updated_at':original.c.now(),
            'pid':os.getpid(),'controller_revision':'multiframe-prefetch-GT-v1','batch_size':width,
            'in_flight_inferred_not_committed':[x[0] for x in pending],**extra})
    def commit():
        nonlocal cursor
        index,future=pending[0]
        if index!=cursor:raise ValueError('Checkpoint order differs')
        current=future.result()
        for b in counts:
            counts[b].confusion+=current[b][0];counts[b].ds+=current[b][1];counts[b].num_frames+=1
        cursor+=1;pending.pop(0)
        tmp=root/'checkpoint.multiframe.tmp'
        import numpy as np
        with tmp.open('wb') as f:
            np.savez(f,completed=cursor,**{b+'_confusion':counts[b].confusion for b in counts},
                     **{b+'_ds':counts[b].ds for b in counts})
        os.replace(tmp,root/'checkpoint.npz')
        for b in counts:original.save(root/('metrics_'+b+'.partial.json'),counts[b].summary())
        progress('RUNNING')
    try:
        import numpy as np
        with np.load(root/'checkpoint.npz',allow_pickle=False) as cp:
            cursor=int(cp['completed'])
            if not 0<=cursor<=TOTAL:raise ValueError('Checkpoint outside full scope')
            for b in counts:
                counts[b].confusion=cp[b+'_confusion'].copy();counts[b].ds=cp[b+'_ds'].copy();counts[b].num_frames=cursor
        if cursor==TOTAL:finalize(counts,progress);return
        engine=Engine();indices=list(range(cursor,TOTAL));retained=[];fresh=[]
        for i in indices:
            frame=root/'frames'/f'frame_{i:06d}'
            if (frame/'execution_receipt.json').exists():
                receipt=original.c.load(frame/'execution_receipt.json')
                if receipt['selected_frames']!=[original.FRAMES[i]] or receipt['state']!='succeeded' or receipt['release']['state']!='RELEASED':
                    raise ValueError('Invalid retained native frame')
                retained.append(i)
            elif (frame/'execution.json').exists():raise ValueError('Unknown native attempt; never replay')
            if requires_reference(frame):fresh.append(i)
        exporter=original.make_pipeline(fresh,root/('multiframe-canonical-'+str(int(time.time())))) if fresh else iter(())
        reference=ReferencePrefetch(engine,indices,original.FRAMES,exporter,root/'frames',original.FULL,reference_config)
        pool=ProcessPoolExecutor(max_workers=gt['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_native_gt,initargs=(1,gt['backend']))
        for group in groups(indices,width,retained):
            frames=[reference.get(i) for i in group['indices']]
            if (original.FULL/'STOP_AFTER_FRAME').exists():
                while pending:commit()
                progress('STOPPED_AT_BOUNDARY');return
            if group['mode']=='infer':
                import shutil
                if shutil.disk_usage(root).free<64*1024**3:raise ValueError('Disk reserve below64GiB')
                import subprocess
                sys.path.insert(0,'/mnt/data/bench-eval/service/gateway')
                from bench_gateway.config import load
                from bench_gateway.core import Store
                while True:
                    q=json.loads(subprocess.run([str(original.old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,check=True,timeout=20).stdout)
                    with Store(load('/mnt/data/bench-eval/service/gateway')['state_root']).connect() as db:
                        busy=bool(db.execute("select id from jobs where state in ('QUEUED','RUNNING')").fetchone())
                    if not (busy or q.get('active') or q.get('queued') or q.get('waiting') or (original.old.RUNNER/'locks/bench.lock').exists()):break
                    if (original.FULL/'STOP_AFTER_FRAME').exists():
                        while pending:commit()
                        progress('STOPPED_AT_BOUNDARY');return
                    progress('WAITING_BENCH');time.sleep(2)
                run_id=f"pcm-v322-subnets-full-multi-{group['indices'][0]:06d}-n{len(frames)}"
                engine.run_many(frames,run_id)
            for i,frame in zip(group['indices'],frames):pending.append((i,pool.submit(original.ensure_gt,i,frame)))
            while len(pending)>=gt['workers']*2:commit()
        while pending:commit()
        finalize(counts,progress)
    except Exception as error:
        progress('NEEDS_REVIEW',error_type=type(error).__name__,error=str(error)[:500]);raise
    finally:
        if reference is not None:reference.close()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if engine is not None:engine.close()
        claim.unlink()


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:print(json.dumps(preflight()))
    elif not sys.argv[1:]:main()
    else:raise ValueError('Unsupported arguments')
