"""Checkpoint-compatible controller; original GT body, bounded CPU/board pipeline."""
import importlib.util,io,json,os,sys,textwrap,time,shutil,hashlib,multiprocessing
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from pcm_eval.gt_runtime import validated_settings,initialize_native_gt
from pcm_eval.reference_prefetch import settings as reference_settings,requires_reference,ReferencePrefetch
import numpy as np
from fast_subnet_driver import FULL,old,sha,save
from batch_subnet_driver import Engine

spec=importlib.util.spec_from_file_location('original_compile_functions',FULL/'code/formal_compile_remote.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
REQUEST=c.load(FULL/'request.json');FRAMES=REQUEST['frames'];TOTAL=len(FRAMES)
assert TOTAL>0 and TOTAL==REQUEST['expected_frames'] and len(set(map(tuple,FRAMES)))==TOTAL
LINK=Path(os.environ.get('ROADMODEL_COMPILE_DATA_LINK',str(c.DATA/'data_link_26-10-20-1_train859_eval135_v7.json')))
SOURCE=c.load(LINK);assert sha(LINK)==REQUEST['data_link_sha256']
original_source=(FULL/'code/formal_compile_remote.py').read_text()
gt_start=original_source.index("            receipt = load(root / 'execution_receipt.json')")
gt_end=original_source.index('            cursor += 1',gt_start)
GT_BODY=textwrap.dedent(original_source[gt_start:gt_end]);GT_BODY_SHA=hashlib.sha256(GT_BODY.encode()).hexdigest()


def occ_for(index):
    scene,fr=FRAMES[index];item=next(f for f in SOURCE['eval'][scene]['frames'] if f['frame_index']==fr)
    return c.DATA/scene/'dataset'/str(fr)/item['occ_file']


def ensure_gt(index,root,validate=False):
    started=time.monotonic();scene,fr=FRAMES[index];occ=occ_for(index)
    receipt=c.load(root/'execution_receipt.json');assert receipt['selected_frames']==[[scene,fr]] and receipt['release']['state']=='RELEASED'
    cp=root/'single-counts.npz';proof=root/'single-counts-receipt.json'
    if cp.exists() or proof.exists():
        assert cp.exists() and proof.exists(),'Incomplete per-frame count receipt'
        r=c.load(proof);assert r['execution_sha256']==sha(root/'execution_receipt.json') and r['GT_sha256']==sha(root/'gt_frame_receipt.json') and r['counts_sha256']==sha(cp)
        with np.load(cp,allow_pickle=False) as data:return {b:(data[b+'_confusion'].copy(),data[b+'_ds'].copy()) for b in ('board','onnx')}
    acc={b:c.Counts() for b in ('board','onnx')}
    if (root/'gt_frame_receipt.json').exists():
        # Counts recovery from retained final predictions; no ONNX/NPU or splat.
        gt,mask,drive,crop=c.adapt_sparse(np.load(occ,allow_pickle=False),[5,-3.6,-2],[.05,.15,.02],[5,-3.6,-1],[400,48,100])
        for b in acc:
            pred=np.load(root/(b+'_final_occ.npy'),allow_pickle=False);acc[b].update(pred,gt,mask,drive)
            assert acc[b].summary()==c.load(root/('gt_metrics_'+b+'.json')),'Existing GT result differs from recovered counts'
        recovery=True
    else:
        namespace={**c.__dict__,'root':root,'scene':scene,'fr':fr,'occ':occ,'cursor':index,'acc':acc,'oracle_checks':0}
        exec(compile(GT_BODY,'original-full-GT-body','exec'),namespace);recovery=False
        if validate:
            # Independent original metric bridge on the very same new-frame
            # predictions, no second inference or Gaussian postprocessing.
            gt,mask=namespace['gt'],namespace['mask']
            checks={}
            for b in acc:
                pred=np.load(root/(b+'_final_occ.npy'),allow_pickle=False)
                oracle=c.MetricsAccumulator(use_cuda=False);oracle.update(pred,gt,mask,compute_pillar=False);summary=oracle.summary()
                own=acc[b].summary()
                for key in ('present_miou','occ_iou'):assert abs(summary[key]-own[key])<1e-10,(b,key,summary[key],own[key])
                checks[b]={'present_miou':own['present_miou'],'occ_iou':own['occ_iou'],'drivable_iou':own['drivable_iou'],'original_metric_oracle_match':True}
            save(root/'GT-consistency-validation.json',{'state':'VERIFIED','original_GT_body_sha256':GT_BODY_SHA,
                 'original_GT_body_unmodified':True,'same6_gaussian_inputs_as_legacy':True,'metrics':checks})
    with cp.open('xb') as f:np.savez(f,**{b+'_confusion':acc[b].confusion for b in acc},**{b+'_ds':acc[b].ds for b in acc})
    save(proof,{'frame_index':index,'frame':FRAMES[index],'execution_sha256':sha(root/'execution_receipt.json'),
          'GT_sha256':sha(root/'gt_frame_receipt.json'),'counts_sha256':sha(cp),'original_GT_body_sha256':GT_BODY_SHA,
          'counts_recovered_from_retained_prediction':recovery,'GT_seconds':time.monotonic()-started,
          'GT_backend':os.environ.get('ROADMODEL_GT_BACKEND','reference'),
          'memory_kernel_sha256':sha(Path(__file__).parent/'pcm_eval/memory_gt.py') if os.environ.get('ROADMODEL_GT_BACKEND')=='memory_optimized' else None})
    return {b:(acc[b].confusion.copy(),acc[b].ds.copy()) for b in acc}


def make_pipeline(indices,where):
    os.environ.update(DATASET_PATH=str(c.DATA),DATA_LINK_FILE=str(LINK),DATASET='SmartChassisDatasetV7')
    pipeline=c.NumpyBenchDataPipeline().setup(save_dir=str(where),prefer_shm=False,max_frames=len(indices),eval_subsample=100,scene_sample=0,load_ground_truth=False)
    assert [[s,f['frame_index']] for s,f in pipeline._frames]==FRAMES
    pipeline._frames=[pipeline._frames[i] for i in indices]
    return iter(pipeline.iter_export())


def validate():
    validation=Path(os.environ['ROADMODEL_OPT_VALIDATION']);indices=[8333,8334]
    assert not validation.exists();validation.mkdir(parents=True)
    live=c.load(c.ROOT/'progress.json');assert live['completed_frames']<min(indices)
    for i in indices:assert not (c.ROOT/'frames'/f'frame_{i:06d}').exists(),'Validation slot already attempted'
    save(validation/'reservation.json',{'indices':indices,'frames':[FRAMES[i] for i in indices],'native_previously_executed':False,'full_PID':live['pid']})
    engine=Engine();exporter=make_pipeline(indices,validation/'canonical');measure=[]
    try:
        for i,man in zip(indices,exporter):
            assert [man.meta['scene_name'],man.meta['frame_idx']]==FRAMES[i]
            root=c.ROOT/'frames'/f'frame_{i:06d}';root.mkdir();start=time.monotonic()
            engine.prepare(SimpleNamespace(cache_frame=str(man.frame_dir),task_root=str(validation)),root)
            engine.run_board(SimpleNamespace(run_id=f'pcm-v322-subnets-full-{i:06d}-optv1'),root,reference_check=True,admission_after_receipt=True)
            gt_start=time.monotonic();ensure_gt(i,root,validate=True);gt_time=time.monotonic()-gt_start
            ex=c.load(root/'execution.json');prep=c.load(root/'prepare.json');native=ex['optimized_board_seconds_before_diagnostic']
            measure.append({'index':i,'state':'VERIFIED','preparation_seconds':prep['preparation_seconds'],
             'board_seconds':native,'one_time_cache_seconds':ex['cache_setup_seconds'],'GT_seconds_with_oracle':gt_time,
             'normal_serial_seconds_excluding_one_time_cache':prep['preparation_seconds']+native-ex['cache_setup_seconds']+gt_time,
             'reference':ex['reference_validation'],'public_bytes':ex['public_tensor_bytes_downloaded'],
             'new_NPU_inferences':1,'new_ONNX_inferences':1,'GT_validation':c.load(root/'GT-consistency-validation.json')})
            save(validation/'progress.json',{'state':'RUNNING','completed_frames':len(measure),'measurements':measure})
        baseline=json.loads((Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/compile-timing-audit/conclusion.json')).read_text())['frame_seconds']
        seconds=sum(x['normal_serial_seconds_excluding_one_time_cache'] for x in measure)/len(measure)
        assert seconds<baseline,'Actual validation did not improve speed; do not hand off'
        save(validation/'verification.json',{'state':'VERIFIED','indices':indices,'native_inferences_per_frame':1,'ONNX_inferences_per_frame':1,
             'measurements':measure,'old_frame_seconds':baseline,'candidate_serial_seconds':seconds,'serial_speedup':baseline/seconds,
             'no_previously_completed_frame_rerun':True,'same_frozen_request_sha256':sha(FULL/'request.json'),
             'same_metric_definitions':True,'all18_tensor_metrics_exact_match':True,'all49_hash_finite_checks_preserved':True,
             'initialization_seconds':engine.initialization_seconds})
        save(validation/'progress.json',{'state':'SUCCEEDED','completed_frames':2,'measurements':measure})
    finally:engine.close()


def finalize(acc,progress):
    progress('FINALIZING')
    comparisons=[c.load(c.ROOT/'frames'/f'frame_{i:06d}/cosine_summary.json')['frames'][0] for i in range(TOTAL)]
    names=[r['onnx_output_name'] for r in comparisons[0]['outputs']]
    tensors={name:{'num_frames':TOTAL,'mean_cosine':sum(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in comparisons)/TOTAL,
                'min_cosine':min(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in comparisons)} for name in names}
    save(c.ROOT/'cosine_summary.json',{'num_frames':TOTAL,'frames':comparisons,'tensors':tensors,
         'mean_cosine_all':sum(f['mean_cosine'] for f in comparisons)/TOTAL,'min_cosine_all':min(f['mean_cosine'] for f in comparisons)})
    for b in acc:save(c.ROOT/('metrics_'+b+'.json'),acc[b].summary())
    progress('SUCCEEDED',version=REQUEST['model_version'])


def main():
    # Refuse before touching the original task's lock or progress on user pause.
    if (FULL/'STOP_AFTER_FRAME').exists():raise ValueError('Full evaluation is paused by user; no native dispatch')
    settings=validated_settings(Path(__file__).with_name('gt-runtime.json'))
    if sha(FULL/'request.json')!=settings['frozen_request_sha256']:raise ValueError('GT settings use a different frozen request')
    reference_config=reference_settings(Path(__file__).with_name('reference-runtime.json'),sha(FULL/'request.json'))
    os.environ['ROADMODEL_GT_BACKEND']=settings['backend']
    claim=c.ROOT/'controller.lock';fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    cursor=0;acc={b:c.Counts() for b in ('board','onnx')};engine=None;gt_pool=None;reference=None;pending=[]
    def progress(state,**extra):save(c.ROOT/'progress.json',{'state':state,'task_id':REQUEST['task_id'],'completed_frames':cursor,
       'expected_frames':TOTAL,'updated_at':c.now(),'pid':os.getpid(),'controller_revision':'batch-board-reference-prefetch-v1','reference_workers':reference_config['workers'],'in_flight_inferred_not_committed':[x[0] for x in pending],**extra})
    try:
        with np.load(c.ROOT/'checkpoint.npz',allow_pickle=False) as cp:
            cursor=int(cp['completed'])
            if not 0<=cursor<=TOTAL:raise ValueError('Checkpoint is outside the frozen scope')
            for b in acc:acc[b].confusion=cp[b+'_confusion'].copy();acc[b].ds=cp[b+'_ds'].copy();acc[b].num_frames=cursor
        if cursor==TOTAL:
            finalize(acc,progress);return
        assert not (FULL/'STOP_AFTER_FRAME').exists();progress('PREPARING')
        engine=Engine();remaining=list(range(cursor,TOTAL))
        fresh=[i for i in remaining if requires_reference(c.ROOT/'frames'/f'frame_{i:06d}')] if reference_config['workers']>1 else remaining
        exporter=make_pipeline(fresh,c.ROOT/('optimized-canonical-'+str(int(time.time()))))
        gt_pool=ProcessPoolExecutor(max_workers=settings['workers'],mp_context=multiprocessing.get_context('spawn'),
                                    initializer=initialize_native_gt,initargs=(1,settings['backend']))
        def prepare_one(i):
            man=next(exporter);assert [man.meta['scene_name'],man.meta['frame_idx']]==FRAMES[i]
            root=c.ROOT/'frames'/f'frame_{i:06d}';root.mkdir(exist_ok=True)
            if not (root/'execution_receipt.json').exists():engine.prepare(SimpleNamespace(cache_frame=str(man.frame_dir),task_root=str(FULL)),root)
            else:assert c.load(root/'execution_receipt.json')['selected_frames']==[FRAMES[i]]
            return root
        def commit_first():
            nonlocal cursor
            i,future=pending[0];assert i==cursor;counts=future.result()
            for b in acc:acc[b].confusion+=counts[b][0];acc[b].ds+=counts[b][1];acc[b].num_frames+=1
            cursor+=1;pending.pop(0);tmp=c.ROOT/'checkpoint.optimized.tmp'
            with tmp.open('wb') as f:np.savez(f,completed=cursor,**{b+'_confusion':acc[b].confusion for b in acc},**{b+'_ds':acc[b].ds for b in acc})
            os.replace(tmp,c.ROOT/'checkpoint.npz')
            for b in acc:save(c.ROOT/('metrics_'+b+'.partial.json'),acc[b].summary())
            progress('RUNNING')
        if reference_config['workers']>1:
            reference=ReferencePrefetch(engine,remaining,FRAMES,exporter,c.ROOT/'frames',FULL,reference_config)
        else:next_prepared=engine.producers.submit(prepare_one,cursor)
        for index in range(cursor,TOTAL):
            root=reference.get(index) if reference else next_prepared.result()
            if (FULL/'STOP_AFTER_FRAME').exists():
                while pending:commit_first()
                progress('STOPPED_AT_BOUNDARY');return
            if shutil.disk_usage(c.ROOT).free<64*1024**3:raise ValueError('Disk reserve below64GiB')
            if not reference and index+1<TOTAL:next_prepared=engine.producers.submit(prepare_one,index+1)
            if not (root/'execution_receipt.json').exists():
                progress('RUNNING',current_frame=FRAMES[index],native_index=index)
                if os.environ.get('ROADMODEL_COMPILE_TASK_ROOT'):
                    namespace=hashlib.sha256(REQUEST['task_id'].encode()).hexdigest()[:10]
                    run_id=f'pcm-v322-subnets-full-{namespace}-{index:06d}'
                else:
                    run_id=f'pcm-v322-subnets-full-{index:06d}-optv1'
                engine.run_board(SimpleNamespace(run_id=run_id),root)
            pending.append((index,gt_pool.submit(ensure_gt,index,root)))
            if len(pending)>=settings['workers']*2:commit_first()
        while pending:commit_first()
        gt_pool.shutdown(wait=True);assert cursor==TOTAL;finalize(acc,progress)
    except Exception as error:progress('NEEDS_REVIEW',error_type=type(error).__name__,error=str(error)[:500]);raise
    finally:
        if reference:reference.close()
        if gt_pool:gt_pool.shutdown(wait=True,cancel_futures=True)
        if engine:engine.close()
        claim.unlink()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='validate':validate()
    else:main()
