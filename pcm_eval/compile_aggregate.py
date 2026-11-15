"""Collect real five-subnet frame receipts and GT into one atomic report.

The running full controller is never imported or stopped. Full GT comes from an
atomic checkpoint snapshot; a single-frame validation uses its native GT file.
"""
from copy import deepcopy
import io
from pathlib import Path
from .common import Result,load_json,save_json,digest,utc_now
from .compile_report import build_compile_report
from .validation import require_result


def aggregate_compile(frame_dirs,selected_frames,output,*,request,run_id,checkpoint=None):
    root=Path(output).resolve()
    if root.exists():raise ValueError('Preserve previous collection; choose a new directory')
    if not frame_dirs or len(frame_dirs)!=len(selected_frames) or len(set(map(tuple,selected_frames)))!=len(selected_frames):
        raise ValueError('Exact unique native frame inventory is required')
    frames=[];proofs=[];manifests=[];base=None
    for folder,selected in zip(map(Path,frame_dirs),selected_frames):
        receipt=load_json(folder/'execution_receipt.json')
        if receipt['state']!='succeeded' or receipt['selected_frames']!=[selected] or receipt['release']['state']!='RELEASED':
            raise ValueError('Native execution or release proof differs from selected frame')
        if base is None:base=receipt
        keys=('version_name','onnx_sha256','source_onnx_sha256','packup_sha256','subnet_archives','data_link_sha256')
        if any(receipt.get(k)!=base.get(k) for k in keys):raise ValueError('Native compiler artifacts differ between frames')
        finite=load_json(folder/'input_validation.json')['finite_outputs']
        integrity=load_json(folder/'output_integrity.json')
        if finite['state']!='passed' or integrity['state']!='fresh_outputs_checked' or not integrity['finite']:
            raise ValueError('Finite/fresh native output evidence is missing')
        gt=load_json(folder/'gt_frame_receipt.json');expected_id=selected[0]+'_'+str(selected[1])
        if gt['frame_id']!=expected_id or [gt['scene'],gt['frame_index']]!=selected:
            raise ValueError('GT proof belongs to a different frame')
        cosine=load_json(folder/'cosine_summary.json')
        if cosine['num_frames']!=1 or len(cosine['frames'])!=1 or cosine['frames'][0]['frame_id']!=expected_id:
            raise ValueError('Tensor comparison belongs to a different frame')
        frames.append(cosine['frames'][0]);proofs.append(gt)
        manifests.append({'frame':selected,'files':[{'path':str(folder/n),'sha256':digest(folder/n)} for n in
                    ('execution_receipt.json','cosine_summary.json','input_validation.json','output_integrity.json',
                     'gt_frame_receipt.json','gt_metrics_board.json','gt_metrics_onnx.json')]})
    count=len(frames);root.mkdir(parents=True)
    if checkpoint:
        import numpy as np
        from .gt_adapter import Counts
        raw=Path(checkpoint).read_bytes();(root/'checkpoint_snapshot.npz').write_bytes(raw)
        with np.load(io.BytesIO(raw),allow_pickle=False) as cp:
            if int(cp['completed'])!=count:raise ValueError('Atomic checkpoint covers a different scope')
            totals={b:Counts() for b in ('board','onnx')}
            for i,folder in enumerate(map(Path,frame_dirs)):
                receipt_file=folder/'single-counts-receipt.json';counts_file=folder/'single-counts.npz'
                receipt=load_json(receipt_file)
                if (receipt['frame']!=selected_frames[i]
                        or receipt['execution_sha256']!=digest(folder/'execution_receipt.json')
                        or receipt['GT_sha256']!=digest(folder/'gt_frame_receipt.json')
                        or receipt['counts_sha256']!=digest(counts_file)):
                    raise ValueError('Per-frame GT counters belong to different results')
                with np.load(counts_file,allow_pickle=False) as counts:
                    for b in totals:
                        confusion,ds=counts[b+'_confusion'],counts[b+'_ds']
                        if (confusion.shape!=(27,27) or ds.shape!=(3,)
                                or not np.issubdtype(confusion.dtype,np.integer) or not np.issubdtype(ds.dtype,np.integer)
                                or (confusion<0).any() or (ds<0).any()):
                            raise ValueError('Invalid per-frame GT counters')
                        totals[b].confusion+=confusion;totals[b].ds+=ds;totals[b].num_frames+=1
                manifests[i]['files'] += [{'path':str(p),'sha256':digest(p)} for p in (counts_file,receipt_file)]
            backends={}
            for backend in ('board','onnx'):
                if (not np.array_equal(cp[backend+'_confusion'],totals[backend].confusion)
                        or not np.array_equal(cp[backend+'_ds'],totals[backend].ds)):
                    raise ValueError('Atomic checkpoint differs from the exact per-frame GT counter sum')
                c=Counts();c.confusion=cp[backend+'_confusion'].copy();c.ds=cp[backend+'_ds'].copy();c.num_frames=count
                backends[backend]=c.summary()
    elif count==1:
        backends={b:load_json(Path(frame_dirs[0])/('gt_metrics_'+b+'.json')) for b in ('board','onnx')}
    else:raise ValueError('Full count checkpoint required; per-frame IoU must not be averaged')
    names=[r['onnx_output_name'] for r in frames[0]['outputs']]
    tensors={name:{'num_frames':count,'mean_cosine':sum(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in frames)/count,
                   'min_cosine':min(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name']==name) for f in frames)} for name in names}
    cosine={'num_frames':count,'frames':frames,'tensors':tensors,'mean_cosine_all':sum(f['mean_cosine'] for f in frames)/count,
            'min_cosine_all':min(f['mean_cosine'] for f in frames)}
    manifest=save_json(root/'native-source-manifest.json',{'frames':manifests,'new_inference_frames':0})
    comparison=save_json(root/'cosine_summary.json',cosine)
    normalized=Result('compile_alignment',base['version_name'],root,count)
    normalized.artifact(manifest,'verified_native_source_manifest');normalized.artifact(comparison,'tensor_comparisons')
    for backend,data in backends.items():
        p=save_json(root/('metrics_'+backend+'.json'),data);normalized.artifact(p,'native_GT_counts')
        for name in ('present_miou','occ_iou','drivable_iou'):
            normalized.metric(name,data[name],'percent',p,'/'+name,backend=backend)
    for name,values in tensors.items():
        for key in ('mean_cosine','min_cosine'):
            normalized.metric(key,values[key],'ratio',comparison,'/tensors/'+name,output=name)
    live=deepcopy(base);live.update(run_id=run_id,requested_frames=count,selected_frames=selected_frames,gt_metrics_computed=True,
                    finished_at=load_json(Path(frame_dirs[-1])/'execution_receipt.json')['finished_at'])
    adaptation={'state':'validated','source_run_id':run_id,'selected_frames':selected_frames,'num_frames':count,
                'frames':proofs,'created_at':utc_now(),'method':'original full-worker GT adaptation and retained predictions',
                **{b+'_metrics_sha256':digest(root/('metrics_'+b+'.json')) for b in backends}}
    normalized.value['provenance']={'live_run':live,'gt_adaptation':adaptation,
            'input_validation':{'finite_outputs':{'state':'passed','checks':count*36}},
            'output_integrity':{'state':'fresh_outputs_checked','frame_count':count},'source_manifest':str(manifest)}
    normalized.value['execution']={'state':'succeeded','run_id':run_id,'source':'native_subnet_frame_receipts'}
    for key in ('tensor_comparison','board_metrics','onnx_metrics'):normalized.observe(key,count)
    result=normalized.finish();require_result(result);save_json(root/'normalized-result.json',result)
    report=build_compile_report(result,cosine,backend_metrics=backends,request=request)
    if report['result_validity']!='validated':raise ValueError('Aggregated compile report failed validity checks')
    save_json(root/'report.json',report)
    return report
