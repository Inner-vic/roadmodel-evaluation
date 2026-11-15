"""Independent compile-GT jobs over retained predictions; no inference APIs.

The numerical body is taken from the frozen original worker, hash checked, and
run in a private output directory. Only the parent commits ordered counts.
"""
from __future__ import annotations
import hashlib,importlib.util,json,os,sys,textwrap,time
from pathlib import Path
from .common import digest,load_json,save_json,utc_now

_CONTEXTS={}
_THREAD_LIMIT=None
_BACKEND="reference"


def initialize_gt_worker(threads=1, backend="reference"):
    global _THREAD_LIMIT, _BACKEND
    if backend not in ("reference", "memory_optimized"):raise ValueError("Invalid GT backend")
    _BACKEND=backend
    if type(threads)!=int or not 1<=threads<=64:raise ValueError('Invalid worker thread count')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]=str(threads)
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT=threadpool_limits(limits=threads)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(threads)
        try:sys.modules["torch"].set_num_interop_threads(1)
        except RuntimeError:pass


def _context(job):
    source=Path(job['legacy_worker']).resolve();frozen=Path(job['frozen_request']).resolve()
    if digest(source)!=job['legacy_worker_sha256'] or digest(frozen)!=job['frozen_request_sha256']:
        raise ValueError('Frozen source/request differs')
    key=(str(source),job['legacy_worker_sha256'],job['frozen_request_sha256'])
    if key not in _CONTEXTS:
        sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location('retained_GT_'+key[1][:12],source)
        c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
        request=load_json(frozen);link=c.DATA/'data_link_26-10-20-1_train859_eval135_v7.json'
        if digest(link)!=request['data_link_sha256']:raise ValueError('GT data-link changed')
        data=load_json(link);raw=source.read_text()
        a=raw.index("            receipt = load(root / 'execution_receipt.json')")
        b=raw.index('            cursor += 1',a);body=textwrap.dedent(raw[a:b])
        if hashlib.sha256(body.encode()).hexdigest()!=job['GT_body_sha256']:raise ValueError('Original GT body changed')
        if _BACKEND == "memory_optimized":
            from .memory_gt import install
            install()
        _CONTEXTS[key]=(c,request,data,compile(body,'frozen-original-GT-body','exec'))
        # Torch's independent metric oracle is used at the same original indices.
        try:
            import torch
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
            torch.set_num_interop_threads(1)
        except (ImportError,RuntimeError):pass
    return _CONTEXTS[key]


def process_retained_gt(job):
    """Reproduce a retained frame in isolation and compare labels/counts exactly."""
    import numpy as np
    started=time.perf_counter();c,request,data,body=_context(job)
    index=job['frame_index'];selected=request['frames'][index]
    source=Path(job['source_root']).resolve()
    expected=Path(job['frozen_request']).resolve().parent/'compile/frames'/f'frame_{index:06d}'
    if source!=expected.resolve():raise ValueError('Prediction source is not the frozen frame slot')
    out=Path(job['output']).resolve();approved=Path(job['output_root']).resolve()
    if out==approved or not out.is_relative_to(approved) or out.is_relative_to(Path(job['frozen_request']).resolve().parent):
        raise ValueError('GT output must be in a private, isolated job root')
    out.mkdir(parents=True,exist_ok=False)
    identity={'frame_index':index,'selected_frame':selected,'frozen_request_sha256':job['frozen_request_sha256'],
              'GT_body_sha256':job['GT_body_sha256'],'GT_backend':_BACKEND,
              'memory_kernel_sha256':digest(Path(__file__).with_name('memory_gt.py')) if _BACKEND=='memory_optimized' else None,'started_at':utc_now()}
    save_json(out/'job-receipt.json',dict(identity,state='RUNNING'))
    try:
        receipt=load_json(source/'execution_receipt.json')
        if receipt['state']!='succeeded' or receipt['selected_frames']!=[selected] or receipt['release']['state']!='RELEASED':
            raise ValueError('Retained native execution is not complete and released')
        for name in ('execution_receipt.json','public-outputs.json','all-subnet-outputs.json'):
            (out/name).write_bytes((source/name).read_bytes())
        (out/'onnx_outputs').symlink_to(source/'onnx_outputs',target_is_directory=True)
        for folder in source.glob('sub[0-4]'):
            (out/folder.name).symlink_to(folder,target_is_directory=True)
        scene,fr=selected;item=next(x for x in data['eval'][scene]['frames'] if x['frame_index']==fr)
        occ=c.DATA/scene/'dataset'/str(fr)/item['occ_file']
        reference_gt=load_json(source/'gt_frame_receipt.json')
        if digest(occ)!=reference_gt['occ_sha256']:raise ValueError('Retained GT identity changed')
        acc={b:c.Counts() for b in ('board','onnx')}
        env={**c.__dict__,'root':out,'scene':scene,'fr':fr,'occ':occ,'cursor':index,'acc':acc,'oracle_checks':0}
        compute_started=time.perf_counter();exec(body,env);compute_seconds=time.perf_counter()-compute_started
        for backend in acc:
            candidate=out/(backend+'_final_occ.npy')
            if digest(candidate)!=reference_gt['predictions'][backend]['sha256']:
                raise ValueError('Voxel labels differ from original '+backend)
            if load_json(out/('gt_metrics_'+backend+'.json'))!=load_json(source/('gt_metrics_'+backend+'.json')):
                raise ValueError('GT metrics differ from original '+backend)
        counts=out/'single-counts.npz'
        with counts.open('xb') as f:np.savez(f,**{b+'_confusion':acc[b].confusion for b in acc},**{b+'_ds':acc[b].ds for b in acc})
        if (source/'single-counts.npz').exists():
            with np.load(source/'single-counts.npz',allow_pickle=False) as original:
                for backend in acc:
                    if not np.array_equal(original[backend+'_confusion'],acc[backend].confusion) or not np.array_equal(original[backend+'_ds'],acc[backend].ds):
                        raise ValueError('Integer counts differ from original '+backend)
        value=dict(identity,state='VERIFIED',counts_path=str(counts),counts_sha256=digest(counts),
                   original_execution_sha256=digest(source/'execution_receipt.json'),GT_seconds=compute_seconds,
                   wall_seconds=time.perf_counter()-started,all_voxel_labels_identical=True,all_metrics_identical=True,
                   integer_counts_identical=True,native_inference_frames=0,source_results_modified=False,finished_at=utc_now())
        save_json(out/'job-receipt.json',value,overwrite=True);return value
    except Exception as error:
        save_json(out/'job-receipt.json',dict(identity,state='NEEDS_REVIEW',error_type=type(error).__name__,error=str(error)[:400]),overwrite=True)
        raise


class OrderedCheckpoint:
    """Sole-writer count commits: no averaging, gaps or duplicate commits."""
    def __init__(self,path,*,cursor=0,initial=None):
        import numpy as np
        self.path=Path(path);self.cursor=cursor
        self.values={b:{'confusion':np.zeros((27,27),np.int64),'ds':np.zeros(3,np.int64)} for b in ('board','onnx')}
        if initial:
            with np.load(initial,allow_pickle=False) as cp:
                if int(cp['completed'])!=cursor:raise ValueError('Initial checkpoint cursor differs')
                for b in self.values:
                    for k in self.values[b]:self.values[b][k]=cp[b+'_'+k].copy()
    def commit(self,index,receipt):
        import numpy as np
        if index!=self.cursor:raise ValueError('Out-of-order or duplicate GT commit')
        if receipt.get('state')!='VERIFIED' or digest(Path(receipt['counts_path']))!=receipt['counts_sha256']:
            raise ValueError('Unverified or changed GT counts')
        candidate={b:{} for b in self.values}
        with np.load(receipt['counts_path'],allow_pickle=False) as cp:
            for b in candidate:
                for k,shape in (('confusion',(27,27)),('ds',(3,))):
                    x=cp[b+'_'+k]
                    if x.shape!=shape or x.dtype!=np.int64 or (x<0).any():raise ValueError('Invalid raw counts')
                    candidate[b][k]=self.values[b][k]+x
        tmp=self.path.with_suffix('.tmp');self.path.parent.mkdir(parents=True,exist_ok=True)
        with tmp.open('wb') as f:np.savez(f,completed=index+1,**{b+'_'+k:x for b,v in candidate.items() for k,x in v.items()})
        os.replace(tmp,self.path);self.values=candidate;self.cursor+=1
