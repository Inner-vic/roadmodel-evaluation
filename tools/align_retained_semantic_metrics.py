"""Rescore existing logits with the original training CE/3-offset GT definition."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
diag=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01/trajectory-diagnostic-01')
src=Path('evidence/model-evaluation/remaining-heads-20261027/input/DataReorganized-V151/2026-09-08/2026-07-07_10-09-25_HLX33B166R1997378/dataset')
rows=[]
for i,frame in enumerate([20,21]):
    p=diag/f'frame-{i}-native.npz'
    with np.load(p,allow_pickle=False) as a:
        logits=a['result_surface_label_logits'].astype(np.float64).reshape(-1,27)
        xy=a['result_surface_xy'].reshape(-1,2)
        mask=a['result_surface_valid_mask'].reshape(-1).astype(bool)
        meshfile=next((src/str(frame)).glob('height_field_*.npy'));mesh=np.load(meshfile,allow_pickle=False)
        ix=np.floor((mesh[:,0]-np.float32(5)) / np.float32(.05)).astype(int);iy=np.floor((mesh[:,1]-np.float32(-3.6)) / np.float32(.05)).astype(int)
        good=(mesh[:,4]>.5)&np.isfinite(mesh[:,:4]).all(1)&(mesh[:,3]>=0)&(mesh[:,3]<26)&(ix>=0)&(ix<400)&(iy>=0)&(iy<144)
        grid=np.full((400,144),26,dtype=int);grid[ix[good],iy[good]]=np.rint(mesh[good,3]).astype(int)
        qx=np.floor((xy[:,0]-np.float32(5))/np.float32(.05)).astype(int);qy=np.floor((xy[:,1]-np.float32(-3.6))/np.float32(.05)).astype(int)
        inside=(qx>=0)&(qx<400)&(qy>=0)&(qy<144);gt=np.full(len(qx),26,dtype=int);gt[inside]=grid[qx[inside],qy[inside]]
        keep=mask&inside&(gt!=26);assert keep.sum()==2040 and np.all(gt[keep]==2)
        l=logits[keep];t=gt[keep];mx=l.max(1);ce=np.log(np.exp(l-mx[:,None]).sum(1))+mx-l[np.arange(len(l)),t]
        pred=l.argmax(1);rows.append({'frame_index':frame,'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'all three lateral offsets, original loss valid mask, original CE with cement class weight 1','valid_surface_tokens':int(keep.sum()),'GT_cement_tokens':int((t==2).sum()),'correct':int((pred==t).sum()),'label_accuracy_percent':float((pred==t).mean()*100),'mean_CE':float(ce.mean()),'predictions':dict(collections.Counter(str(x) for x in pred))})
out={'schema':'roadmodel.aligned-semantic-rescore/v1','frames':rows,'method':'stable float64 CE from retained FP32 logits; no model forward; matches original deterministic CE formula for cement-only support','new_inference_frames':0,'new_training_runs':0}
(root/'aligned-semantic-metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
