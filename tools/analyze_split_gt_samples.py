import collections,hashlib,json,re
from pathlib import Path
import numpy as np
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
receipt=json.loads((root/'gt-sample-receipt.json').read_text(encoding='utf-8'))
ref=Path('evidence/model-evaluation/remaining-heads-20261027/collected/trajectory-01-recovered/frame-0.npz')
if not ref.exists():
    ref=next(Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01/trajectory-diagnostic-01').glob('frame-0-native.npz'))
with np.load(ref,allow_pickle=False) as f:
    print('reference keys',list(f)[:12])
    xy=f['traj_xy_used'].reshape(-1,2);mask=f['traj_valid_mask'].reshape(-1).astype(bool)
def hist(v):
    a,n=np.unique(v,return_counts=True);return {str(int(x)):int(y) for x,y in zip(a,n)}
groups=collections.defaultdict(dict)
for f in receipt['files']:
    assert hashlib.sha256(Path(f['local_path']).read_bytes()).hexdigest()==f['sha256']
    groups[(f['split'],f['scene'],f['frame_index'])][f['kind']]=f
rows=[]
for (split,scene,frame),g in groups.items():
    mesh=np.load(g['height_field_file']['local_path'],allow_pickle=False);occ=np.load(g['occ_file']['local_path'],allow_pickle=False)
    ix=np.floor((mesh[:,0]-np.float32(5))/.05).astype(int);iy=np.floor((mesh[:,1]-np.float32(-3.6))/.05).astype(int)
    valid=np.isfinite(mesh[:,:4]).all(1)&(mesh[:,4]>.5)&(mesh[:,3]>=0)&(mesh[:,3]<26)&(ix>=0)&(ix<400)&(iy>=0)&(iy<144)
    grid=np.full((400,144),26,dtype=int);grid[ix[valid],iy[valid]]=np.rint(mesh[valid,3]).astype(int)
    qx=np.floor((xy[:,0]-np.float32(5))/.05).astype(int);qy=np.floor((xy[:,1]-np.float32(-3.6))/.05).astype(int)
    keep=mask&(qx>=0)&(qx<400)&(qy>=0)&(qy<144)
    labels=grid[qx[keep],qy[keep]];validq=labels!=26
    ox=np.floor((xy[keep,0]-np.float32(5))/.05).astype(int);oy=np.floor((xy[keep,1]-np.float32(-3.6))/.15).astype(int)
    pillars=collections.defaultdict(set)
    for x,y,_,label,*_ in occ.astype(int):pillars[(x,y)].add(label)
    supported=sum(int(label in pillars[(x,y)]) for x,y,label in zip(ox[validq],oy[validq],labels[validq]))
    rows.append({'split':split,'scene':scene,'frame_index':frame,'mesh_shape':list(mesh.shape),'occ_shape':list(occ.shape),'raw_mesh_label_counts':hist(mesh[:,3]),'valid_mesh_label_counts':hist(mesh[valid,3]),'valid_mesh_cement_fraction':float((mesh[valid,3]==2).mean()),'fixed_straight_query':{'scope':'reuse existing failure-frame straight front wheel XY; common diagnostic probes, not original training trajectories','query_count':int(keep.sum()),'valid':int(validq.sum()),'label_counts':hist(labels[validq]),'cement_fraction':float((labels[validq]==2).mean()) if validq.any() else None,'OCC_pillar_contains_mesh_label_count':supported}})
manifest=json.loads((root/'effective-splits.json').read_text(encoding='utf-8'))
for kind,v in manifest.items():
    shared=[x for x in v['scenes_details'] if x['vehicle']=='HLX33B166R1997378'];print(kind,'shared vehicle months',dict(collections.Counter(x['collection_month'] for x in shared)))
result={'schema':'roadmodel.split-gt-samples/v1','selection':'diagnostic matched/coverage controls, not random; proportions cannot estimate entire training or validation distribution','reference_query_file':str(ref),'frames':rows,'new_training_runs':0,'new_inference_frames':0}
(root/'gt-sample-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
