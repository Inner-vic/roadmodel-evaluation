from pathlib import Path
import json,numpy as np
root=Path('evidence/model-evaluation/remaining-heads-20261027/input')
for p in root.rglob('img_label*.json'):
 print('VISION_LABEL',p.name,json.dumps(json.loads(p.read_text(encoding='utf-8')),ensure_ascii=True))
for p in root.rglob('*.npy'):
 if not p.name.startswith(('occ_','height_')):continue
 a=np.load(p,allow_pickle=False)
 out={'file':p.name,'shape':list(a.shape),'dtype':str(a.dtype),'min':np.min(a,axis=0).tolist(),'max':np.max(a,axis=0).tolist()}
 if a.ndim==2 and a.shape[1]>=4:
  vals,counts=np.unique(a[:,3],return_counts=True);out['label_counts']=dict(zip(vals.astype(int).tolist(),counts.tolist()))
 print('TRAJ_GT',json.dumps(out))
