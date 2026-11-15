import json
from pathlib import Path
import numpy as np
r=Path('evidence/model-evaluation/remaining-heads-20261027/collected')
for head in ['vision-01','trajectory-01-recovered']:
 d=json.loads((r/head/'normalized_result.json').read_text())
 print(head,json.dumps({'provenance_keys':list(d['provenance']),'native_run':d['provenance']['native_run'],'artifacts':d['artifacts']},ensure_ascii=True))
 print('METRICS',json.dumps([{k:v for k,v in m.items() if k!='source'} for m in d['metrics']],ensure_ascii=True))
for name in ['frame-0.npz','frame-1.npz']:
 with np.load(r/'trajectory-01-recovered'/name) as a:
  print(name,{k:a[k].shape for k in a.files})
  for k in ['traj_xy_used','traj_valid_mask']:
   print(k,a[k].reshape(-1,2)[:4].tolist() if k=='traj_xy_used' else a[k].sum(axis=-1).tolist())
