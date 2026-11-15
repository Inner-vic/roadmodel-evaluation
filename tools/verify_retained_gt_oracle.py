"""Compare real GT results to the untouched supplied semantic counter."""
import json
from pathlib import Path
from sync_requested_eval import ssh

command=r'''
import json,pathlib,sys,numpy as np,hashlib
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval')
sys.path.insert(0,str(root/'package/benchEval-skill-20261020'))
sys.path.insert(0,str(root/'gt-adapter-v1'))
from eval_metrics_bridge import MetricsAccumulator
from pcm_eval.gt_adapter import adapt_sparse,DRIVABLE_IDS
r=root/'runs/pcm-v7-multiscene-001';g=r/'gt-adapted-20261027'
rec=json.load(open(g/'gt_receipt.json'))
acc={b:MetricsAccumulator(use_cuda=False) for b in ['board','onnx']}
ds={b:[0,0,0] for b in acc}
for i,item in enumerate(rec['frames']):
 a=np.load(item['occ_path'])
 gt,mask,drive,_=adapt_sparse(a,[5,-3.6,-2],[.05,.15,.02],[5,-3.6,-1],(400,48,100))
 for b in acc:
  pred=np.load(item['predictions'][b]['path'])
  acc[b].update(pred,gt,mask,compute_pillar=False)
  ds[b][0]+=int((drive==1).sum())
  pd=np.isin(pred,DRIVABLE_IDS).reshape(-1,100);gd=(drive==1).reshape(-1,100)
  ds[b][2]+=int(pd.sum())
  ds[b][1]+=sum(int(gd[n,np.flatnonzero(row)[0]]) for n,row in enumerate(pd) if row.any())
for b in acc:
 expected=json.load(open(g/f'metrics_{b}.json'));legacy=acc[b].summary()
 assert abs(legacy['present_miou']-expected['present_miou'])<1e-10
 assert abs(legacy['occ_iou']-expected['occ_iou'])<1e-10
 assert len(expected['present_class_ious'])==len(legacy['present_class_ious'])
 assert all({k:v for k,v in e.items() if k!='status'}==a for e,a in zip(expected['present_class_ious'],legacy['present_class_ious']))
 assert ds[b]==[expected['drivable_counts'][k] for k in ['seen','correct','positive']]
result={'state':'passed','frames':20,'backends':2,
 'semantic_oracle':'unmodified supplied eval_metrics_bridge.MetricsAccumulator',
 'semantic_oracle_sha256':hashlib.sha256((root/'package/benchEval-skill-20261020/eval_metrics_bridge.py').read_bytes()).hexdigest(),
 'drivable_oracle':'independent column-wise first-true-Z enumeration',
 'all_class_counts_equal':True,'ds_counts_equal':True}
(g/'crosscheck.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
raw=ssh(['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python','-c',command],timeout=120)
out=Path('evidence/actual-runs/pcm-v7-multiscene-001/gt-adapted-20261027/crosscheck.json')
out.write_bytes(raw)
print(raw.decode())
