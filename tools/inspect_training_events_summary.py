import json,statistics
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
d=json.loads((root/'tensorboard-scalars.json').read_text(encoding='utf-8'));s=d['scalar_series']
for k in ['surface_patch/label_ce','surface_patch/label_acc','Train/SurfacePatchDDPMLoss/surface_label_ce','Validation/Traj_cls_cement','Validation/Traj_cls_count_cement','Validation/Traj_label_acc','Validation/Loss']:
    v=s[k];print(json.dumps({'tag':k,'n':len(v),'first':v[:2],'last':v[-2:]},ensure_ascii=False))
for e in [0,43,59]:
    out={'checkpoint_epoch':e+1}
    for k in ['surface_patch/label_ce','surface_patch/label_acc']:
        v=[x['value'] for x in s[k] if e*523<=x['step']<(e+1)*523]
        out[k]={'n':len(v),'mean':statistics.mean(v) if v else None}
    print(json.dumps(out))
print('validation tags',json.dumps([k for k in s if k.startswith('Validation/')]))
