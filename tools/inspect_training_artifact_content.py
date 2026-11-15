import json,re
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
log=(root/'original-output/logs/20260907_051943.log').read_text(encoding='utf-8').splitlines()
patterns=['dataset length','dataset size','samples','class distribution','histogram','cement','traj checkpoint','trajload','train_len','val_len']
short=[{'line':i+1,'text':l} for i,l in enumerate(log) if len(l)<1200 and any(p in l.lower() for p in patterns)]
(root/'original-log-index.json').write_text(json.dumps(short,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('original log',len(log),'index',len(short),'non-validation lines',[x for x in short if 'cement' not in x['text']][:30])
files=json.loads((root/'original-output-inventory.json').read_text(encoding='utf-8'))['files']
orig=next(f for f in files if f['path']=='checkpoint/epoch_44.pth')
rel=json.loads(Path('evidence/model-evaluation/model-reference-20261027/files-26-09-08-2.json').read_text(encoding='utf-8'))['files']
release=next(f for f in rel if f['path'].endswith('epoch_44.pth'))
binding={'original':orig,'release':release,'size_equal':orig['bytes']==release['bytes'],'etag_equal':orig['etag']==release['etag']}
(root/'checkpoint-object-binding.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8');print('object binding',json.dumps(binding))
