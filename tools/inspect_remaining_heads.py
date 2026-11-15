import json,re
from pathlib import Path
base=Path('evidence/model-evaluation/model-reference-20261027')
for name in ['vision-training-context.json','trajectory-training-context.json']:
 d=json.loads((base/name).read_text(encoding='utf-8'))
 print('\nCONTEXT',name)
 print(json.dumps({k:d.get(k) for k in ['task_uid','task','git','dataset','image_reference','command_entry_points','command_flags']},ensure_ascii=True))
 print('ACTIVE',json.dumps(d['active_launch_settings'],ensure_ascii=True)[:23000])
root=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
print('EVAL_SCRIPT_CANDIDATES',[p.relative_to(root).as_posix() for p in root.rglob('*.py') if any(s in p.name.lower() for s in ['eval','metric','surface','traj','scheme','vision'])])
for name,pattern in [('config/3-0-0-temporal-dit.py',r'MESH|mesh|param_file|DENSE_SURFACE_GT_SOURCE'),('dataset/transform_3d.py',r'Mesh|mesh|height_field|LoadOcc'),('eval.py',r'traj|scheme|surface|img_class'),('dataset/dataset.py',r'mesh|f30|img_label_file')]:
 print('\nSOURCE',name)
 for i,line in enumerate((root/name).read_text(encoding='utf-8').splitlines(),1):
  if re.search(pattern,line,re.I):print(f'{i}: {line}')
