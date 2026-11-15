from pathlib import Path
import tarfile,re,json
archive=Path('evidence/model-evaluation/model-reference-20261027/source.tar')
root=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert (root/m.name).resolve().is_relative_to(root.resolve())
 t.extractall(root,filter='data')
for name in ['model/__init__.py','model/backbone/__init__.py','model/head/gaussian_occ_head.py','model/encoder/__init__.py','requirements.txt','misc/dist_utils.py']:
 p=root/name
 if not p.exists():continue
 print('\nFILE',name)
 lines=p.read_text(encoding='utf-8').splitlines()
 if name=='misc/dist_utils.py':
  found=False
  for i,line in enumerate(lines,1):
   if line.startswith('def init_distributed_eval'):found=True
   if found:print(f'{i}: {line}')
 else:print('\n'.join(lines[:170]))
print('OP_FILES',[p.relative_to(root).as_posix() for p in root.rglob('*') if p.name in ['setup.py','__init__.py'] and any(x in str(p) for x in ['ops','local_aggregate','pointops'])])
