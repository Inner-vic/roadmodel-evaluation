import os,sys,pathlib,json,traceback,hashlib,shutil
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'source';old=pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')
os.chdir(root);sys.path.insert(0,str(root))
result={'source':str(root),'operator_reuse':[]}
# Reuse binaries only when all of their C++ / CUDA / header sources are identical.
newops=root/'model/backbone/ops';oldops=old/'model/backbone/ops'
def source_hashes(p):
 return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in p.rglob('*') if f.suffix in ['.cpp','.cu','.h','.cuh']}
newh,oldh=source_hashes(newops),source_hashes(oldops)
if newh and newh==oldh:
 for so in oldops.glob('*.so'):
  target=newops/so.name
  if not target.exists():shutil.copy2(so,target)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(so.read_bytes()).hexdigest()
  result['operator_reuse'].append({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'sources_identical':True})
try:
 import model
 result['import_model']='passed'
except Exception:
 result['import_model']='failed';result['error']=traceback.format_exc()
(base/'import-preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
