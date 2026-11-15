import pathlib,hashlib,json,shutil,importlib.util
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'source';old=pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')
out={'source_comparison':{},'installed_operators':{},'toolchain':{}}
for prefix in ['model/backbone','model/encoder/gaussian_encoder/ops','model/head/localagg_prob','model/head/localagg_prob_fast','pointops']:
 def h(r):return {str(p.relative_to(r/prefix)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/prefix).rglob('*') if p.suffix in ['.cpp','.cu','.h','.cuh']}
 a,b=h(root),h(old)
 out['source_comparison'][prefix]={'count':len(a),'old_count':len(b),'identical':a==b,'differing':[p for p in set(a)|set(b) if a.get(p)!=b.get(p)]}
for name in ['deformable_aggregation_ext','local_aggregate_prob','local_aggregate_prob_fast','pointops_cuda','pointops','mmseg','mmcv']:
 try:
  spec=importlib.util.find_spec(name);out['installed_operators'][name]=None if spec is None else spec.origin
 except Exception as e:out['installed_operators'][name]=str(e)
for name in ['nvcc','ninja','g++']:out['toolchain'][name]=shutil.which(name)
print(json.dumps(out))
