import pathlib,hashlib,json,difflib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'source';old=pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')
out={}
for prefix in ['model/backbone','model/encoder/gaussian_encoder/ops','model/head/localagg_prob','model/head/localagg_prob_fast','pointops']:
 def h(r):return {str(p.relative_to(r/prefix)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in (r/prefix).rglob('*') if p.suffix in ['.cpp','.cu','.h','.cuh']}
 a,b=h(root),h(old);different=[p for p in set(a)|set(b) if a.get(p)!=b.get(p)]
 out[prefix]={'identical_except_line_endings':a==b,'different':different,'diff':[]}
 for name in different[:2]:
  pa,pb=root/prefix/name,old/prefix/name
  if pa.exists() and pb.exists():out[prefix]['diff'].append('\n'.join(difflib.unified_diff(pa.read_text().splitlines(),pb.read_text().splitlines()))[:2800])
print(json.dumps(out))
