"""Compile fixed source operators in the isolated run; no package installation."""
import pathlib,subprocess,sys,os,json,time,hashlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
root=base/'source';state={'state':'running','started_at':time.time(),'builds':[]}
def save():
 p=base/'operator-build-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
save()
env=os.environ.copy();env.update(MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.0',CUDA_VISIBLE_DEVICES='1')
for rel in ['model/encoder/gaussian_encoder/ops','model/head/localagg_prob']:
 p=root/rel;argv=[sys.executable,'setup.py','build_ext','--inplace']
 entry={'source':rel,'argv':argv,'started_at':time.time()};state['builds'].append(entry);save()
 with (base/(rel.split('/')[-1]+'-build.log')).open('w') as log:
  ret=subprocess.run(argv,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 entry.update(exit_code=ret,finished_at=time.time(),binaries=[{'path':str(so.relative_to(root)),'sha256':hashlib.sha256(so.read_bytes()).hexdigest()} for so in p.rglob('*.so')])
 save()
 if ret:
  state['state']='failed';save();raise SystemExit(ret)
state.update(state='succeeded',finished_at=time.time());save()
