"""Execute frozen, isolated head runs; never launch training or expand sample scope."""
import pathlib,json,subprocess,os,sys,hashlib,datetime,time
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
parent=base.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest=json.loads((base/'input-manifest.json').read_text())
assert all(sha(base/'input'/x['path'])==x['sha256'] for x in manifest)
source_manifest=json.loads((parent/'source-manifest.json').read_text())
assert all(sha(parent/'source'/p)==h for p,h in source_manifest['files'].items())
state={'state':'running','started_at':now(),'runs':{}}
def save():
 p=base/'workflow-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
save()
for kind in ['vision','trajectory']:
 run_dir=base/(kind+'-01');launch=json.loads((run_dir/'launch.json').read_text())
 assert not (run_dir/'execution_receipt.json').exists(),'Prior attempt must be preserved'
 env=os.environ.copy();env.update(launch['environment'])
 receipt={'schema':'roadmodel.model-execution/v1','source':'lpai_devspace_native_eval','run_id':launch['run_id'],
  'kind':kind,'state':'preflight','expected_frames':launch['expected_frames'],'selected_frames':launch['selected_frames'],
  'checkpoint_sha256':launch['checkpoint_sha256'],'source_commit':launch['source_commit'],
  'dataset_version':launch['dataset_version'],'dataset_class':launch['dataset_class'],
  'data_source':launch['data_source'],'environment':launch['environment'],'argv':launch['argv'],'started_at':now(),
  'launch_sha256':sha(run_dir/'launch.json'),'input_manifest_sha256':sha(base/'input-manifest.json'),
  'data_link_sha256':sha(pathlib.Path(launch['environment']['DATA_LINK_FILE']))}
 def save_receipt():(run_dir/'execution_receipt.json').write_text(json.dumps(receipt,indent=2))
 for path,h in launch['checkpoint_sha256'].items():assert sha(parent/'weights'/path)==h
 save_receipt();state['runs'][kind]={'state':'preflight'};save()
 with (run_dir/'preflight-console.log').open('w') as log:
  ret=subprocess.run([sys.executable,str(base/'head_preflight.py'),str(run_dir)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 if not ret:
  receipt['state']='running';save_receipt();state['runs'][kind]={'state':'running'};save()
  start=time.monotonic()
  with (run_dir/'eval-console.log').open('w') as log:
   ret=subprocess.run(launch['argv'],cwd=launch['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  receipt['duration_seconds']=time.monotonic()-start
 receipt.update(state='succeeded' if ret==0 else 'failed',exit_code=ret,finished_at=now())
 receipt['outputs']=[{'path':str(p.relative_to(run_dir)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in run_dir.rglob('*') if p.is_file() and p.name!='execution_receipt.json']
 save_receipt();state['runs'][kind]={k:receipt[k] for k in ['state','exit_code','run_id','finished_at']};save()
state.update(state='succeeded' if all(x['state']=='succeeded' for x in state['runs'].values()) else 'partial',finished_at=now());save()
