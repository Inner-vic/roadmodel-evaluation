"""One bounded native model evaluation, with an inspectable execution receipt."""
import os,sys,pathlib,json,subprocess,time,hashlib,datetime
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
launch=json.loads((base/'native-launch.json').read_text())
assert json.loads((base/'model-preflight.json').read_text())['state']=='passed'
assert json.loads((base/'operator-build-state.json').read_text())['state']=='succeeded'
assert not (base/'run-01').exists(),'Preserve existing evaluation outputs'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((base/'source-manifest.json').read_text())
assert all(digest(base/'source'/p)==h for p,h in manifest['files'].items()),'Fixed source changed'
weights={name:digest(base/'weights'/name) for name in ['epoch_44.pth','epoch_98.pth']}
assert weights=={'epoch_44.pth':'21923351c1fa301b431a9b5b5f8d3c87d632a3d121bbd158789461cc0015cdfd',
                 'epoch_98.pth':'05cdfe80814b68acc849ddc7a3aec3944ea759f917a2932d5a721c979dbaa110'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt={'schema':'roadmodel.model-execution/v1','source':'lpai_devspace_native_eval','run_id':'model-v3.2.2-smoke-20261027-run-01',
 'state':'running','started_at':now(),'source_commit':manifest['commit'],'checkpoint_sha256':weights,
 'launch_sha256':digest(base/'native-launch.json'),'data_link_sha256':digest(base/'input/native-two-frame-data-link.json'),
 'expected_frames':2,'selected_frames':launch['selected_frames'],'argv':launch['argv'],
 'environment':launch['environment'],'enabled_metrics':launch['enabled_metrics'],'disabled_metrics':launch['disabled_metrics']}
def save():
 p=base/'execution_receipt.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(p)
save()
env=os.environ.copy();env.update(launch['environment'])
start=time.monotonic()
with (base/'eval-console.log').open('w') as log:
 ret=subprocess.run(launch['argv'],cwd=base/'source',env=env,stdout=log,stderr=subprocess.STDOUT).returncode
receipt.update(exit_code=ret,state='succeeded' if ret==0 else 'failed',finished_at=now(),duration_seconds=time.monotonic()-start,
 outputs=[{'path':str(p.relative_to(base)),'bytes':p.stat().st_size} for p in (base/'run-01').rglob('*') if p.is_file()])
save()
