"""Exactly four native vision eval frames; unchanged model, no training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/coverage-20261029')
parent = BASE.parent
rd = BASE / 'vision-01'
launch = json.loads((rd / 'launch.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert launch['expected_frames'] == 4 and len(launch['selected_frames']) == 4
manifest = json.loads((BASE / 'input-manifest.json').read_text())
assert all(sha(BASE / 'input' / r['path']) == r['sha256'] for r in manifest)
source = json.loads((parent / 'source-manifest.json').read_text())
assert all(sha(parent / 'source' / p) == h for p, h in source['files'].items())
assert all(sha(parent / 'weights' / p) == h for p, h in launch['checkpoint_sha256'].items())
env = os.environ.copy()
env.update(launch['environment'])
receipt = {'schema': 'roadmodel.model-execution/v1', 'source': 'lpai_devspace_native_eval',
           **{k: launch[k] for k in ('run_id', 'kind', 'expected_frames', 'selected_frames', 'checkpoint_sha256', 'source_commit', 'dataset_version', 'dataset_class', 'data_source', 'environment', 'argv')},
           'state': 'preflight', 'started_at': now(), 'launch_sha256': sha(rd / 'launch.json'),
           'input_manifest_sha256': sha(BASE / 'input-manifest.json'), 'data_link_sha256': sha(Path(env['DATA_LINK_FILE']))}
def save(): (rd / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2))
assert not (rd / 'execution_receipt.json').exists()
save()
with (rd / 'preflight-console.log').open('w') as log:
    code = subprocess.run([sys.executable, str(BASE / 'head_preflight.py'), str(rd)], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
if not code:
    pre = json.loads((rd / 'model-preflight.json').read_text())
    assert pre['dataset_size'] == 4 and [[r['scene'], r['frame_index']] for r in pre['samples']] == launch['selected_frames']
    receipt['state'] = 'running'
    save()
    started = time.monotonic()
    with (rd / 'eval-console.log').open('w') as log:
        code = subprocess.run(launch['argv'], cwd=launch['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    receipt['duration_seconds'] = time.monotonic() - started
receipt.update(state='succeeded' if code == 0 else 'failed', exit_code=code, finished_at=now())
receipt['outputs'] = [{'path': str(p.relative_to(rd)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in rd.rglob('*') if p.is_file() and p.name != 'execution_receipt.json']
save()
print(json.dumps({'state': receipt['state'], 'frames': 4}), flush=True)
