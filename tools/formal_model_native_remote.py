"""Execute one native full-evaluation job/batch with strict frozen input checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

rd = Path(sys.argv[1])
launch = json.loads((rd / 'launch.json').read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
assert not (rd / 'execution_receipt.json').exists(), 'Never silently replay a launched batch'
receipt = {**launch, 'schema': 'roadmodel.model-execution/v1', 'source': 'lpai_devspace_native_eval', 'state': 'preflight', 'started_at': now()}
def save(): (rd / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2))
save()
try:
    env = os.environ.copy(); env.update(launch['environment'])
    for name, expected in launch['checkpoint_sha256'].items():
        assert sha(Path(launch['source_root']).parent / 'weights' / name) == expected
    if launch['kind'] == 'occ':
        binding = json.loads((rd / 'input/data-binding.json').read_text())
        assert binding['selected_frames'] == launch['selected_frames']
        assert all(sha(rd / 'input' / r['path']) == r['sha256'] for r in binding['files'])
    pre = r'''import os,sys,json,pathlib,torch
r=pathlib.Path(sys.argv[1]);a=json.loads((r/'launch.json').read_text());os.environ.update(a['environment']);os.environ['eval']='true';os.chdir(a['source_root']);sys.path.insert(0,a['source_root'])
import model
from mmengine import Config
from mmseg.models import build_segmentor
from dataset import get_dataloader
from misc.ckpt_load_utils import overlay_export_heads
cfg=Config.fromfile(a['py_config'],lazy_import=False);net=build_segmentor(cfg.model);sd=torch.load(a['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict'];net.load_state_dict(sd,strict=True);overlay_export_heads(net,classifier_resume=a['classifier_checkpoint'])
_,dl=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
assert len(dl.dataset)==a['expected_frames'],(len(dl.dataset),a['expected_frames'])
(r/'model-preflight.json').write_text(json.dumps({'state':'passed','dataset_size':len(dl.dataset),'strict_main_keys':len(sd),'classifier_overlay_keys':sum(k.startswith('classifier.') for k in net.state_dict()),'pc_range':list(cfg.pc_range),'torch':torch.__version__,'cuda':torch.version.cuda}))
'''
    with (rd / 'preflight-console.log').open('w') as log:
        p = subprocess.run([sys.executable, '-c', pre, str(rd)], env=env, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError('Native preflight failed')
    receipt['state'] = 'running'; save(); start = time.monotonic()
    with (rd / 'eval-console.log').open('w') as log:
        p = subprocess.run(launch['argv'], cwd=launch['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(state='succeeded' if not p.returncode else 'failed', exit_code=p.returncode, duration_seconds=time.monotonic() - start, finished_at=now())
except Exception as e:
    receipt.update(state='failed', error_type=type(e).__name__, error=str(e)[:300], finished_at=now())
save()
print(json.dumps({'state': receipt['state'], 'expected_frames': launch['expected_frames']}), flush=True)
