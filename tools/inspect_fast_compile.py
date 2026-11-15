"""Compact, read-only validation and continuation status."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,os
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');v=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-fast-20261030-r2');d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-fast-20261030-r2')
out={'full':json.loads((full/'compile/progress.json').read_text()),'frames':[]}
for i in (8333,8334):
 root=full/'compile/frames'/f'frame_{i:06d}';row={'index':i}
 for name in ('prepare.json','execution.json','GT-consistency-validation.json','single-counts-receipt.json'):
  p=root/name
  if p.exists():
   x=json.loads(p.read_text())
   if name=='execution.json':x={k:x[k] for k in ('state','cache_setup_seconds','optimized_board_seconds_before_diagnostic','public_tensor_bytes_downloaded','reference_validation','error','lock_retained') if k in x}
   row[name]=x
 out['frames'].append(row)
for name in ('verification.json','ORT-session-reuse-preflight.json','progress.json'):
 p=v/name
 if p.exists():out[name]=json.loads(p.read_text())
out['validation_log_tail']=(d/'validation.log').read_text(errors='replace')[-2400:]
if (d/'handoff.json').exists():out['handoff']=json.loads((d/'handoff.json').read_text());out['full_log_tail']=(d/'full-controller.log').read_text(errors='replace')[-2000:]
print(json.dumps(out,ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE))
