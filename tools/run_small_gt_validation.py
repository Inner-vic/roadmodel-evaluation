"""Deploy the GT-only worker and collect metadata; no model/board execution."""
import json
from pathlib import Path
from sync_requested_eval import ssh

base=Path(__file__).resolve().parents[1]
remote='/mnt/data/developer/pcm-roadmodel-eval'
worker_root=remote+'/gt-adapter-v1'
writer='import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())'
for local,dest in [('pcm_eval/__init__.py','pcm_eval/__init__.py'),
                   ('pcm_eval/gt_adapter.py','pcm_eval/gt_adapter.py'),
                   ('tools/evaluate_retained_gt.py','tools/evaluate_retained_gt.py')]:
    ssh(['python3','-c',writer,worker_root+'/'+dest],(base/local).read_bytes())
out=remote+'/runs/pcm-v7-multiscene-001/gt-adapted-20261027'
response=ssh([remote+'/venv/bin/python',worker_root+'/tools/evaluate_retained_gt.py',
              '--run-dir',remote+'/runs/pcm-v7-multiscene-001',
              '--dataset-root',remote+'/datasets/smartchassisdata-v3/26-10-20-1',
              '--output-dir',out],timeout=180)
local=base/'evidence/actual-runs/pcm-v7-multiscene-001/gt-adapted-20261027'
local.mkdir(parents=True,exist_ok=False)
(local/'worker-output.jsonl').write_bytes(response)
for name in ['gt_receipt.json','metrics_board.json','metrics_onnx.json']:
    data=ssh(['cat',out+'/'+name]);(local/name).write_bytes(data)
receipt=json.loads((local/'gt_receipt.json').read_text())
print(json.dumps({'state':receipt['state'],'frames':receipt['num_frames'],
                  'minimum_height_alignment':min(f['geometry_check']['within_one_z_voxel_fraction'] for f in receipt['frames']),
                  'directory':str(local)}))
