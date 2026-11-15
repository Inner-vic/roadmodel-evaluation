"""Reproducible offline composition of existing scoped model results only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.cli import main
from pcm_eval.common import load_json

root=Path('evidence/model-evaluation/remaining-heads-20261027')
dest=Path('evidence/reports/model-heads-v2-20261027')
args=['report-model','--result','evidence/model-evaluation/native-v3.2.2-smoke-20261027/normalized_result.json',
      '--binding-file','evidence/model-evaluation/model-reference-20261027/model-input-binding.json',
      '--gt-preparation-file','evidence/model-evaluation/model-reference-20261027/native-gt-preparation.json',
      '--head-result',str(root/'collected/vision-01/normalized_result-reviewed.json'),
      '--head-result',str(root/'collected/trajectory-01-recovered/normalized_result-reviewed.json'),
      '--trajectory-figure-receipt',str(root/'figures/figure-receipt.json'),'--output',str(dest)]
support=[root/n for n in ['vision-context.json','trajectory-context.json','scope-inventory.json','input-receipt.json','vision-subset.json','trajectory-subset.json']]
source=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
support.extend(source/n for n in ['misc/traj_payload.py','misc/traj_eval_utils.py','misc/param.txt',
    'model/head/traj_head/traj_surface_front_dit_ddpm_head.py','config/3-0-0-temporal-dit.py','config/2-1-0.py'])
support.extend([root/'collected/trajectory_head_adapter.py',root/'collected/head_preflight.py',
    root/'collected/run_remaining_heads.py',Path('tools/recover_retained_trajectory_remote.py'),root/'collected/trajectory-01/eval-console.log'])
support.extend((root/'collected/vision-01/output/3-0-0-temporal-dit/logs').glob('*.py'))
support.extend((root/'collected/vision-01/output/3-0-0-temporal-dit/vis_vision').glob('*.jpg'))
for path in support:
    if not path.is_file():raise FileNotFoundError(path)
    args+=['--supporting-artifact',str(path)]
status=main(args)
if status:raise SystemExit(status)
r=load_json(dest/'report.json');assert len(r['all_metrics'])==426
assert r['head_results']['vision']['coverage']['observed']['model_samples']==4
assert r['head_results']['trajectory']['coverage']['observed']['model_samples']==2
assert r['source_result']==load_json('evidence/model-evaluation/native-v3.2.2-smoke-20261027/normalized_result.json')
assert not any(m.get('dimensions',{}).get('task')=='used' for m in r['all_metrics'])
text=(dest/'report.xml').read_text(encoding='utf-8')
for value in ['53.13','57.62','50.00','0.72','1360','front-only','未覆盖']:
    assert value in text,value
print(json.dumps({'offline_checks':'passed','OCC_preserved':True,'metric_records':len(r['all_metrics']),'new_inference_frames':0}))
