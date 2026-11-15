import json,re
from pathlib import Path
base=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
for f in (base/'model/head/traj_head').glob('*.py'):
    lines=f.read_text(encoding='utf-8').splitlines()
    matches=[(i+1,l) for i,l in enumerate(lines) if 'surface_loss_valid' in l or 'surface_supervision_valid =' in l or 'def _prepare_common' in l]
    if matches:print(f.name,json.dumps(matches))
probe=json.loads(Path('evidence/model-evaluation/remaining-heads-20261027/datalink-range-probe.json').read_text(encoding='utf-8'))
for r in probe:print('datalink',r['path'],r['bytes'],r['head'][:380])
lines=(root/'algorithm.log').read_text(encoding='utf-8').splitlines()
print('short setup logs')
for l in lines:
    if len(l)<1200 and any(x in l for x in ['[TrajLoad]','[Traj-Load]','[Freeze]','[loaders]','stats','Loaded','Data','TrainDone']):print(l)
