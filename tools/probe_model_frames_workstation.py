import json
from pathlib import Path
from sync_requested_eval import ssh,REMOTE_ROOT
code=r'''
import json,pathlib,hashlib
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1')
scene=root/'2026-06-30_18-39-31_HLX34B167T1300224'
paths=list(scene.glob('*'))
paths += list((scene/'dataset'/'18').glob('*'))+list((scene/'dataset'/'19').glob('*'))
out=[]
for p in paths:
 out.append({'path':str(p.relative_to(root)),'is_file':p.is_file(),'bytes':p.stat().st_size})
print(json.dumps(out))
'''
raw=ssh(['python3','-c',code])
Path('evidence/model-evaluation/model-reference-20261027/workstation-two-frame-files.json').write_bytes(raw)
print(raw.decode())
