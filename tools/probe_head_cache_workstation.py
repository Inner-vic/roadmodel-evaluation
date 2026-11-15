import json
from pathlib import Path
from sync_requested_eval import ssh
code=r'''
import pathlib,json
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3')
print(json.dumps({'cache_root':str(root),'versions':[p.name for p in root.iterdir()] if root.exists() else []}))
'''
raw=ssh(['python3','-c',code]);Path('evidence/model-evaluation/remaining-heads-20261027/workstation-cache.json').write_bytes(raw);print(raw.decode())
