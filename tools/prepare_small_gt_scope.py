import hashlib
import json
from pathlib import Path

root=Path('evidence/requested-dataset')
inventory=json.loads((root/'eval135-file-inventory.json').read_text())
receipt=json.loads(Path('evidence/actual-runs/pcm-v7-multiscene-001/collected/execution_receipt.json').read_text())
selected={(s,str(f)) for s,f in receipt['selected_frames']}
files=[r for r in inventory['files'] if len(r['path'].split('/'))>=4
       and (r['path'].split('/')[0],r['path'].split('/')[2]) in selected
       and Path(r['path']).name.startswith(('occ_frame_','height_field_frame_'))]
assert len(files)==40,len(files)
policy=json.loads((root/'compile-with-occ-required-files.json').read_text())
policy['files']=files
policy['scope']={'source_run':'pcm-v7-multiscene-001','selected_frames':receipt['selected_frames']}
(root/'small-scene-gt-required-files.json').write_text(json.dumps(policy,indent=2)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(r['size'] for r in files),'source_inventory_sha256':hashlib.sha256((root/'eval135-file-inventory.json').read_bytes()).hexdigest(),'declared':policy.get('source_inventory_sha256')}))
