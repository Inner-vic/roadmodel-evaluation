import json
from pathlib import Path
p=Path('evidence/model-evaluation/native-v3.2.2-smoke-20261027/result.json')
d=json.loads(p.read_text(encoding='utf-8'))
print('sections',sorted({m.get('dimensions',{}).get('scope','') for m in d['metrics']}))
print('coverage',d['coverage'])
print('collection',d['collection'])
for m in d['metrics']:
 if m['name']=='occ.seen' and m['value']>0:print(m)
print('model_doc')
p=Path('evidence/model-evaluation/model-reference-20261027/model-report-before.json')
if p.exists() and p.stat().st_size:
 d=json.loads(p.read_text(encoding='utf-8-sig'));print(json.dumps(d,ensure_ascii=True)[:8000])
