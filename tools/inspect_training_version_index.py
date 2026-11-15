import json
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
v=json.loads((root/'output-modelset-versions.json').read_text(encoding='utf-8'))
print('metadata',v.get('metadata'),'count',len(v['items']))
for x in v['items']:
    m=x.get('metadata',{});s=x.get('spec',{})
    print(json.dumps({'name':m.get('name'),'uid':m.get('uid'),'created':m.get('creationTimestamp'),'spec_keys':list(s),'task':s.get('task'),'source':s.get('source')},ensure_ascii=False))
    if '8d226ad5' in json.dumps(x):print('MATCH',json.dumps(x,ensure_ascii=False))
