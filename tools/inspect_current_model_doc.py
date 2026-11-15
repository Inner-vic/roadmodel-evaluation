import json,xml.etree.ElementTree as ET
from pathlib import Path
d=json.loads(Path('evidence/reports/model-heads-current-before.json').read_text(encoding='utf-8-sig'))['data']['document']
t=ET.fromstring('<doc>'+d['content']+'</doc>')
print('revision',d['revision_id'])
for i,e in enumerate(t):
    print(i,e.tag,e.get('id'),''.join(e.itertext())[:160] if e.tag in ['h1','title','p'] else '')
print('resources',[(e.tag,e.attrib) for e in t.iter() if e.tag in ['source','img','cite']])
