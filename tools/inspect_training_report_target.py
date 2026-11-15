import json,xml.etree.ElementTree as ET
from pathlib import Path
p=Path('evidence/reports/model-training-semantic-audit-20261027/before.json')
d=json.loads(p.read_text(encoding='utf-8-sig'))['data']['document'];t=ET.fromstring('<doc>'+d['content']+'</doc>')
print('revision',d['revision_id'])
for e in t.iter():
    text=''.join(e.itertext())
    if e.tag in ['h2','h3','p'] and (e.tag!='p' or any(k in text for k in ['训练侧','未验证','训练样本','训练分布','训练日志','待核','未采集','仍需'])):
        print(e.tag,e.get('id'),text[:1800])
print('sources',len(list(t.iter('source'))),'images',len(list(t.iter('img'))))
