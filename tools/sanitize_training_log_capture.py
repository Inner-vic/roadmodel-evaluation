"""Discard platform configuration payloads; retain original algorithm log lines."""
import json,re
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
for name in ['semantic','end']:
    p=root/(name+'-response.json');v=json.loads(p.read_text(encoding='utf-8'))
    rows=[row for s in v['result'] for row in s.get('values',[])]
    kept=[row for row in rows if re.match(r'\d\d/\d\d \d\d:\d\d:\d\d - mmengine - ',row[1])]
    if not v.get('platform_initialization_excluded'):
        v={'resultType':'streams','result':[{'values':kept}],'platform_initialization_excluded':True,'returned_before_filter':len(rows)}
        p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/(name+'.log')).write_text('\n'.join(row[1] for row in sorted(kept,key=lambda r:int(r[0])))+'\n',encoding='utf-8')
    print(name,len(kept))
