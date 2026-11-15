import json
from pathlib import Path
root=Path('evidence/reports/compile-gt-update-20261027')
d=json.loads((root/'before.json').read_text(encoding='utf-8-sig'))
print(json.dumps(d,ensure_ascii=True)[:22000])
report=json.loads(Path('evidence/reports/compile-gt-single-20261027.json').read_text(encoding='utf-8'))
print('METRICS',json.dumps(report['tables'][0],ensure_ascii=True))
