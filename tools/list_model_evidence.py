from pathlib import Path
import json
root=Path('evidence/model-evaluation/model-reference-20261027')
for p in root.iterdir():
 if p.suffix=='.json' and any(x in p.name for x in ['start','ready','approval']):
  print(p.name,p.read_text(encoding='utf-8')[:4000])
print('FILES',[p.name for p in root.iterdir()])
