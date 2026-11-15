import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
data=request(['audit','query','--kind','devspace','--namespace','vc-chassis','--name','bmad-r3-a100-20261015','--begin','2026-10-27T00:00:00Z','--page-size','20'])
Path('evidence/model-evaluation/model-reference-20261027/space-audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data,ensure_ascii=True)[:9000])
