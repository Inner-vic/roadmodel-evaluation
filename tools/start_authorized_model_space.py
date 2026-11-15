"""Start the exact devspace explicitly authorized by the user in this chat."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
data=request(['devspaces','start','--namespace','vc-chassis','--cluster','cnhb01-algorithm-bidu-p-03','--name','bmad-r3-a100-20261015'])
Path('evidence/model-evaluation/model-reference-20261027/authorized-start-receipt.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data,ensure_ascii=True)[:5000])
