import json
from pathlib import Path

root=Path('evidence/actual-runs/pcm-v7-multiscene-001')
for name in ['collected/input_validation.json','collected/output_integrity.json','collected/tensor_summary_receipt.json','postprocess_consistency.json','gt-adapted-20261027/result.json']:
    data=json.loads((root/name).read_text(encoding='utf-8'))
    print(name)
    print(json.dumps({k:(list(v) if isinstance(v,dict) else {'length':len(v),'first':v[:1]} if isinstance(v,list) else v) for k,v in data.items()},ensure_ascii=True)[:6000])
cos=json.loads((root/'collected/cosine_summary.json').read_text())
print('cosine_first',json.dumps(cos['frames'][0],ensure_ascii=True)[:1800])
print('integrity_first',json.dumps(json.loads((root/'collected/output_integrity.json').read_text()),ensure_ascii=True)[:3500])
