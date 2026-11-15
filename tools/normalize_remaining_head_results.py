from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.parsers import collect_model
from pcm_eval.common import save_json
root=Path('evidence/model-evaluation/remaining-heads-20261027/collected')
for kind,folder,count in [('vision','vision-01',4),('trajectory','trajectory-01-recovered',2)]:
 path=root/folder
 if kind=='vision':
  logs=list((path/'output').glob('*/*.log'));assert len(logs)==1
  result=collect_model(path,'v3.2.2',expected_frames=count,log_file=logs[0])
 else:result=collect_model(path,'v3.2.2',expected_frames=count)
 save_json(path/'normalized_result-reviewed.json',result)
 print(json.dumps({'kind':kind,'collection':result['collection'],'execution':result['execution'],
  'coverage':result['coverage'],'metrics':len(result['metrics'])},ensure_ascii=True))
