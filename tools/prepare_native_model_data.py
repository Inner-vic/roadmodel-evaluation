import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pcm_eval.model_dataset import native_sparse_gt

root=Path('evidence/model-evaluation/model-reference-20261027')
remote='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
binding=json.loads((root/'model-input-binding.json').read_text())
link=json.loads((root/'two-frame-data-link.json').read_text())
receipt={'schema':'roadmodel.native-model-gt/v1','contract':binding['gt_contract'],'frames':[],'state':'validated'}
for scene,info in link['eval'].items():
 for frame in info['frames']:
  path=root/'two-frame-input'/scene/'dataset'/str(frame['frame_index'])/frame['occ_file']
  arr=np.load(path,allow_pickle=False)
  out,row=native_sparse_gt(arr,binding['gt_contract'])
  relative=f'adapted-occ/{scene}/{frame["frame_index"]}.npy'
  target=root/'two-frame-input'/relative
  target.parent.mkdir(parents=True,exist_ok=True)
  np.save(target,out,allow_pickle=False)
  row.update(scene=scene,frame_index=frame['frame_index'],source=path.as_posix(),
             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             adapted_path=relative,adapted_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
  receipt['frames'].append(row)
  frame['occ_file']=remote+'/input/'+relative
  frame['height_field_file']=None
link['joint_eval']=link['eval']
(root/'two-frame-input'/'native-two-frame-data-link.json').write_text(json.dumps(link,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'native-gt-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
