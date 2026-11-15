"""Record finite tensor checks and the observed V161/legacy GT contract mismatch."""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT=pathlib.Path('/home/lixiang/developer/pcm-roadmodel-eval')
run=ROOT/'runs'/sys.argv[1]
frame=run/'frame_000000'
source=ROOT/'datasets/smartchassisdata-v3/26-10-20-1/2026-06-30_18-39-31_HLX34B167T1300224/dataset/18'
occ_path=next(source.glob('occ_frame*.npy'))
height_path=next(source.glob('height_field*.npy'))
occ=np.load(occ_path)
height=np.load(height_path)
detail=json.loads((frame/'cosine_board_onnx.json').read_text())
checks=[]
for entry in detail['outputs']:
    path=pathlib.Path(entry['board_file'])
    if not path.is_absolute():
        path=pathlib.Path(detail['board_dir'])/path
    value=np.fromfile(path,dtype=np.dtype(entry['dtype_board']))
    checks.append({'backend':'board','output':entry['onnx_output_name'],'finite':bool(np.isfinite(value).all()),'elements':int(value.size)})
for path in sorted(pathlib.Path(detail['ort_dir']).glob('*.bin')):
    value=np.fromfile(path,dtype=np.dtype(path.name.rsplit('.',2)[-2]))
    checks.append({'backend':'onnx','output_file':path.name,'finite':bool(np.isfinite(value).all()),'elements':int(value.size)})
expected=detail.get('num_onnx_outputs',18)
backend_counts={b:sum(x['backend']==b for x in checks) for b in ('board','onnx')}
record={
 'schema':'pcm.input-validation/v1',
 'finite_outputs':{'state':'passed' if all(backend_counts[b]==expected for b in backend_counts) and all(x['finite'] for x in checks) else 'failed','backend_counts':backend_counts,'expected_each_backend':expected,'checks':checks},
 'ground_truth':{
  'state':'grid_contract_mismatch',
  'raw_occ_shape':list(occ.shape),'raw_index_min':occ[:,:3].min(0).tolist(),'raw_index_max':occ[:,:3].max(0).tolist(),
  'height_field_xyz_min':height[:,:3].min(0).tolist(),'height_field_xyz_max':height[:,:3].max(0).tolist(),
  'legacy_target_shape':[400,48,100], 'legacy_target_pc_range':[5.,-3.6,-1.,25.,3.6,1.],
  'legacy_target_voxel_size':[.05,.15,.02],
  'source_grid_note':'Paired height-field and OCC evidence indicates an expanded source grid. Raw voxel indices must not be copied into the legacy grid without an explicit validated source geometry.',
  'decision':'Raw GT-based metrics are retained for debugging but invalid for quality conclusions; tensor-to-tensor comparison is independent of GT.',
  'source_occ_sha256':hashlib.sha256(occ_path.read_bytes()).hexdigest(),
  'source_height_field_sha256':hashlib.sha256(height_path.read_bytes()).hexdigest()
 }
}
(run/'input_validation.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'tensor_checks':len(checks),'finite':record['finite_outputs']['state'],'ground_truth':record['ground_truth']['state']}))
