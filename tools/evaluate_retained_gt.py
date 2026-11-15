"""Evaluate an exact existing prediction inventory; never starts inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pcm_eval.gt_adapter import Counts, adapt_sparse, check_height_geometry

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();r=args.run_dir
    receipt=json.loads((r/'execution_receipt.json').read_text())
    selected=receipt['selected_frames']
    if not 1<=len(selected)<=100:
        raise ValueError('Only bounded existing-scene evaluation is supported (1..100 frames)')
    data=json.loads((r/'effective-data-link.json').read_text())['eval']
    geometry={'source_min':[5,-3.6,-2],'source_shape':[800,48,200], 'voxel_size':[.05,.15,.02],
              'target_min':[5,-3.6,-1],'target_shape':[400,48,100],
              'source_definition':'V161 sparse indices; origin cross-checked against independent physical height-field coordinates',
              'metric_source_commit':'524e079723820582de1ebfe88a2053b072c1893d'}
    accum={b:Counts() for b in ('board','onnx')};frames=[]
    # Do not create the result directory until every input passes validation.
    for i,(scene,frame) in enumerate(selected):
        f=next(f for f in data[scene]['frames'] if f['frame_index']==frame)
        gtroot=args.dataset_root/scene/'dataset'/str(frame)
        occ=gtroot/f['occ_file'];height=gtroot/f['height_field_file']
        raw=np.load(occ,allow_pickle=False);hf=np.load(height,allow_pickle=False)
        check=check_height_geometry(raw,hf,geometry['source_min'],geometry['voxel_size'])
        gt,mask,drive,crop=adapt_sparse(raw,geometry['source_min'],geometry['voxel_size'],geometry['target_min'],geometry['target_shape'])
        fd=r/f'frame_{i:06d}';fm=json.loads((fd/'frame_manifest.json').read_text())
        if fm['meta']['scene_name']!=scene or fm['meta']['frame_idx']!=frame:
            raise ValueError('Retained prediction frame inventory differs from selected frames')
        item={'scene':scene,'frame_index':frame,'frame_id':fm['frame_id'], 'occ_path':str(occ),'occ_sha256':sha(occ),
              'height_sha256':sha(height),'geometry_check':check,'adaptation':crop,'predictions':{}}
        for b,folder in [('board','infer_output'),('onnx','onnx_infer_output')]:
            prediction=fd/folder/'final_occ.npy'
            accum[b].update(np.load(prediction,allow_pickle=False),gt,mask,drive)
            item['predictions'][b]={'path':str(prediction),'sha256':sha(prediction)}
        frames.append(item)
        print(json.dumps({'validated':i+1,'total':len(selected),'height_alignment':check['within_one_z_voxel_fraction']}),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result={'schema':'pcm.gt-adaptation/v1','state':'validated','created_at':datetime.now(timezone.utc).isoformat(),
            'source_run_id':receipt['run_id'],'selected_frames':selected,'num_frames':len(frames),'geometry':geometry,
            'frames':frames,'source_execution_receipt_sha256':sha(r/'execution_receipt.json'),
            'source_cosine_summary_sha256':sha(r/'cosine_summary.json'),
            'source_output_integrity_sha256':sha(r/'output_integrity.json'),
            'implementation_sha256':{'worker':sha(Path(__file__)),'gt_adapter':sha(Path(__file__).resolve().parents[1]/'pcm_eval/gt_adapter.py')},
            'inference_reused':True,'new_inference_frames':0}
    for b in accum:
        metrics=accum[b].summary()
        path=args.output_dir/f'metrics_{b}.json';path.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        result[b+'_metrics_sha256']=sha(path)
    (args.output_dir/'gt_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':'validated','num_frames':len(frames),'metrics':{b:accum[b].summary() for b in accum}}),flush=True)

if __name__=='__main__': main()
