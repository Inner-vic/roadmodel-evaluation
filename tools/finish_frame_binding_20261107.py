import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-frame-binding-20261107'
read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
candidates=read('candidate-binding.json');lineage=read('frame-lineage.json');validation=read('real-small-validation.json')
missing=[]
for row in candidates['selected']:
    fid=row['frame']['frame_index'];camera=lineage['metadata']['refined_transform_camera.json'][str(fid)]['cameras']['f_ca_120']
    missing.append({'scene':row['scene'],'frame_index':fid,'current_image':row['image']['path'],
      'original_camera_source_path':camera['source_path'],'current_generation_timestamp_us':camera['generation_timestamp_us'],
      'old_label_candidate':row['label']['path'],'needed':'Per-image classification JSON for this exact current camera image, or a verifiable original exporter annotation mapping. No scene labels or class histogram broadcast.'})
proof={'state':'BLOCKED_CURRENT_IMAGE_GT_IDENTITY','evaluation_dataset_version':'26-10-20-1','dataset_replaced':False,
 'training_tasks_verified':read('training-GT-contract.json'),'historical_per_frame_annotations_found':True,
 'original_label_task_fields':23,'same_scene_old_evaluation_overlap':36,'same_scene_frame_pairs_checked_in_metadata':400,'identical_image_object_metadata_matches':0,
 'actual_small_validation':validation,'missing':missing,'adapter_implementation':{'source':'pcm_eval/vision_gt_binding.py','native_integration':'pcm_eval/lpai_native_model.py',
 'tests':read('candidate-tests-v3.json'),'state':'IMPLEMENTED_ISOLATED_TESTED_NOT_ACTIVATED_IN_PRODUCTION'},
 'source_export_code_access':read('original-export-commit.json'),'new_model_inference_frames':0,'new_compile_inference_frames':0,'report_scores_modified':False,
 'scope_limit':'The two real candidates are unverified for this submitted image version. This does not assert that every possible image in the dataset lacks labels.'}
(OUT/'completion.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
idle_code='''import pathlib,os,json,subprocess
visible=set();unknown=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 visible.add(int(p.name))
 if int(p.name)==os.getpid():continue
 try:
  names=[pathlib.PurePosixPath(x.decode(errors='ignore')).name for x in (p/'cmdline').read_bytes().split(bytes([0]))[:3]]
  if names and names[0].startswith('python') and not any(n in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server','supervisord','update_pubkey.py') for n in names[1:]):unknown.append(int(p.name))
 except OSError:pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
busy=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) in visible] if gpu.returncode==0 else [-1]
print(json.dumps({'idle':not unknown and not busy,'unknown_python':unknown,'gpu_processes':busy}))
'''
idle=json.loads(run(['python3','-c',idle_code],timeout=30));(OUT/'idle-proof.json').write_text(json.dumps(idle,indent=2))
print(json.dumps({'state':proof['state'],'adapter_tests':proof['adapter_implementation']['tests']['tests'],'real_checked_frames':2,'missing':missing,'idle':idle,'new_inference_frames':0}))
