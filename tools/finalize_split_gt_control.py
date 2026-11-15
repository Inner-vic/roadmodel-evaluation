import hashlib,json
from pathlib import Path
root=Path('evidence/model-evaluation/split-gt-audit-20261027')
a=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
v=json.loads((root/'model-control-verification.json').read_text(encoding='utf-8'));assert v['state']=='verified'
r=json.loads((root/'model-control-01/run-01/receipt.json').read_text());assert r['state']=='succeeded' and r['new_forward_frames']==2
a['model_control']=v;a['model_control']['receipt_path']=str(root/'model-control-01/run-01/receipt.json')
a['file_collection']['new_inference_frames']=2
a['file_collection']['additional_model_input_download_bytes']=json.loads((root/'two-frame-input-receipt.json').read_text(encoding='utf-8'))['new_bytes']
a['compute_probe']={'initial_SSH_probe':'timed_out_while_space_stopped','manual_UI_start':True,'existing_space':'bmad-r3-a100-20261015','model_forward_passes':2,'new_training_runs':0,'results_collected_and_hash_verified':True,'runtime_packages_installed':False}
a['conclusions']['confirmed']+=['Under identical model, validation preprocessing and exactly the frame-20 XY, the training cement frame recalls 657/657 cement points while the retained validation cement frame recalls 0/680.','This supports scene/conditional-feature generalization differences; a global inability to predict cement is excluded on this limited scope.']
a['conclusions']['unresolved']=['Exact training cement support and per-class loss across actual sampled trajectories are absent from retained logs/index/TensorBoard.','The contribution of scene appearance, camera/LiDAR calibration shifts, sampled query distribution and training augmentation is not causally separated.','One training cement frame is insufficient to conclude model-wide overfitting or quantify all training cement performance.']
photos=[('训练水泥帧：2026-04-23，58 帧',root/'model-input/DataReorganized-V151/2026-09-04/2026-04-23_10-16-00_HLX33B166R1997378/dataset/58/58_f_ca_120.jpg'),('验证水泥帧：2026-07-07，20 帧',Path('evidence/model-evaluation/remaining-heads-20261027/input/DataReorganized-V151/2026-09-08/2026-07-07_10-09-25_HLX33B166R1997378/dataset/20/20_f_ca_120.jpg'))]
a['example_images']=[{'caption':caption,'path':str(p).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for caption,p in photos]
(root/'analysis-with-control.json').write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'state':'finalized','new_forward_frames':2,'training_cement_recall':v['frames'][0]['classes']['2']['recall_percent'],'training_cement_CE':v['frames'][0]['class_mean_CE']['2'],'validation_cement_CE':v['retained_validation_frames'][0]['cement_mean_CE']}))
