import json,hashlib
from pathlib import Path

root=Path('evidence/model-evaluation/remaining-heads-20261027')
ref=Path('evidence/model-evaluation/model-reference-20261027')
prior=json.loads((ref/'native-launch.json').read_text())
scope=json.loads((root/'scope-inventory.json').read_text())
parent='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
base=parent+'/heads-20261027'
for kind,expected in [('vision',4),('trajectory',2)]:
 env=dict(prior['environment'])
 env.update(DATA_LINK_FILE=base+'/input/'+kind+'-subset.json',BATCH_SIZE='1',NUM_FRAMES='1',FRAME_STRIDE='1',
            NUM_WORKERS='0',PERSISTENT_WORKERS='False',PIN_MEMORY='False',CKPT_SKIP_KEYS='')
 if kind=='vision':
  env.update(DATASET_PATH=base+'/input/DataReorganized-VisionHead,'+base+'/input/DataReorganized-VisionHead-standard',
             DATASET='SmartChassisDatasetV6',TRAIN_MODE='vision',HEAD_MODE='gs_only',DENSE_SURFACE_GT_SOURCE='occ')
 else:
  env.update(DATASET_PATH=base+'/input/DataReorganized-V151',DATASET='SmartChassisDatasetV7',
             TRAIN_MODE='occ',HEAD_MODE='traj_only',DENSE_SURFACE_GT_SOURCE='mesh',DENSE_SURFACE_MESH_MISSING_POLICY='error')
 run_id='model-v3.2.2-'+kind+'-smoke-20261027'
 launch={'schema':'roadmodel.head-launch/v1','kind':kind,'run_id':run_id,'version_name':'v3.2.2',
  'source_root':parent+'/source','source_commit':'9e16e29351d17a726ce6beca6f821de79b7ade40',
  'main_checkpoint':parent+'/weights/epoch_44.pth','classifier_checkpoint':parent+'/weights/epoch_98.pth',
  'checkpoint_sha256':{'epoch_44.pth':'21923351c1fa301b431a9b5b5f8d3c87d632a3d121bbd158789461cc0015cdfd',
                      'epoch_98.pth':'05cdfe80814b68acc849ddc7a3aec3944ea759f917a2932d5a721c979dbaa110'},
  'py_config':'config/3-0-0-temporal-dit.py','environment':env,'expected_frames':expected,'selected_frames':scope['selected'][kind],
  'data_source':scope[kind+'_source'],'dataset_version':'26-08-14-0','dataset_class':env['DATASET'],
  'task_uid':'8fc0b602-344c-4f73-8fe7-f7b6f77b60aa' if kind=='vision' else '8d226ad5-e769-411b-9c72-57f9dd822c0e'}
 if kind=='vision':launch['argv']=['python3','eval.py','--py-config',launch['py_config'],'--work-dir',base+'/vision-01/output',
  '--resume-from',launch['main_checkpoint'],'--classifier-resume',launch['classifier_checkpoint'],'--vis-occ','--sample-freq','1','--zip-vis','none']
 else:launch['argv']=['python3',base+'/trajectory_head_adapter.py',base+'/trajectory-01']
 (root/(kind+'-launch.json')).write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'launches':['vision','trajectory'],'frozen_samples':scope['selected'],'full_dataset':False}))
