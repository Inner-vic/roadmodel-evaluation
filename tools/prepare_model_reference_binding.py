"""Freeze the existing release reference and a two-frame validation input scope."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import digest,utc_now,json_digest

root=Path('evidence/model-evaluation/model-reference-20261027')
trajectory=json.loads((root/'trajectory-training-context.json').read_text(encoding='utf-8'))
vision=json.loads((root/'vision-training-context.json').read_text(encoding='utf-8'))
comparison=json.loads((root/'weights/component-comparison.json').read_text())
assert comparison['components']['base']['equal']==comparison['components']['base']['keys']==883
assert not comparison['unmatched_keys']
receipts={}
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 path=root/'weights'/version/filename
 receipt=json.loads(path.with_suffix('.receipt.json').read_text())
 assert digest(path)==receipt['sha256']
 receipts[version]=receipt
source_data=Path('evidence/requested-dataset/data_link_26-10-20-1_train859_eval135_v7.json')
data=json.loads(source_data.read_text(encoding='utf-8'))
scene=next(iter(data['eval']));frames=data['eval'][scene]['frames'][:2]
subset={key:{} for key in data}
subset['eval']={scene:dict(data['eval'][scene],frames=frames)}
subset['joint_eval']=subset['eval']
subset_path=root/'two-frame-data-link.json';subset_path.write_text(json.dumps(subset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
selected=[[scene,f['frame_index']] for f in frames]
config_root=Path('inspection/release-model-source/9e16e29351d17a726ce6beca6f821de79b7ade40')
model_env=trajectory['active_launch_settings']['environment']
# Preserve actual training switches as provenance; execution overrides remain explicit.
binding={'schema':'roadmodel.model-input-binding/v1','checked_at':utc_now(),
 'model_version':'v3.2.2','model_reference_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'reference_read_access':'verified','main_checkpoint':receipts['26-09-08-2'],
 'classifier_checkpoint':receipts['26-09-08-1'],'separate_occ_checkpoint_required':False,
 'loading':{'base':'full state_dict from epoch_44.pth','classifier_overlay':'classifier.* from epoch_98.pth',
            'trajectory':'already present in epoch_44.pth; not enabled in first smoke',
            'native_implementation':'misc.ckpt_load_utils.overlay_export_heads',
            'validation':'exact matching base tensor storage hashes across both checkpoints; native model shape/load check pending GPU'},
 'code':{'repository':'https://gitlab.chehejia.com/smartchassis/roadmodel.git',
         'evaluation_commit':'9e16e29351d17a726ce6beca6f821de79b7ade40',
         'classifier_training_commit':'597278d0f86c6f22548f62fa31b43e1ac16c49f7',
         'config':'config/3-0-0-temporal-dit.py','config_sha256':digest(config_root/'config/3-0-0-temporal-dit.py'),
         'source_manifest':str(config_root/'manifest.json')},
 'training_environment':model_env,
 'first_validation_overrides':{'HEAD_MODE':'gs_only','TRAIN_MODE':'occ','USE_TEMPORAL':'False',
      'SEQUENTIAL_TRAIN':'False','DENSE_SURFACE_GT_SOURCE':'occ','BATCH_SIZE':'1','NUM_WORKERS':'0'},
 'dataset':{'name':'smartchassisdata-v3','version':'26-10-20-1','class_name':'SmartChassisDatasetV7',
            'original_data_link_sha256':digest(source_data),'subset_data_link':str(subset_path),
            'subset_data_link_sha256':digest(subset_path),'selected_frames':selected,'expected_frames':2,
            'note':'Target is current V161 small-scene validation, not a reproduction of historical V151 quality scores'},
 'gt_contract':{'source_min':[5,-3.6,-2],'source_shape':[800,48,200],
                'voxel_size':[.05,.15,.02],'target_min':[5,-3.6,-1],'target_shape':[400,48,100],
                'source':'existing V161 physical-coordinate validation and pinned model config',
                'native_loader_adaptation_required':True,
                'note':'Do not crop original V161 sparse indices directly with the older model loader; preserve model ROI and adapt GT'},
 'environment':{'name':'bmad-r3-a100-20261015','namespace':'vc-chassis','cluster':'cnhb01-algorithm-bidu-p-03',
                'user_start_authorization':'explicit async reply: allowed startup for small-scene validation',
                'platform_start_approval_id':'12b196ad-f926-4e32-a56d-a1685fc32e42',
                'platform_start_state':'pending_approval','observed_state':'stopped',
                'runtime_torch_custom_ops_preflight':'not_yet_possible','dataset_access_preflight':'not_yet_possible'},
 'state':'inputs_resolved_execution_preflight_pending','new_model_inference_frames':0,
 'user_missing_weight_or_config_information':[],
 'remaining_actions':['Confirm LPAI startup card in Feishu','Connect and inspect GPU/runtime and native data access',
                      'Prepare the frozen two-frame scope and GT geometry adaptation in an independent run directory',
                      'Validate checkpoint keys/shapes and run native eval on the two frames',
                      'Collect results and render an independent model-accuracy single report']}
(root/'model-input-binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
prior=Path('evidence/environment/model-input-review-20261027/provided-model-reference.json')
old=json.loads(prior.read_text(encoding='utf-8'))
old.setdefault('access_history',[]).append({k:old[k] for k in ['checked_at','access_state','browser_observation','base_api_error_code'] if k in old})
old.pop('browser_observation',None);old.pop('base_api_error_code',None)
old.update(checked_at=utc_now(),access_state='readable',underlying_checkpoint='resolved: epoch_44.pth plus classifier overlay epoch_98.pth',
           paired_configuration='resolved: pinned config/3-0-0-temporal-dit.py and active training switches',
           eligibility_for_model_evaluation='inputs resolved; GPU runtime and loading preflight pending')
prior.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assets_path=Path('configs/known_assets.json');assets=json.loads(assets_path.read_text(encoding='utf-8'))
assets['provided_pcm_model_reference']['access_state']='readable'
assets['model_accuracy_readiness']=str(root/'model-input-binding.json')
assets['model_accuracy_input_binding']=str(root/'model-input-binding.json')
assets_path.write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':binding['state'],'model_version':'v3.2.2','checkpoints':2,'matching_base_parameters':883,
                  'frozen_small_scene_frames':selected,'new_inference_frames':0,'remaining_user_action':'LPAI startup card confirmation'},ensure_ascii=True))
