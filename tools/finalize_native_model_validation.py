import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.parsers import collect_model
root=Path('evidence/model-evaluation/native-v3.2.2-smoke-20261027')
log=next((root/'run-01').glob('*/*.log'))
result=collect_model(root,'v3.2.2',expected_frames=2,log_file=log)
assert result['execution']['state']=='succeeded' and result['collection']['status']=='complete'
(root/'normalized_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
ref=Path('evidence/model-evaluation/model-reference-20261027')
binding=json.loads((ref/'model-input-binding.json').read_text())
receipt=json.loads((root/'execution_receipt.json').read_text())
binding.update(state='small_scene_occ_validation_succeeded',new_model_inference_frames=2,
 first_validation_result=str(root/'normalized_result.json'),user_missing_weight_or_config_information=[],
 remaining_actions=['Choose classification-labelled small samples for vision-head scoring',
                    'Confirm the desired trajectory/mesh evaluation scope before enabling that head',
                    'Business confirmation of formal scene coverage and runtime budget; no full evaluation started'])
binding['loading']['validation']='passed: strict 1263-key main load, 216-key classifier overlay, zero missing/unexpected/shape mismatches'
binding['environment'].update(platform_start_state='started_via_browser_at_user_request',observed_state='ready',
 runtime_torch_custom_ops_preflight='passed; existing matching PointPillars binaries reused, fixed-source DAF and LocalAggregate built in isolated directory',
 dataset_access_preflight='passed: 9 files / 13,419,177 bytes reused from workstation cache; no full dataset download',
 runtime={'python':'3.12.3','torch':'2.7.1+cu126','cuda_runtime':'12.6','nvcc':'12.8'},
 run_id=receipt['run_id'],execution_seconds=receipt['duration_seconds'])
binding['first_validation_overrides'].update(NUM_FRAMES='1',CKPT_SKIP_KEYS='')
(ref/'model-input-binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':binding['state'],'metrics':len(result['metrics']),'coverage':result['coverage']}))
