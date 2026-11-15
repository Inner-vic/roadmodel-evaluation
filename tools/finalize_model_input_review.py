"""Summarize confirmed inputs and distinguish human choices from agent checks."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import utc_now

root=Path('evidence/environment/model-input-review-20261027')
task=json.loads((root/'source-task-review.json').read_text(encoding='utf-8'))
ssh=json.loads((root/'ssh-probe.json').read_text(encoding='utf-8'))
snapshot=json.loads(Path('evidence/environment/lpai-source-task-current.json').read_text(encoding='utf-8-sig'))
def configs(value,depth=0):
    if depth>4:return []
    if isinstance(value,str) and value.lstrip().startswith('{'):
        try:return configs(json.loads(value),depth+1)
        except ValueError:return []
    if not isinstance(value,dict):return []
    found=[value] if 'lpai_task' in value else []
    for key in ['data','config','configuration','pipeline_config','pipeline_params','task_config','spec']:
        if key in value:found+=configs(value[key],depth+1)
    return found
images=[]
for config in configs(snapshot):
    image=config.get('lpai_task',{}).get('image_info')
    if isinstance(image,str) and re.fullmatch(r'[A-Za-z0-9_.:/@-]{1,1024}',image):images.append(image)

current=task.get('task',task.get('existing_snapshot_summary',{}))
summary={'schema':'pcm.model-readiness/v2','checked_at':utc_now(),'state':'not_ready',
 'atomic_capabilities':{'evaluation_execution':'existing parameterized eval.py adapter; no new model run yet',
                        'result_collection':'existing independent model_accuracy parser',
                        'native_single_report':'legacy pending template exists; independent result-driven layout still to implement',
                        'pcm_release_composition':'out_of_scope'},
 'confirmed':{'dataset':'smartchassisdata-v3','dataset_version':'26-10-20-1','dataset_class':'SmartChassisDatasetV7',
              'data_link_file':'data_link_26-10-20-1_train859_eval135_v7.json',
              'execution_location':'LPAI mounted data; no workstation dataset copy',
              'source_task_uid':current.get('task_uid'),'source_task_name':current.get('name'),
              'source_task_status':current.get('status'),'source_task_is_evaluation':current.get('evaluation'),
              'source_task_configuration':{'branch':'dev_v41','commit_id_recorded':False,
                  'training_entry':'train.py','config':'config/4-0-0.py','HEAD_MODE':'gs_only',
                  'USE_TEMPORAL':False,'SEQUENTIAL_TRAIN':False,'model_register_configured':False},
              'historical_image_references':images},
 'weight_candidates':json.loads(Path('evidence/environment/lpai-model-input-candidates.json').read_text(encoding='utf-8'))['checkpoint_candidates'],
 'weight_candidate_status':'training input references, not confirmed target evaluation checkpoints',
 'requires_user':[
    {'id':'target_weights','provide':'Target model/version and final PTH/CKPT asset link or environment path; training/output task link can substitute if it identifies the chosen checkpoint',
     'optional_head_weights':'Only if needed by selected heads: trajectory/classifier checkpoint and loading/overlay instructions'},
    {'id':'paired_configuration','provide':'Corresponding training/evaluation task link, or code commit + main config + runtime overrides; identify OCC-only or additional heads and temporal mode'},
    {'id':'usable_environment','provide':'Accessible LPAI devspace link/ID or a known runnable evaluation environment; existing success command if available'}],
 'agent_to_verify':['checkpoint/config/dependency hashes','exact Git commit and inherited config closure',
                    'checkpoint keys and head overlay order','actual Python/Conda image, GPU and custom ops',
                    'real data/weight/code mount paths and writable output directory',
                    'effective ROI/grid/labels/masks and GT geometry','small sample scope and result-source provenance'],
 'environment_probe':ssh,
 'not_required_from_user':['dataset re-upload or workstation copy','baseline model/version','manually calculated hashes','passwords or access tokens'],
 'target_weight_need_not_match_prior_compile_reference':True,
 'scope':'Start with a bounded small model-evaluation sample after confirming the model configuration; no full evaluation',
 'new_model_evaluation_executed':False,'new_platform_task_submitted':False}
reference_path=root/'provided-model-reference.json'
if reference_path.exists():
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    summary['provided_model_reference']=reference
    if reference.get('access_state')!='readable':
        summary['requires_user'][0]={'id':'access_existing_model_reference',
            'provide':'Read access to the already supplied model record, or its actual model asset link/path',
            'reason':'The reference exists but its contents are not currently readable'}
    else:
        binding_path=Path('evidence/model-evaluation/model-reference-20261027/model-input-binding.json')
        if binding_path.exists():
            summary['resolved_model_input_binding']=str(binding_path)
            summary['state']='execution_preflight_pending'
            summary['requires_user']=[{'id':'platform_start_confirmation','provide':'Confirm the LPAI startup card already sent to Feishu'}]
(root/'readiness.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':summary['state'],'source_task_status':current.get('status'),
                  'configuration':summary['confirmed']['source_task_configuration'],
                  'historical_image_references':images,'environment':ssh['state'],
                  'user_input_groups':len(summary['requires_user']),
                  'new_evaluation_started':False},ensure_ascii=True))
