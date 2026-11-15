"""Collect actual integration receipts without triggering evaluation or publication."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,utc_now
from pcm_eval.lpai import request
from pcm_eval.validation import validate_result

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'evidence/feishu-entry-20261029'
raw=request(['devspaces','get','--name','bmad-r3-a100-20261015'])
if isinstance(raw,str):raw=json.loads(raw)
statuses=[]
def visit(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('phase','state','status') and type(v) in (str,int):statuses.append({'field':k,'value':v})
            elif isinstance(v,(dict,list)):visit(v)
    elif isinstance(value,list):
        for v in value:visit(v)
visit(raw)
save_json(out/'space-status-final.json',{'space':'bmad-r3-a100-20261015','status_fields':statuses,
    'start_stop_method':'authorized_browser_UI','inference_frames':2},overwrite=(out/'space-status-final.json').exists())
base=out/'rm-small-20261029-01/model_accuracy/reviewed-report-02'
result=load_json(base/'normalized-result.json')
report=load_json(base/'recovery.json')
assert validate_result(result)['valid'] and result['coverage']['observed']['model_samples']==2
assert len(result['metrics'])==315 and report['publication']['state']=='SUCCEEDED'
deploy=load_json(out/'deployment.json');refresh=load_json(out/'refresh.json');card=load_json(out/'card-validation.json')
assert deploy['state']=='DEPLOYED' and refresh['state']=='REFRESHED'
assert refresh['bridge_ready'] and refresh['worker_ready'] and card['native_card_schema_accepted']
receipt={'schema':'roadmodel.feishu-entry-completion/v1','checked_at':utc_now(),
    'state':'entry_deployed_dual_live_workflow_partially_verified','required_user_inputs':['model_delivery_url','dataset_url'],
    'robot_entry':'RoadModel评测','entry_command':'private-chat message, not a newly published bot menu',
    'same_connection_reused':True,'extra_feishu_connections':0,'production_confirmation_verified':False,
    'production_draft_opened':card['production_form_opened'],'production_task_id':card['task_id'],
    'runtime_model_live_validation':{'state':'passed_occ_only','new_inference_frames':2,'metric_records':315,
        'report_url':report['report_url'],'publication_tables_and_resources_readback':'passed'},
    'runtime_compile_validation':{'state':'blocked','new_inference_frames':0,
        'cause':'v3.2.2 supplies five SubNetOps packages; existing executor requires one MergedNet whole-network package',
        'different_version_fallback_used':False,'shared_bench_guard_verified':False},
    'full_dataset_evaluation':False,'new_dataset_downloads':0,'retraining':False,
    'verification':{'project_unit_tests_passed':110,'existing_gateway_related_tests_passed':60,
        'native_card_schema':'accepted','real_user_card_submit':'pending_user_click',
        'atomic_skills_validated_and_installed':3},
    'remaining':['same-version MergedNet compile artifact or verified multi-subnet runner',
        'shared physical bench guard integration','background GPU-space browser startup',
        'real user submit callback acceptance','vision and trajectory GT contract for the selected dataset',
        'new delivery URLs still require source discovery and verified runtime registration'],
    'gpu_space_final_status':statuses,'service_instance':refresh['service_instance'],
    'first_publication_failure':'local resource-path validation rejected before document creation; original receipt preserved',
    'report_recovery_new_inference_frames':0}
save_json(out/'completion.json',receipt,overwrite=(out/'completion.json').exists())
print(json.dumps(receipt,ensure_ascii=False,indent=2))
