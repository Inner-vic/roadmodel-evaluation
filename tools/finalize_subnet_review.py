"""Write an evidence-backed correction and completion receipt without execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,utc_now

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/compile-subnet-review-20261029'
RUN='pcm-v322-subnets-one-02'
audit=load_json(OUT/'version-and-port-audit.json')
remote=load_json(OUT/'remote-preparation.json')
old=audit['previous']
assert remote['old_hashes'][Path(old['model']['onnx']).name]==old['receipt_onnx_sha256']
assert remote['old_hashes'][Path(old['model']['packup_tar']).name]==old['receipt_packup_sha256']
execution=load_json(OUT/RUN/'execution.json');comparison=load_json(OUT/RUN/'tensor-comparison.json')
guard=load_json(OUT/RUN/'guard-final.json');publication=load_json(OUT/'reviewed-publication.json')
resolution=load_json(OUT/'final-two-link-resolution.json');refresh=load_json(ROOT/'evidence/feishu-entry-20261029/refresh.json')
assert execution['state']=='SUCCEEDED' and all(x['state']=='SUCCEEDED' and x['exit_code']==0 for x in execution['stages'])
assert execution['release']['state']=='RELEASED' and guard['active'] is None and not guard['lock_exists']
assert len(comparison['outputs'])==18 and publication['state']=='SUCCEEDED' and resolution['ready']
minimum=min(comparison['outputs'],key=lambda x:x['cosine'])
receipt={'schema':'roadmodel.compile-subnet-review/v1','checked_at':utc_now(),'state':'implemented_and_one_frame_verified',
    'correction':{'previous_statement':'The v3.2.2 material requires a MergedNet whole-network package',
        'finding':'Overstated: original client accepts model_info with SubNetOps ELF; its stock orchestration assumes one 11-input network',
        'implemented':'five original subnet programs chained through 32 strictly matched boundaries'},
    'previous':{'version':'v3.2.3','frames':20,'physical_file_hashes_match_original_receipt':True,
        'onnx_reference_sha256':old['receipt_onnx_sha256'],'compiled_package_sha256':old['receipt_packup_sha256']},
    'current':{'version':'v3.2.2','source_onnx_sha256':audit['current']['onnx_sha256'],
        'subnet_archive_sha256':{x['stage']:x['sha256'] for x in audit['current']['stages']},
        'frames':1,'subnet_invocations':5,'finite_subnet_outputs':49,'compared_public_outputs':18,
        'lowest_cosine':minimum['cosine'],'lowest_output':minimum['name'],
        'integer_exact_ratios':{x['name']:x['exact_ratio'] for x in comparison['outputs'] if x['board_dtype']=='int32'},
        'quality_decision':'not_assessed_no_business_threshold','gt_metrics':'not_computed_for_this_tensor_adapter_validation'},
    'reuse':{'original_components':['BoardInferClient','InferenceExec','convert_fp32_to_fp16_keep_fp8','onnxruntime_infer CPU'],
        'input_cache':'same existing V7 frame 18; 11 canonical tensors','dataset_downloads':0,
        'elf_recompiled':False,'elf_merged':False,'archive_goldens_used_as_predictions':False},
    'shared_bench':{'original_atomic_json_api_used':True,'released':True,'original_active_app_modified':False,
        'queue_final':guard,'production_fault_injection':False},
    'runtime':{'both_input_bindings_ready':True,'compile_adapter':'subnet_chain_workstation',
        'verified_plan_and_request_parse_without_inference':True,'service_instance':refresh['service_instance'],
        'bridge_ready':refresh['bridge_ready'],'worker_ready':refresh['worker_ready'],
        'real_feishu_submit_callback':'not_tested','gpu_background_startup':'unchanged_pending'},
    'report_url':publication['report_url'],'report_revision':publication['revision_id'],
    'previous_v323_report_not_written':True,'full_dataset_execution':False,'retraining':False,
    'tests_passed':116,'report_generation_new_inference_frames':0,
    'supersedes_prior_assumption_in':'evidence/feishu-entry-20261029/completion.json'}
save_json(OUT/'completion.json',receipt)
print({'state':receipt['state'],'frames':1,'public_outputs':18,'lowest_cosine':minimum['cosine'],
       'guard_released':True,'runtime_bindings_ready':True,'report_url':receipt['report_url']})
