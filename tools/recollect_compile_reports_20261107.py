"""Repair collection metadata only; all native executions remain immutable."""
import json
from pathlib import Path
from deploy_environment_flow import remote

CODE='''import pathlib,sys,json,copy,shutil,numpy as np
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4');sys.path[:0]=[str(release),str(release/'tools')]
from pcm_eval.common import digest,save_json,load_json
from pcm_eval.compile_aggregate import aggregate_compile
from pcm_eval.combined_delivery import write_combined
from pcm_eval.evaluation import EvaluationRuntime
from evaluation_worker import evaluate
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-live-20261107-r4')
root=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-997e5dda6e8546a4a935')
workflow=load_json(root/'workflow.json');cfg=load_json(release/'configs/evaluation.json')
def recollect(output,plan):
 scope=output/'native-scope';old_pointer=load_json(output/'collection.json');old_receipt=load_json(output/'execution-receipt.json');old_result=load_json(old_receipt['result_file'])
 repair=output/'report-contract-recollection';repair.mkdir();shutil.copy2(output/'collection.json',repair/'before-pointer.json');shutil.copy2(output/'execution-receipt.json',repair/'before-execution-receipt.json')
 request=copy.deepcopy(plan['request']);request['evaluation'].update(expected_frames=2,max_frames=2,selection={'frames':plan['frames']},compute_gt_metrics=True)
 target=repair/'collection'
 aggregate_compile([scope/'compile/frames'/f'frame_{i:06d}' for i in range(2)],plan['frames'],target,request=request,run_id=plan['run_id'],checkpoint=scope/'compile/checkpoint.npz')
 new_result=load_json(target/'normalized-result.json')
 values=lambda r:[(m['name'],m['value'],m.get('unit'),m.get('dimensions')) for m in r['metrics']]
 assert values(old_result)==values(new_result),'Metric values changed'
 for b in ('board','onnx'):assert load_json(pathlib.Path(old_receipt['result_file']).parent/('metrics_'+b+'.json'))==load_json(target/('metrics_'+b+'.json'))
 save_json(target/'collection-binding.json',{'plan_sha256':plan['plan_sha256'],'checkpoint_sha256':digest(scope/'compile/checkpoint.npz'),'source_manifest_sha256':digest(target/'native-source-manifest.json'),'result_sha256':digest(target/'normalized-result.json')})
 native=copy.deepcopy(old_receipt);native.update(result_file=str(target/'normalized-result.json'),result_sha256=digest(target/'normalized-result.json'))
 save_json(output/'collection.json',{'plan_sha256':plan['plan_sha256'],'directory':str(target)},overwrite=True);save_json(output/'execution-receipt.json',native,overwrite=True)
 return native
canary_native=recollect(stage/'execution',load_json(stage/'plan.json'))
shutil.copy2(stage/'result.json',stage/'report-contract-fix/before-canary-result.json');save_json(stage/'result.json',canary_native,overwrite=True)
native=recollect(root/'effective/compile_alignment/execution',workflow['jobs'][1]['plan'])
inference=root/'effective/compile_alignment/inference.json';old=load_json(inference);assert old['state']=='SUCCEEDED'
shutil.copy2(inference,root/'effective/compile_alignment/execution/report-contract-recollection/before-inference.json');old['receipt']=native;save_json(inference,old,overwrite=True)
model_hash=digest(root/'effective/model_accuracy/inference.json')
def no_inference(*args,**kwargs):raise AssertionError('Report recovery must not execute any inference')
run_cfg=copy.deepcopy(cfg);run_cfg['bench_available']=True
original,result=evaluate(copy.deepcopy(workflow),root,run_cfg,executor=no_inference)
compiled=next(c for c in result['chains'] if c['kind']=='compile_alignment');assert compiled['state']=='SUCCEEDED',compiled.get('issues')
assert model_hash==digest(root/'effective/model_accuracy/inference.json')
report=write_combined(original,result,root/'combined/generation',reference_file=cfg['reference_file'],template_file=release/'configs/report_template.json')
EvaluationRuntime(root.parents[1]).queue_publication(root.name,report,result)
proof={'state':'QUEUED_REPORT_UPDATE','task':root.name,'report_compile_scope':workflow['jobs'][1]['plan']['frames'],'report_compile_source':'verified historical same-version results','new_inference_frames':0,'model_result_unchanged':True,'metrics_unchanged':True}
save_json(stage/'backfill-status.json',proof,overwrite=True);print(json.dumps(proof))
'''

if __name__=='__main__':
    result=remote(CODE,timeout=100)
    Path('evidence/compile-live-20261107/recollection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
