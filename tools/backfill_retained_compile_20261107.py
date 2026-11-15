"""Collect the original card's exact historical scope without any inference."""
import json
from pathlib import Path
from deploy_environment_flow import remote

WORKER='''import pathlib,json,sys,os,copy,shutil,importlib.util,numpy as np,traceback
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
sys.path[:0]=[str(release),str(release/'tools')]
from pcm_eval.common import digest,json_digest,save_json,load_json
from pcm_eval.compile_scope import prepare,execute
from pcm_eval.combined_delivery import write_combined
from pcm_eval.evaluation import EvaluationRuntime
from evaluation_worker import evaluate
root=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-997e5dda6e8546a4a935')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-live-20261107-r4')
status=stage/'backfill-status.json'
try:
 canary=load_json(stage/'status.json');assert canary['state']=='SUCCEEDED'
 workflow=load_json(root/'workflow.json');job=workflow['jobs'][1];plan=job['plan'];cfg=load_json(release/'configs/evaluation.json')
 assert workflow['scope']=='validation' and plan['expected_frames']==2
 assert not (root/'effective/compile_alignment/inference.json').exists(),'Never replace prior inference'
 model=load_json(root/'effective/model_accuracy/inference.json');model_digest=digest(root/'effective/model_accuracy/inference.json')
 assert model['state']=='SUCCEEDED' and digest(pathlib.Path(model['receipt']['result_file']))==model['receipt']['result_sha256']
 full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
 checkpoint_before=digest(full/'compile/checkpoint.npz');request=load_json(full/'request.json')
 target=root/'effective/compile_alignment/execution';prepared=prepare(plan,target/'native-scope',cfg['compile_profile']);scope=pathlib.Path(prepared['task_root'])
 sources=[]
 for i,pair in enumerate(plan['frames']):
  old=full/'compile/frames'/f"frame_{request['frames'].index(pair):06d}";new=scope/'compile/frames'/f'frame_{i:06d}'
  receipt=load_json(old/'execution_receipt.json')
  assert receipt['state']=='succeeded' and receipt['release']['state']=='RELEASED' and receipt['selected_frames']==[pair]
  assert receipt['source_onnx_sha256']==plan['request']['model']['onnx_sha256'] and receipt['data_link_sha256']==plan['index_sha256']
  assert {r['stage']:r['sha256'] for r in receipt['subnet_archives']}=={r['stage']:r['sha256'] for r in plan['request']['model']['subnet_archives']}
  new.mkdir(exist_ok=False)
  names=['execution_receipt.json','cosine_summary.json','input_validation.json','output_integrity.json','gt_frame_receipt.json','gt_metrics_board.json','gt_metrics_onnx.json','board_final_occ.npy','onnx_final_occ.npy']
  for name in names:os.link(old/name,new/name)
  sources.append({'frame':pair,'source':str(old),'finished_at':receipt['finished_at'],'source_files':{name:digest(old/name) for name in names}})
 os.environ.update(prepared['environment']);sys.path.insert(0,str(scope/'code'))
 spec=importlib.util.spec_from_file_location('retained_count_recovery',prepared['controller']);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
 totals={b:controller.c.Counts() for b in ('board','onnx')}
 for i in range(2):
  counts=controller.ensure_gt(i,scope/'compile/frames'/f'frame_{i:06d}')
  for b in totals:totals[b].confusion+=counts[b][0];totals[b].ds+=counts[b][1];totals[b].num_frames+=1
 with (scope/'compile/checkpoint.npz').open('wb') as handle:
  np.savez(handle,completed=2,**{b+'_confusion':totals[b].confusion for b in totals},**{b+'_ds':totals[b].ds for b in totals})
 save_json(scope/'dispatch.json',{'state':'ADOPTED_RETAINED_RESULTS','plan_sha256':plan['plan_sha256'],'started_at':min(load_json(pathlib.Path(s['source'])/'execution_receipt.json')['started_at'] for s in sources),'new_inference_frames':0})
 save_json(scope/'compile/progress.json',{'state':'SUCCEEDED','task_id':plan['run_id'],'expected_frames':2,'completed_frames':2,'new_inference_frames':0,'retained_results':sources})
 save_json(target/'retained-adoption.json',{'state':'VERIFIED','sources':sources,'new_inference_frames':0,'original_timestamps_preserved':True})
 old=load_json(root/'effective/workflow.json');expected=copy.deepcopy(workflow);expected['jobs'][1]['ready']=False;expected['jobs'][1]['issues']=list(dict.fromkeys(expected['jobs'][1]['issues']+['台架暂不可用，编译材料准备结果已保存']));expected.pop('workflow_sha256');expected['workflow_sha256']=json_digest(expected)
 assert old==expected,'Unknown legacy workflow changes'
 archive=root/'retained-r3-preparation';archive.mkdir()
 with np.load(root/'compile-scope/compile/checkpoint.npz',allow_pickle=False) as cp:assert int(cp['completed'])==0
 shutil.move(root/'compile-scope',archive/'compile-scope');shutil.move(root/'effective/workflow.json',archive/'effective-workflow.json');save_json(root/'effective/workflow.json',workflow)
 # Only collection of the adopted exact scope is allowed by this executor.
 def collect_only(p,where):
  assert p['plan_sha256']==plan['plan_sha256'] and pathlib.Path(where)==target
  return execute(p,where,cfg['compile_profile'],bench_available=False)
 runtime_cfg=copy.deepcopy(cfg);runtime_cfg['bench_available']=True
 original,result=evaluate(copy.deepcopy(workflow),root,runtime_cfg,executor=collect_only)
 assert next(c for c in result['chains'] if c['kind']=='compile_alignment')['state']=='SUCCEEDED',result
 assert digest(root/'effective/model_accuracy/inference.json')==model_digest
 assert digest(full/'compile/checkpoint.npz')==checkpoint_before
 report=write_combined(original,result,root/'combined/generation',reference_file=cfg['reference_file'],template_file=release/'configs/report_template.json')
 EvaluationRuntime(root.parents[1]).queue_publication(root.name,report,result)
 proof={'state':'QUEUED_REPORT_UPDATE','task':root.name,'model_inference_repeated':False,'report_compile_scope':plan['frames'],'report_compile_source':'verified historical same-version results','report_compile_new_inference_frames':0,'canary_new_inference_frames':2,'canary_elapsed_seconds':canary['elapsed_seconds'],'full_checkpoint_unchanged':True}
 status.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
except Exception as error:
 status.write_text(json.dumps({'state':'NEEDS_REVIEW','error':str(error)},indent=2));traceback.print_exc();raise
'''

START='''import pathlib,sys,json,subprocess
root=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-live-20261107-r4')
assert json.loads((root/'status.json').read_text())['state']=='SUCCEEDED'
script=root/'backfill.py';assert not script.exists(),'Keep prior backfill receipts; do not repeat'
script.write_bytes(sys.stdin.buffer.read())
with (root/'backfill.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'state':'STARTED','pid':child.pid,'purpose':'exact-scope historical collection and same-document backfill, no inference'}))
'''

if __name__=='__main__':
    result=remote(START,WORKER.encode(),timeout=35)
    Path('evidence/compile-live-20261107/backfill-dispatch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
