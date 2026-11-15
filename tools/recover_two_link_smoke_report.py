"""Repair report metadata and resource paths from the successful two-frame run. No inference."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.atomic_result import prepare_model_result
from pcm_eval.common import load_json,save_json
from pcm_eval.model_report import write_model_report
from pcm_eval.publication import publish_report
from pcm_eval.two_link import resolve_pair

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'evidence/feishu-entry-20261029/rm-small-20261029-01/model_accuracy'
old=load_json(base/'execution/execution_receipt.json')
assert old['state']=='succeeded' and old['steps'][-1]['name']=='model_inference' and old['steps'][-1]['exit_code']==0
job=resolve_pair(load_json(ROOT/'examples/two_link_submission.json'),'rm-small-20261029-01')['jobs'][0]
original=load_json(base/'execution/plan.json')
assert original['source']==job['plan']['source'] and original['input_identity']==job['plan']['input_identity']
assert original['dataset']==job['plan']['dataset']
request=json.loads(json.dumps(job['request']))
request['evaluation'].pop('selected_frames')
assert request==load_json(base/'request.json'),'Only recovered selected-frame provenance may differ'
reviewed=base/'reviewed-report-02'
if reviewed.exists():raise ValueError('Inspect previous report recovery; do not repeat publication')
reviewed.mkdir()
bound=prepare_model_result(job,old,reviewed/'normalized-result.json')
report=write_model_report(reviewed/'generated',result_file=bound,binding_file=job['binding_file'],
                         gt_preparation_file=job['gt_preparation_file'],
                         supporting_artifacts=[base/'request.json',base/'execution/plan.json',base/'execution/execution_receipt.json'])
publication=publish_report(report['report_xml'],reviewed/'publication')
receipt={'state':'PARTIAL','new_inference_frames':0,'successful_inference_frames_reused':2,
         'run_id':old['run_id'],'report':report,'publication':publication,'report_url':publication['report_url'],
         'issues':job['warnings'],'old_publication_preserved':str(base/'publication/publication.json')}
save_json(reviewed/'recovery.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
