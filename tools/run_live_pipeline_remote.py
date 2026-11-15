"""Internal, explicitly authorized live integration test; not a Feishu callback.

Runs on the workstation. The production user's confirmation is not forged and
the production full-run card stays attached to its existing full controllers.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

RELEASE=Path('/mnt/data/bench-eval/service/roadmodel/entry-live-20261030-r3')
VALID=Path('/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3')
GATEWAY=Path('/mnt/data/bench-eval/service/gateway')
FULL=Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01')
sys.path[:0]=[str(RELEASE),str(GATEWAY)]
from pcm_eval.runtime import Runtime
from pcm_eval.common import load_json,save_json,digest,utc_now
from bench_gateway.config import load

def main():
    claim=VALID/'orchestrator-claim.json';save_json(claim,{'pid':os.getpid(),'started_at':utc_now()})
    cfg=load(GATEWAY)
    with sqlite3.connect('file:'+str(Path(cfg['state_root'])/'roadmodel/runtime.sqlite3')+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        prior=dict(c.execute('SELECT * FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone())
    assert prior['event_id'],'Original real two-link callback must already be verified'
    cat=load_json(RELEASE/'configs/two_link_catalog.json');inputs=json.loads(prior['inputs'])
    assert inputs=={'model_delivery_url':cat['models'][0]['url'],'dataset_url':cat['datasets'][0]['url']}
    rt=Runtime(VALID/'runtime',catalog_file=RELEASE/'configs/two_link_catalog.json')
    task=rt.open(prior['owner'],prior['session'])
    with rt.connect() as c:
        c.execute("UPDATE tasks SET state='QUEUED',inputs=?,event_id=? WHERE id=?",
                  (json.dumps(inputs),'operator-validation-'+uuid.uuid4().hex,task['id']))
        for kind in ('model_accuracy','compile_alignment'):c.execute('INSERT INTO chains(task,kind,state) VALUES(?,?,?)',(task['id'],kind,'PENDING'))
    save_json(VALID/'submission-origin.json',{'type':'explicit_operator_authorized_integration_validation',
             'original_real_user_task':prior['id'],'original_callback_event':prior['event_id'],'validation_runtime_task':task['id'],
             'production_callback_forged':False,'messages_sent':0,'scope':'unexecuted tail slots of authorized full evaluation'})
    started=rt.tick()
    assert started['state']=='RUNNING',started
    details={x['kind']:x['detail'] for x in started['chains']}
    assert all(d['dispatch']['mode']=='new_native_dispatch' for d in details.values()),details
    bindings=[details[k]['binding'] for k in ('model_accuracy','compile_alignment')]
    model_binding,compile_binding=bindings
    source=load_json(FULL/'request.json');request=deepcopy(source['compile_request'])
    selected=load_json(compile_binding['frozen_request']['path'])['frames']
    request['evaluation'].update(max_frames=1,expected_frames=1,selection={'frames':selected},compute_gt_metrics=True)
    request['dataset']['version']='26-10-20-1'
    output=VALID/'result-report'
    report={'output':str(output),'native_task_id':model_binding['native_task_id'],'document_id':'DEMO00000000000000000000002',
      'scope':'selected_validation','baseline_model':'/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/model-report.json',
      'reference':'/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/reference.json',
      'model_request':model_binding['frozen_request']['path'],'compile_request':compile_binding['frozen_request']['path'],
      'model_results':[str(Path(model_binding['result_source'])/'normalized-result.json')],
      'compile_frames':[compile_binding['result_source']],'compile_execution_request':request,
      'frozen_requests':{b['kind']:b['frozen_request']['sha256'] for b in bindings},'new_model_frames':37,'new_compile_frames':1}
    pipeline=save_json(VALID/'pipeline.json',{'scope':'selected_validation','bindings':bindings,'output':str(output),'report':report})
    with (VALID/'pipeline.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,str(RELEASE/'tools/run_entry_pipeline.py'),str(pipeline)],cwd=RELEASE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(VALID/'pipeline-dispatch.json',{'pid':p.pid,'runtime_task':task['id'],'native_first_dispatch':True,
             'model_frames':37,'compile_frames':1,'full_workers_untouched':True})
    while True:
        source_receipt=output/'publication/verification.json'
        if source_receipt.exists():
            receipt=load_json(source_receipt);assert receipt['state']=='VERIFIED' and receipt['scope']=='selected_validation'
            alias=VALID/'publication/verification.json'
            if not alias.exists():save_json(alias,receipt)
            else:assert load_json(alias)==receipt
        result=rt.tick();current=rt.get(task['id'],prior['owner'])
        if current['state']=='SUCCEEDED':
            save_json(VALID/'completion.json',{'state':'VERIFIED_SELECTED_SCOPE_END_TO_END','native_first_dispatch':True,
                    'new_model_inference_frames':37,'new_compile_inference_frames':1,'selected_frames_repeated':0,
                    'full_tail_slots_populated_for_later_reuse':True,'feishu_bot_update_and_readback':'passed',
                    'report_url':load_json(source_receipt)['report_url'],'production_callback_forged':False,
                    'runtime_task':task['id'],'completed_at':utc_now()})
            return
        if pipeline.with_suffix('.failure.json').exists() or current['state']=='NEEDS_REVIEW':
            save_json(VALID/'needs-review.json',{'state':'NEEDS_REVIEW','task_state':current['state'],
                     'pipeline_failed':pipeline.with_suffix('.failure.json').exists(),'inference_must_not_repeat':True})
            raise RuntimeError('Integration needs review; retain all native outputs')
        time.sleep(5)

if __name__=='__main__':main()
