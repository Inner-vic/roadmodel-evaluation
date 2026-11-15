"""Reviewed recovery of only the board work proven not to have dispatched."""
from copy import deepcopy
import json,os,signal,sqlite3,subprocess,sys,time
from pathlib import Path
RELEASE=Path('/mnt/data/bench-eval/service/roadmodel/entry-live-20261030-r3')
VALID=Path('/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3')
ATTEMPT=VALID/'board-admission-02'
sys.path.insert(0,str(RELEASE))
from pcm_eval.common import load_json,save_json,digest,utc_now
from pcm_eval.runtime import Runtime,execute_atomic

def main():
    root=Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01/compile/frames/frame_008336')
    assert not (root/'execution.json').exists() and not (root/'execution_receipt.json').exists()
    assert load_json(root/'prepare.json')['state']=='READY'
    old=load_json(VALID/'compile/progress.json');assert old['state']=='NEEDS_REVIEW' and old['error']=='Original Runner has work; no hardware dispatch'
    assert load_json(VALID/'model/progress.json')['state']=='OCC_SUCCEEDED'
    assert not (VALID/'result-report/generation.json').exists() and not (VALID/'result-report/publication/journal.json').exists()
    pid=load_json(VALID/'pipeline-dispatch.json')['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'run_entry_pipeline.py' in cmd and str(VALID/'pipeline.json') in cmd
    children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().strip();assert not children
    save_json(VALID/'queue-recovery-proof.json',{'no_NPU_execution_record':True,'ONNX_ready_reused':True,'model_37_completed_kept':True,
             'stopped_waiter_pid':pid,'waiter_had_no_children':True,'publication_not_started':True,'at':utc_now()})
    os.kill(pid,signal.SIGTERM) # only this verified read-only waiter, never a native controller.
    rt=Runtime(VALID/'runtime',catalog_file=RELEASE/'configs/two_link_catalog.json')
    task_id=load_json(VALID/'pipeline-dispatch.json')['runtime_task']
    with rt.connect() as c:row=c.execute('SELECT owner FROM tasks WHERE id=?',(task_id,)).fetchone()
    task=rt.get(task_id,row['owner']);save_json(VALID/'before-queue-recovery.json',task)
    chain=next(x for x in task['chains'] if x['kind']=='compile_alignment');assert chain['state']=='NEEDS_REVIEW'
    binding=deepcopy(chain['detail']['binding']);binding['execution_root']=str(ATTEMPT)
    script=VALID/'entry_compile_admit.py';program=binding['launch']['programs'][0]
    program.update(script=str(script),sha256=digest(script));program['env']['ROADMODEL_VALIDATION_ROOT']=str(ATTEMPT)
    record_path=save_json(VALID/'compile-admission-02-binding.json',binding)
    job={'kind':'compile_alignment','version':'v3.2.2','dataset_id':binding['dataset_id'],'material_root':str(VALID),
         'active_execution':{'path':record_path.name,'sha256':digest(record_path)}}
    detail=execute_atomic(job,ATTEMPT/'runtime-execution');assert detail['state']=='RUNNING'
    with rt.connect() as c:
        c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',('RUNNING',json.dumps(detail,ensure_ascii=False),task_id,'compile_alignment'))
        c.execute("UPDATE tasks SET state='RUNNING',notice=? WHERE id=?",('板端此前未派发；现续跑板端，不重复模型或ONNX。',task_id))
    config=load_json(VALID/'pipeline.json');config['bindings']=[binding if b['kind']=='compile_alignment' else b for b in config['bindings']]
    path=save_json(VALID/'pipeline-admission-02.json',config)
    with (VALID/'pipeline-admission-02.log').open('xb') as log:
        proc=subprocess.Popen([sys.executable,str(RELEASE/'tools/run_entry_pipeline.py'),str(path)],cwd=RELEASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(VALID/'queue-recovery-dispatch.json',{'pipeline_pid':proc.pid,'native_board_dispatch':'new after proven pre-dispatch rejection','repeated_completed_frames':0})
    while True:
        v=VALID/'result-report/publication/verification.json'
        if v.exists() and not (VALID/'publication/verification.json').exists():save_json(VALID/'publication/verification.json',load_json(v))
        rt.tick();current=rt.get(task_id,row['owner'])
        if current['state']=='SUCCEEDED':
            save_json(VALID/'completion.json',{'state':'VERIFIED_SELECTED_SCOPE_END_TO_END','native_first_dispatch':True,
                'new_model_inference_frames':37,'new_compile_inference_frames':1,'repeated_completed_frames':0,
                'full_tail_slots_populated_for_later_reuse':True,'feishu_bot_update_and_readback':'passed',
                'report_url':load_json(v)['report_url'],'production_callback_forged':False,'runtime_task':task_id,'completed_at':utc_now()});return
        if path.with_suffix('.failure.json').exists() or current['state']=='NEEDS_REVIEW':raise RuntimeError('Review preserved outputs; do not repeat native work')
        time.sleep(5)

if __name__=='__main__':main()
