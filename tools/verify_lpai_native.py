"""Operator-authorized two-frame end-to-end validation; no simulated callback."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from deploy_lpai_native import remote, STAGE, LOCAL
from evaluation_agent import exchange, prepare_materials, publish
from pcm_eval.common import load_json, save_json


def config():
    cfg = load_json(LOCAL/'agent-config.json')
    cfg['state_root'] = STAGE+'/native-live-state'
    cfg['local_state'] = str(ROOT/'.runtime/native-live-publication')
    return cfg


def start():
    observed = load_json(ROOT/'evidence/model-evaluation/model-reference-20261027/bmad-r3-a100-20261015.json')
    observed['status']['phase'] = 'ready'
    observed['verification'] = {'source': 'LPAI_UI', 'registered_identity': True}
    observed['spec']['dataVols'].append({'name': 'smartchassisdata-v3', 'type': 'dataset', 'version': '26-10-20-1',
        'mountPath': '/mnt/datasets/smartchassisdata-v3/26-10-20-1', 'mode': 'ReadOnly',
        'envKey': 'LPAI_INPUT_DATASET_1', 'envValue': '/mnt/datasets/smartchassisdata-v3/26-10-20-1', 'fsType': 'ofs'})
    # Values above were read from the visible page. The in-space mount probe is
    # still required and proves the actual readonly volume and index hash.
    observation = {'source': 'LPAI_UI', 'space_name': observed['metadata']['name'], 'namespace': observed['metadata']['namespace'],
                   'observed_at': time.time(), 'exists': True, 'space': observed}
    cfg = config()
    row = remote('''import sys,pathlib,json,time
sys.path.insert(0,NEW)
from pcm_eval.evaluation import EvaluationRuntime
root=pathlib.Path(STAGE)/'native-live-state';rt=EvaluationRuntime(root)
cfg=json.loads((pathlib.Path(NEW)/'configs/evaluation.json').read_text())
stage=pathlib.Path(STAGE);journal=stage/'native-validation.json';assert not journal.exists(),'Inspect original native validation before retrying'
task=rt.open(cfg['owner'],'feishu:oc_native_validation:'+cfg['owner'])
assert task['options']=={'scope':'validation','selection':{'scene_count':1,'frames_per_scene':2}}
value={'model_delivery_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'dataset_url':'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1'}
with rt.connect() as c:
 c.execute("update tasks set state='QUEUED',inputs=?,notice=? where id=?",(json.dumps(value),'用户授权的小场景原生执行验证；未模拟生产回调',task['id']))
 for kind in ('model_accuracy','compile_alignment'):c.execute("insert into chains(task,kind,state) values(?,?,'PENDING')",(task['id'],kind))
rt.tick();row=rt.claim_materials(cfg['agent_id'],cfg['owner'])
journal.write_text(json.dumps({'state':'PREPARING','task':task['id'],'maximum_occ_frames':2,'bench_available':False,'real_user_callback_verified':False}))
print(json.dumps(row))
''')
    exchange(cfg, 'record_space_observation', observation=observation)
    prepare_materials(cfg, row)
    result = remote('''import pathlib,sys,json
sys.path.insert(0,NEW)
from pcm_eval.evaluation import EvaluationRuntime
rt=EvaluationRuntime(pathlib.Path(STAGE)/'native-live-state')
with rt.connect() as c:row=c.execute('select * from materials').fetchone()
detail=json.loads(row['detail']);assert row['state']=='PREPARED' and detail.get('workflow'),detail
workflow=detail['workflow'];job=next(j for j in workflow['jobs'] if j['kind']=='model_accuracy')
assert job['ready'] and job.get('native_adapter')=='lpai_mounted_dataset',job['issues']
assert len(job['plan']['heads']['occ'])==2 and job['plan']['scope']=='validation'
rt.tick()
with rt.connect() as c:task=c.execute('select id,owner from tasks').fetchone()
task=rt.get(task['id'],task['owner']);print(json.dumps({'state':task['state'],'task_id':task['id'],'selected_frames':job['plan']['heads']['occ'],'model_location':'lpai','bench_available':False}))
''')
    save_json(LOCAL/'native-validation-start.json', result)
    print(json.dumps(result))


def status():
    result = remote('''import pathlib,sys,json
sys.path.insert(0,NEW)
from pcm_eval.evaluation import EvaluationRuntime
rt=EvaluationRuntime(pathlib.Path(STAGE)/'native-live-state');rt.tick()
with rt.connect() as c:row=c.execute('select id,owner from tasks').fetchone()
task=rt.get(row['id'],row['owner']);run=rt.root/'runs'/task['id']
out={'state':task['state'],'task_id':task['id'],'chains':[{k:c[k] for k in ('kind','state','detail')} for c in task['chains']]}
if task.get('report_url'):out['report_url']=task['report_url']
log=run/'worker.log'
if log.exists():out['worker_log_tail']=log.read_text(errors='replace')[-3000:]
p=run/'effective/model_accuracy/execution/native-progress.json'
if p.exists():out['native_progress']=json.loads(p.read_text())
print(json.dumps(out,ensure_ascii=False))
''')
    print(json.dumps(result, ensure_ascii=False))


def publication():
    cfg = config()
    row = exchange(cfg, 'claim_publication')
    if not row:
        raise ValueError('原生执行尚未进入报告发布阶段，请先查看原任务状态')
    detail = publish(cfg, row)
    exchange(cfg, 'complete_publication', task=row['task'], digest=row['digest'], detail=detail)
    save_json(LOCAL/'native-publication.json', detail)
    print(json.dumps(detail))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'status', 'publish'))
    args = parser.parse_args()
    {'start': start, 'status': status, 'publish': publication}[args.action]()
