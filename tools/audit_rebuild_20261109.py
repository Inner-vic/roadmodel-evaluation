"""Read-only current-account rebuild and existing deployment health evidence."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from deploy_environment_flow import remote
from pcm_eval.lpai import request

CODE='''import pathlib,json,sys,sqlite3,hashlib
base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path.insert(0,str(base))
from bench_gateway.config import load
from bench_gateway.deployment import status
live=status(load(base))
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3') as db:
 active=db.execute("select count(*) from tasks where state in ('QUEUED','READY','RUNNING','PREPARING','WAITING_REPORT')").fetchone()[0]
cfg=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6/configs/evaluation.json').read_text())
print(json.dumps({'active':live['active'],'worker_ready':live['children']['worker']['ready'],
 'bridge_ready':live['children']['bridge']['ready'],'active_evaluation_tasks':active,
 'space_name':cfg['profile']['space_name'],'space_uid':cfg['profile']['space_uid'],
 'execution_scope':cfg['execution_scope'],'validation_selection':cfg['validation_selection'],
 'old_checkpoint_sha256':hashlib.sha256((full/'compile/checkpoint.npz').read_bytes()).hexdigest(),
 'old_full_controller_running':(full/'compile/controller.lock').exists()}))
'''

if __name__=='__main__':
    output=ROOT/'evidence/rebuild-initialization-20261109'
    raw=request(['devspaces','list','--namespace','vc-chassis','--cluster','cnhb01-algorithm-bidu-p-03','--page-size','100'])
    spaces=[{'name':s['metadata']['name'],'uid':s['metadata']['uid'],'phase':s['status']['phase']} for s in raw['items']]
    assert raw['total']==len(spaces)
    health=remote(CODE,timeout=45)
    assert health['old_checkpoint_sha256']=='2c737cb749fb6268e4c1f8c630e86f60ebc83e9be490f19b6b8c1b4fca3f0bed'
    assert not health['old_full_controller_running']
    agent=json.loads((ROOT/'.runtime/evaluation-flow-agent/status.json').read_text())
    value={'checked_at':time.time(),'spaces':spaces,'space_count':raw['total'],'deployment':health,
        'desktop_agent':{k:agent.get(k) for k in ('state','pid','updated_at')},
        'new_space_created':any(s['name']=='roadmodel-eval-fa60526293a9959e3bff' for s in spaces),
        'test_space_deleted':not any(s['uid']=='bae3ffe6-03d0-487b-aa92-b2c3c569b13b' for s in spaces),
        'new_inference_frames':0,'production_deployment_changed':False}
    (output/'final-audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value))
