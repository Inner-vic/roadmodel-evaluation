"""Exercise fresh online material preparation in an isolated queue, no inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from pcm_eval.common import load_json, save_json
from deploy_delivery_demo import call, LOCAL, RELEASE, STAGE


def queue():
    code = r'''import sys,pathlib,json,time
sys.path.insert(0,RELEASE)
from pcm_eval.delivery_demo import DemoRuntime
from pcm_eval.common import json_digest
cfg=json.loads((pathlib.Path(RELEASE)/'configs/delivery_demo.json').read_text())
root=pathlib.Path(STAGE)/'live-material-validation';rt=DemoRuntime(root)
request={'model_delivery_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'dataset_ref':'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1'}
with rt.connect() as c:
 c.execute('insert into materials values(?,?,?,?,?,?)',('rm-demo-source-check',json.dumps(request),json_digest(request),'QUEUED','{}',time.time()))
print(json.dumps({'release':RELEASE,'state_root':str(root),'owner':cfg['owner'],'agent_id':cfg['agent_id'],'python':sys.executable}))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'validation-agent-config.json', result)
    print(json.dumps({'state': 'ISOLATED_SOURCE_CHECK_QUEUED', 'new_inference_frames': 0}))


def check():
    code = r'''import sys,pathlib,json,time
sys.path.insert(0,RELEASE)
from pcm_eval.delivery_demo import DemoRuntime
root=pathlib.Path(STAGE)/'live-material-validation';rt=DemoRuntime(root)
with rt.connect() as c:r=dict(c.execute('select * from materials where task=?',('rm-demo-source-check',)).fetchone())
assert r['state']=='PREPARED',r['state'];d=json.loads(r['detail']);assert d.get('workflow'),d
w=d['workflow'];assert w['scope']=='validation' and all(j['ready'] for j in w['jobs']),w
counts={j['kind']:j['request']['evaluation']['expected_frames'] for j in w['jobs']}
assert counts=={'model_accuracy':2,'compile_alignment':1}
assert not (root/'runs').exists(),'This check must never dispatch inference'
receipt={'state':'VERIFIED','live_online_record_read':True,'source_agent_roundtrip':True,'counts':counts,'new_inference_frames':0}
(pathlib.Path(STAGE)/'live-source-verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'live-source-verification.json', result)
    print(json.dumps(result))


def health():
    code = r'''import sys,pathlib,json,time
sys.path[:0]=[RELEASE,'/mnt/data/bench-eval/service/gateway']
from bench_gateway.config import load
from bench_gateway.deployment import status
from pcm_eval.delivery_demo import DemoRuntime
cfg=load(pathlib.Path('/mnt/data/bench-eval/service/gateway'));s=status(cfg)
rt=DemoRuntime(pathlib.Path(cfg['state_root'])/'roadmodel/delivery-demo')
with rt.connect() as c:
 agents=[dict(r) for r in c.execute('select * from source_agent')]
 tasks=[dict(r) for r in c.execute('select state,count(*) as count from tasks group by state')]
age=time.time()-max([r['heartbeat'] for r in agents],default=0)
assert s['active'] and s['phase']=='RUNNING' and all(c.get('ready') for c in s['children'].values())
assert age<45,age
import ast,re
plugin=pathlib.Path('/mnt/data/bench-eval/service/gateway/bench_gateway/roadmodel_plugin.py').read_text()
active=ast.literal_eval(re.search(r'ROOT\s*=\s*Path\(([^\n]+)\)',plugin).group(1))
tests={}
for p in pathlib.Path(STAGE).glob('r2-*.log'):
 text=p.read_text();m=re.search(r'Ran (\d+) tests?',text)
 if m:tests[p.name]=int(m.group(1))
receipt={'state':'HEALTHY','release':active,'service_instance':s['instance'],'bridge_ready':s['children']['bridge']['ready'],
 'worker_ready':s['children']['worker']['ready'],'source_agent_heartbeat_age_seconds':round(age,2),'tasks':tasks,
 'new_production_submissions':0,'real_card_callback_verified':False,'bench_available':False,
 'workstation_tests':tests,'workstation_test_count':sum(tests.values())}
(pathlib.Path(STAGE)/'health.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'health.json', result, overwrite=True)
    print(json.dumps(result))


def start_live():
    code = r'''import sys,pathlib,json,subprocess,time
sys.path.insert(0,RELEASE)
from pcm_eval.delivery_demo import DemoRuntime
from pcm_eval.common import save_json
stage=pathlib.Path(STAGE);rt=DemoRuntime(stage/'live-material-validation')
with rt.connect() as c:r=dict(c.execute('select * from materials where task=?',('rm-demo-source-check',)).fetchone())
w=json.loads(r['detail'])['workflow'];assert w['scope']=='validation'
run=stage/'live-execution-validation';run.mkdir(exist_ok=False);save_json(run/'workflow.json',w)
with (run/'worker.log').open('xb') as f:
 p=subprocess.Popen([sys.executable,str(pathlib.Path(RELEASE)/'tools/delivery_demo_worker.py'),str(run)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'STARTED','pid':p.pid,'scope':{'model_occ_frames':2,'compile_tensor_frames':0},
 'isolated_validation_not_a_production_callback':True,'bench_connection_dispatched':False}
save_json(run/'dispatch.json',receipt);print(json.dumps(receipt))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'live-execution-dispatch.json', result)
    print(json.dumps(result))


def live_status():
    code = r'''import pathlib,json
root=pathlib.Path(STAGE)/'live-execution-validation';p=root/'progress.json'
if p.exists():print(p.read_text())
else:
 result={'state':'RUNNING'}
 for kind in ['model_accuracy','compile_alignment']:
  r=root/'effective'/kind/'chain.json'
  if r.exists():result[kind]=json.loads(r.read_text())
 log=root/'worker.log';result['log_tail']=log.read_text()[-2500:];print(json.dumps(result))
'''
    result = call('STAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'live-execution-status.json', result, overwrite=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['queue', 'check', 'health', 'start-live', 'live-status'])
    args = p.parse_args()
    {'queue': queue, 'check': check, 'health': health, 'start-live': start_live, 'live-status': live_status}[args.action]()
