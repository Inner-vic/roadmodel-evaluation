"""Install a narrow plugin into the existing gateway after normal drain and backup."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path('D:/developer/Desktop/Project/台架飞书统一入口')
OUT = ROOT / 'evidence/feishu-entry-20261029'
sys.path[:0] = [str(ROOT), str(TARGET)]


def patches():
    agent = (TARGET / 'bench_gateway/cc_agent.py').read_text(encoding='utf-8')
    worker = (TARGET / 'bench_gateway/worker.py').read_text(encoding='utf-8')
    marker = '        # The CC adapter adds project/platform preambles on initial turns.'
    hook = '''        # RoadModel independent business entry on this authenticated connection.
        from .roadmodel_plugin import handle as roadmodel_handle
        roadmodel_answer = roadmodel_handle(service, session, text)
        if roadmodel_answer is not None:
            emit({'type':'item.completed','item':{'type':'agent_message','text':roadmodel_answer}})
            emit({'type':'turn.completed','usage':{'input_tokens':0,'cached_input_tokens':0,'output_tokens':0}})
            return
'''
    if 'roadmodel_handle' in agent: raise ValueError('Plugin already installed; inspect deployment receipt instead of repeating')
    assert agent.count(marker) == 1
    agent = agent.replace(marker, hook + marker)
    old = '        recover_arrival(store)\n'
    assert worker.count(old) == 1
    worker = worker.replace(old, old + '        from .roadmodel_plugin import recover as recover_roadmodel\n        recover_roadmodel(cfg)\n')
    old = '                try:arrival_tick(cfg,store)\n'
    assert worker.count(old) == 1
    worker = worker.replace(old, '                try:\n                    from .roadmodel_plugin import tick as roadmodel_tick\n                    roadmodel_tick(cfg)\n                except Exception:print(json.dumps({"component":"roadmodel","event":"tick_failed"}),flush=True)\n' + old)
    old = '    # Card API failures must not starve final results or required user actions.\n'
    assert worker.count(old) == 1
    worker = worker.replace(old, '    if not expired():\n        from .roadmodel_plugin import deliver as deliver_roadmodel\n        deliver_roadmodel(cfg,store)\n' + old)
    plugin = ('"""Fixed local RoadModel plugin; no credentials or additional connection."""\n'
              'import sys\nfrom pathlib import Path\n'
              'ROOT = Path(' + repr(str(ROOT)) + ')\n'
              'if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))\n'
              'from pcm_eval.feishu_entry import handle, recover, tick, deliver\n')
    return {'bench_gateway/cc_agent.py': agent, 'bench_gateway/worker.py': worker,
            'bench_gateway/roadmodel_plugin.py': plugin}


def live_status():
    from bench_gateway.config import load
    from bench_gateway.deployment import status
    cfg=load(TARGET)
    value=status(cfg)
    if cfg.get('deployment_role') == 'ai-client':
        from bench_gateway.transfer import SSH
        code='''import pathlib,sys,json,sqlite3
sys.path.insert(0,sys.argv[1])
from bench_gateway.config import load
cfg=load(sys.argv[1]);base=pathlib.Path(cfg['state_root'])/'roadmodel';out={'runtime_root':str(base)}
db=base/'runtime.sqlite3'
if db.exists():
 with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as c:
  out['tasks']={state:count for state,count in c.execute('SELECT state,count(*) FROM tasks GROUP BY state')}
deployment=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030')
for name in ('report-runtime-r4-deployment.json','core-deployment.json','deployment.json'):
 p=deployment/name
 if p.exists():out['deployment']=json.loads(p.read_text());break
print(json.dumps(out))
'''
        value['roadmodel']=json.loads(SSH(cfg).python(code,(cfg['remote_project_root'],),timeout=30))
        return value
    db=Path(cfg['state_root'])/'roadmodel/runtime.sqlite3'
    if db.exists():
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
            value['roadmodel_tasks']={state:count for state,count in c.execute('SELECT state,count(*) FROM tasks GROUP BY state')}
    return value


def refresh():
    from bench_gateway.config import load,save
    from bench_gateway.core import Store
    from bench_gateway.deployment import start,status,stop
    from tools.deploy_inline_materials import idle
    cfg=load(TARGET)
    if cfg.get('deployment_role') == 'ai-client':
        raise ValueError('Gateway moved to workstation; use migrate_workstation_entry.py, never restart the PC gateway')
    store=Store(cfg['state_root']);idle(store)
    db=Path(cfg['state_root'])/'roadmodel/runtime.sqlite3'
    if db.exists():
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
            if c.execute("SELECT 1 FROM tasks WHERE state IN ('QUEUED','RUNNING')").fetchone():
                raise ValueError('RoadModel tasks must drain before refreshing')
    installed=json.loads((OUT/'deployment.json').read_text(encoding='utf-8'))
    pending=OUT/'catalog.pending.json'
    if pending.exists():
        candidate=json.loads(pending.read_text(encoding='utf-8'))
        if candidate.get('schema')!='roadmodel.two-link-catalog/v1' or len(candidate.get('models',[]))!=1:
            raise ValueError('Pending RoadModel catalog is invalid')
        spec=candidate['models'][0]['chains']['compile_alignment']
        if spec.get('shared_bench_guard_verified'):
            validation=json.loads((ROOT/'evidence/compile-subnet-review-20261029/runtime-validation.json').read_text(encoding='utf-8'))
            if validation.get('state')!='PASSED' or validation.get('new_inference_frames')!=0:
                raise ValueError('Subnet Runtime preflight evidence is missing')
            if hashlib.sha256((ROOT/spec['request_file']).read_bytes()).hexdigest()!=spec['request_sha256']:
                raise ValueError('Subnet request changed after verification')
    assert all(hashlib.sha256((TARGET/rel).read_bytes()).hexdigest()==expected for rel,expected in installed['files'].items())
    old=status(cfg)
    stopped=stop(cfg,wait_seconds=40)
    if stopped.get('active') or stopped['phase']!='STOPPED':raise ValueError('Existing work is still draining')
    idle(store)
    if pending.exists():
        target=ROOT/'configs/two_link_catalog.json'
        backup=ROOT/'evidence/compile-subnet-review-20261029/catalog-before-subnets.json'
        if not backup.exists():shutil.copy2(target,backup)
        shutil.copy2(pending,target)
        pending.rename(pending.with_name('catalog.applied-subnets.json'))
    # Let the native host observe the original supervisor exit before requesting a restart.
    time.sleep(2)
    start(cfg)
    for _ in range(55):
        state=status(cfg)
        if state.get('active') and state['phase']=='RUNNING' and state['instance']!=old['instance']:break
        time.sleep(1)
    else:raise ValueError('Fresh service instance not ready')
    receipt={'state':'REFRESHED','service_instance':state['instance'],'bridge_ready':state['children']['bridge']['ready'],
             'worker_ready':state['children']['worker']['ready'],'backend_sha256':{
                str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'pcm_eval').glob('*.py')},
             'roadmodel_inputs':2,'existing_connection_reused':True,'hardware_dispatched':False}
    save(OUT/'refresh.json',receipt)
    return {k:v for k,v in receipt.items() if k!='backend_sha256'}


def stage():
    files = patches()
    root = OUT / 'staged'
    for name, value in files.items():
        compile(value, name, 'exec')
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value, encoding='utf-8')
    return files


def deploy():
    from bench_gateway.config import load, save
    from bench_gateway.core import Store
    from bench_gateway.deployment import status, stop, start
    from tools.deploy_inline_materials import idle
    # tools is a namespace shared with the target project, import its file explicitly.
    files = stage()
    cfg = load(TARGET)
    if cfg.get('deployment_role') == 'ai-client':
        raise ValueError('Gateway moved to workstation; use migrate_workstation_entry.py')
    store = Store(cfg['state_root'])
    idle(store)
    current = status(cfg)
    if not current['active']: raise ValueError('Existing gateway must be healthy before maintenance')
    backup = TARGET / 'runtime/backups' / time.strftime('roadmodel-entry-%Y%m%d-%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for rel in files:
        src = TARGET / rel
        if src.exists():
            source_hashes[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
            dest = backup / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
    config_hash = hashlib.sha256((TARGET / 'config/gateway.json').read_bytes()).hexdigest()
    stopped = stop(cfg, wait_seconds=40)
    if stopped.get('active') or stopped['phase'] != 'STOPPED': raise ValueError('Gateway is still draining; no files changed')
    idle(store)
    with store.connect() as c:
        with sqlite3.connect(backup / 'gateway.sqlite3') as dest: c.backup(dest)
    receipt = {'backup': str(backup), 'state': 'DEPLOYING', 'hardware_dispatched': False,
               'entry_command': 'RoadModel评测', 'extra_connections': 0}
    try:
        for rel, value in files.items():
            path = TARGET / rel
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[rel]:
                raise ValueError('Gateway source changed during drain')
            path.write_text(value, encoding='utf-8')
        assert hashlib.sha256((TARGET / 'config/gateway.json').read_bytes()).hexdigest() == config_hash
        start(cfg)
        for _ in range(55):
            state = status(cfg)
            if state.get('active') and state['phase'] == 'RUNNING': break
            time.sleep(1)
        else: raise ValueError('Gateway did not become ready after deployment')
        receipt.update(state='DEPLOYED', bridge_ready=state['children']['bridge']['ready'],
                       worker_ready=state['children']['worker']['ready'], config_unchanged=True,
                       service_instance=state['instance'], files={k: hashlib.sha256((TARGET/k).read_bytes()).hexdigest() for k in files})
    except BaseException:
        receipt['state'] = 'FAILED'
        # Preserve the live DB. Restore code only after a normal drain.
        drained = stop(cfg, wait_seconds=40)
        if not drained.get('active') and drained['phase'] == 'STOPPED':
            for rel in files:
                if (backup / rel).exists(): shutil.copy2(backup / rel, TARGET / rel)
            start(cfg)
        raise
    finally:
        save(OUT / 'deployment.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage','status','deploy','refresh'])
    args = p.parse_args()
    if args.action == 'stage': print(json.dumps({'staged': list(stage())}, ensure_ascii=False))
    else: print(json.dumps(live_status() if args.action == 'status' else refresh() if args.action=='refresh' else deploy(), ensure_ascii=False, indent=2))
