"""Deploy UID-pinned official lifecycle without changing browser permissions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from deploy_environment_flow import remote
from deploy_model_core_20261107 import schedule
from verify_model_core_20261107 import MODULES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5'
RELEASE = '/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/space-api-20261107-r6'
FILES = ['pcm_eval/lpai.py','pcm_eval/space_api.py','pcm_eval/space_lifecycle.py','pcm_eval/evaluation.py',
         'tools/evaluation_agent.py','tools/evaluation_space_actor.py']

PREPARE = '''import pathlib,json,shutil,sys,hashlib,unittest
source=pathlib.Path(SOURCE);release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);stage.mkdir(parents=True,exist_ok=True)
candidate=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/space-api-20261107/candidate');proof=json.loads((candidate.parent/'verification.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert proof['state']=='VERIFIED'
assert all(sha(candidate/p)==proof['source_files'][p] for p in FILES)
if not release.exists():shutil.copytree(source,release,ignore=shutil.ignore_patterns('__pycache__'))
for name in FILES:shutil.copy2(candidate/name,release/name)
for p in (candidate/'tests').glob('*.py'):shutil.copy2(p,release/'tests'/p.name)
old=json.loads((source/'configs/evaluation.json').read_text());cfg=json.loads(json.dumps(old))
cfg['space_backend']='official_cli';cfg['profile']['space_uid']=UID;cfg['profile']['lifecycle_provider']='official_cli'
assert cfg['execution_scope']=='validation' and cfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}
(release/'configs/evaluation.json').write_text(json.dumps(cfg,indent=2))
sys.path.insert(0,str(release));suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
with (stage/'tests.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
receipt={'state':'VERIFIED' if result.wasSuccessful() else 'FAILED','tests':result.testsRun,'files':{p:sha(release/p) for p in FILES},
 'config_sha256':sha(release/'configs/evaluation.json'),'space_uid':UID,'browser_permissions_changed':False,
 'source_release':SOURCE,'release':RELEASE,'new_inference_frames':0,
 'failed_cases':[{'test':str(c),'trace':t[-1800:]} for c,t in result.failures+result.errors]}
(stage/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''

DEPLOY = '''import pathlib,json,sys,hashlib,sqlite3,os,time,signal,shutil
source=pathlib.Path(SOURCE);release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);proof=json.loads((stage/'verification.json').read_text());assert proof['state']=='VERIFIED'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(release/p)==h for p,h in proof['files'].items()) and sha(release/'configs/evaluation.json')==proof['config_sha256']
assert not (stage/'deployment.json').exists(),'Already deployed'
base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path.insert(0,str(base))
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status
cfg=load(base);before=status(cfg);assert before['active'] and before['children']['worker']['ready'] and before['children']['bridge']['ready']
with Store(cfg['state_root']).connect() as db:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if db.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not db.execute("select 1 from "+table+" where state in ('QUEUED','RUNNING','STARTING','PREPARING')").fetchone(),'Shared work active'
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3') as db:
 assert not db.execute("select 1 from tasks where state in ('QUEUED','READY','RUNNING','PREPARING','WAITING_REPORT')").fetchone(),'Evaluation is active'
 assert not db.execute("select 1 from materials where state='READING'").fetchone()
 assert not db.execute("select 1 from publications where state='WRITING'").fetchone()
 with sqlite3.connect(stage/'before-runtime.sqlite3') as target:db.backup(target)
 # Discard only the cached page observation; preserve all work orders/intents.
 db.execute('delete from space_observations where name=?',('roadmodel-eval-v322-20261106',))
plugin=base/'bench_gateway/roadmodel_plugin.py';old=plugin.read_bytes();assert sha(plugin)=='aea0c29c42e9d95d7a9ef1efe2497761105beb1910b1cd32dce0834aef43d64e'
line=("ROOT = Path('"+SOURCE+"')").encode();assert old.count(line)==1
new=old.replace(line,("ROOT = Path('"+RELEASE+"')").encode());compile(new,str(plugin),'exec');shutil.copy2(plugin,stage/'before-plugin.py')
tmp=plugin.with_suffix('.space-api.tmp');tmp.write_bytes(new);os.replace(tmp,plugin)
worker=before['children']['worker']['pid'];bridge=before['children']['bridge']['pid']
try:
 assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker),'cmdline').read_bytes();os.kill(worker,signal.SIGTERM)
 for _ in range(35):
  time.sleep(1);after=status(cfg);assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge
  if after['children']['worker']['pid']!=worker and after['children']['worker']['ready']:break
 else:raise ValueError('Worker not ready')
except Exception:
 tmp.write_bytes(old);os.replace(tmp,plugin);raise
result={'state':'DEPLOYED','release':RELEASE,'space_backend':'official_cli','space_uid':UID,'tests':proof['tests'],
 'bridge_pid_unchanged':True,'worker_ready':after['children']['worker']['ready'],'browser_permissions_changed':False,
 'existing_tasks_resumed':False,'new_inference_frames':0,'plugin_sha256':sha(plugin)}
(stage/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''


def main(action):
    uid = json.loads((ROOT/'evidence/sdk-space-20261107/official-read.json').read_text())['metadata']['uid']
    prefix = 'SOURCE='+repr(SOURCE)+'\nRELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\nFILES='+repr(FILES)+'\nUID='+repr(uid)+'\nMODULES='+repr(MODULES+['tests.test_space_api'])+'\n'
    if action == 'prepare':
        result = remote(prefix+PREPARE, timeout=150)
        (ROOT/'evidence/sdk-space-20261107/release-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result));return
    proof = json.loads((ROOT/'evidence/sdk-space-20261107/release-verification.json').read_text())
    assert proof['state'] == 'VERIFIED'
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == proof['files'][name] for name in FILES)
    pinned = ROOT/'.runtime/evaluation-environment-desktop-20261106-r3';backup=ROOT/'.runtime/evaluation-desktop-before-space-api-r6'
    assert not backup.exists()
    schedule('Stop')
    try:
        shutil.copytree(pinned,backup,ignore=shutil.ignore_patterns('__pycache__'))
        result = remote(prefix+DEPLOY, timeout=65)
        for name in FILES:shutil.copy2(ROOT/name,pinned/name)
        p=pinned/'configs/agent.json';cfg=json.loads(p.read_text(encoding='utf-8'))
        cfg.update(release=RELEASE,space_backend='official_cli');cfg['profile'].update(space_uid=uid,lifecycle_provider='official_cli')
        p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        result['desktop_release_binding']=RELEASE
        (ROOT/'evidence/sdk-space-20261107/deployment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
    finally:schedule('Start')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','deploy'));main(parser.parse_args().action)
