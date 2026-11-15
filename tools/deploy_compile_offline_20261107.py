"""Activate verified code with bench execution disabled; preserve the shared bridge."""
import json
from pathlib import Path
import shutil
import subprocess

from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
TASK='RoadModelEvaluationAgent-20261106-r3'
CANDIDATE='/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4'
STAGE='/mnt/data/bench-eval/roadmodel/deployment/compile-offline-20261107-r4'

DEPLOY='''import pathlib,sys,json,hashlib,sqlite3,os,time,signal,shutil
candidate=pathlib.Path(CANDIDATE);stage=pathlib.Path(DEPLOY_STAGE)
proof=json.loads((stage/'verification.json').read_text());assert proof['state']=='OFFLINE_VERIFIED'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(candidate/name)==h for name,h in proof['source_files'].items()),'Code changed after verification'
evalcfg=json.loads((candidate/'configs/evaluation.json').read_text())
assert evalcfg['bench_available'] is False and evalcfg['execution_scope']=='validation'
assert evalcfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}
base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path.insert(0,str(base))
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status
cfg=load(base);before=status(cfg);assert before['active'] and before['children']['bridge']['ready'] and before['children']['worker']['ready']
with Store(cfg['state_root']).connect() as db:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if db.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not db.execute("select 1 from "+table+" where state in ('QUEUED','RUNNING','STARTING','PREPARING')").fetchone(),'Shared work is active'
state=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3')
with sqlite3.connect(state) as db:
 assert not db.execute("select 1 from tasks where state in ('QUEUED','READY','RUNNING','PREPARING','WAITING_REPORT')").fetchone(),'RoadModel work is active'
 assert not db.execute("select 1 from materials where state='READING'").fetchone()
 assert not db.execute("select 1 from publications where state='WRITING'").fetchone()
 with sqlite3.connect(stage/'before-deployment-runtime.sqlite3') as target:db.backup(target)
plugin=base/'bench_gateway/roadmodel_plugin.py'
assert sha(plugin)=='b8b87e2c14bdf95aac7493d7dcd1b4c7c53441ffd5673411bd2485a42f90d8f5','Review concurrent routing changes first'
assert not (stage/'deployment.json').exists(),'Already deployed'
old_line=("ROOT = Path('"+NEW+"')").encode();raw=plugin.read_bytes();assert raw.count(old_line)==1
shutil.copy2(plugin,stage/'before-plugin.py')
new=raw.replace(old_line,("ROOT = Path('"+CANDIDATE+"')").encode());compile(new,str(plugin),'exec')
tmp=plugin.with_suffix('.r4.tmp');tmp.write_bytes(new);os.replace(tmp,plugin)
worker=before['children']['worker']['pid'];bridge=before['children']['bridge']['pid']
assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker),'cmdline').read_bytes()
os.kill(worker,signal.SIGTERM)
for _ in range(30):
 time.sleep(1);after=status(cfg)
 assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge,'Shared bridge changed'
 if after['children']['worker']['pid']!=worker and after['children']['worker']['ready']:break
else:raise ValueError('Worker did not become ready')
result={'state':'DEPLOYED_BENCH_DISABLED','release':CANDIDATE,'bench_available':False,'execution_scope':'validation',
        'validation_selection':evalcfg['validation_selection'],'worker_reloaded':True,'bridge_pid_unchanged':True,
        'plugin_sha256':sha(plugin),'native_inference_frames':0,'old_tasks_resumed':False}
(stage/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''


def schedule(action):
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
                    action+'-ScheduledTask -TaskName '+repr(TASK)],check=True,capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':
    # The dedicated source/publisher process must not hold a queue claim while
    # its release binding changes. The shared Feishu connection is not stopped.
    schedule('Stop')
    pinned=ROOT/'.runtime/evaluation-environment-desktop-20261106-r3'
    backup=ROOT/'.runtime/evaluation-desktop-before-compile-r4'
    try:
        assert not backup.exists(),'Preserve the previous deployment backup'
        shutil.copytree(pinned,backup,ignore=shutil.ignore_patterns('__pycache__'))
        result=remote('CANDIDATE='+repr(CANDIDATE)+'\nDEPLOY_STAGE='+repr(STAGE)+'\n'+DEPLOY,timeout=65)
        for path in (ROOT/'pcm_eval').glob('*.py'):shutil.copy2(path,pinned/'pcm_eval'/path.name)
        shutil.copy2(ROOT/'tools/evaluation_agent.py',pinned/'tools/evaluation_agent.py')
        config=pinned/'configs/agent.json';cfg=json.loads(config.read_text(encoding='utf-8'));cfg['release']=CANDIDATE
        config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        result['desktop_release_binding']=CANDIDATE
        destination=ROOT/'evidence/compile-offline-20261107/deployment.json'
        destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
    finally:schedule('Start')
