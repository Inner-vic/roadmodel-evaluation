"""Install tested, paired current-account dedicated-space Runtime configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from deploy_environment_flow import remote
from deploy_model_core_20261107 import schedule
from pcm_eval.installation import export_agent_config

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/rebuild-initialization-20261109'
SOURCE='/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6'
RELEASE='/mnt/data/bench-eval/service/roadmodel/evaluation-initialized-20261109-r7'
STAGE='/mnt/data/bench-eval/roadmodel/deployment/initialized-20261109-r7'
PINNED=ROOT/'.runtime/evaluation-environment-desktop-20261106-r3'
TOOLS=['evaluation_agent.py','evaluation_space_actor.py','notebook_sdk_worker.py','initialize_evaluation_environment.py']
BACKUP=ROOT/'.runtime/evaluation-desktop-before-initialized-r7'
SELECTION={'scene_count':1,'frames_per_scene':2}

PREPARE='''import pathlib,json,sys,shutil,hashlib
binding=json.load(sys.stdin)
source=pathlib.Path(SOURCE);release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);stage.mkdir(parents=True,exist_ok=True)
candidate=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/initialization-20261109/candidate')
proof=json.loads((candidate.parent/'verification.json').read_text());assert proof['state']=='VERIFIED'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(candidate/n)==h for n,h in proof['files'].items())
assert not (stage/'deployment.json').exists(),'Already deployed'
if not release.exists():shutil.copytree(source,release,ignore=shutil.ignore_patterns('__pycache__'))
for n in ('pcm_eval','tests'):
 for p in (candidate/n).glob('*.py'):shutil.copy2(p,release/n/p.name)
for n in TOOLS:shutil.copy2(candidate/'tools'/n,release/'tools'/n)
sys.path.insert(0,str(release))
from pcm_eval.installation import export_server_config
cfg=json.loads((source/'configs/evaluation.json').read_text());cfg['feishu_app_id']=binding['request']['account']['feishu_app_id']
assert cfg['owner']==binding['request']['account']['feishu_user_id']
cfg=export_server_config(binding,cfg,stage/'server-config.json')
cfg['validation_selection']=SELECTION
assert cfg['execution_scope']=='validation' and cfg['validation_selection']['scene_count']==1 and cfg['validation_selection']['frames_per_scene']==2
(release/'configs/evaluation.json').write_text(json.dumps(cfg,indent=2))
files={str(p.relative_to(release)):sha(p) for folder in ('pcm_eval','tools','configs') for p in (release/folder).glob('*.py' if folder!='configs' else '*.json')}
receipt={'state':'PREPARED','release':RELEASE,'space_uid':cfg['profile']['space_uid'],'space_name':cfg['profile']['space_name'],
 'tests':proof['tests'],'files':files,'config_sha256':sha(release/'configs/evaluation.json'),'new_inference_frames':0}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
'''

DEPLOY='''import pathlib,json,sys,sqlite3,shutil,os,time,signal,hashlib
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);proof=json.loads((stage/'preparation.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert proof['state']=='PREPARED' and all(sha(release/n)==h for n,h in proof['files'].items())
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
 assert not db.execute("select 1 from tasks where state in ('QUEUED','READY','RUNNING','PREPARING','WAITING_REPORT')").fetchone(),'Evaluation active'
 assert not db.execute("select 1 from materials where state='READING'").fetchone()
 assert not db.execute("select 1 from publications where state='WRITING'").fetchone()
 with sqlite3.connect(stage/'before-runtime.sqlite3') as backup:db.backup(backup)
plugin=base/'bench_gateway/roadmodel_plugin.py';old=plugin.read_bytes();line=("ROOT = Path('"+SOURCE+"')").encode()
assert old.count(line)==1,'Concurrent deployment changed'
shutil.copy2(plugin,stage/'before-plugin.py');new=old.replace(line,("ROOT = Path('"+RELEASE+"')").encode());compile(new,str(plugin),'exec')
temp=plugin.with_suffix('.initialized.tmp');temp.write_bytes(new);os.replace(temp,plugin)
worker=before['children']['worker']['pid'];bridge=before['children']['bridge']['pid']
try:
 assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker),'cmdline').read_bytes();os.kill(worker,signal.SIGTERM)
 for _ in range(35):
  time.sleep(1);after=status(cfg);assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge
  if after['children']['worker']['pid']!=worker and after['children']['worker']['ready']:break
 else:raise ValueError('Worker not ready')
except Exception:
 temp.write_bytes(old);os.replace(temp,plugin);raise
result={'state':'DEPLOYED','release':RELEASE,'space_uid':proof['space_uid'],'space_name':proof['space_name'],'tests':proof['tests'],
 'bridge_pid_unchanged':True,'worker_ready':after['children']['worker']['ready'],'new_inference_frames':0,'old_results_preserved':True}
(stage/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''

def main(action):
    binding=json.loads((OUT/'plan/initialization.json').read_text(encoding='utf-8'))
    assert binding['state']=='BOUND'
    prefix='SOURCE='+repr(SOURCE)+'\nRELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\nTOOLS='+repr(TOOLS)+'\nSELECTION='+repr(SELECTION)+'\n'
    if action=='prepare':
        base=json.loads((PINNED/'configs/agent.json').read_text(encoding='utf-8'))
        base.update(feishu_app_id=binding['request']['account']['feishu_app_id'],release=RELEASE,auth_probe_on_start=False)
        export_agent_config(binding,base,OUT/'agent-config.json',local_state=ROOT/'.runtime/evaluation-initialized-agent')
        value=remote(prefix+PREPARE,json.dumps(binding).encode(),timeout=50)
        (OUT/'release-preparation.json').write_text(json.dumps(value,indent=2));print(json.dumps(value));return
    backup=BACKUP
    if backup.exists():
        assert not (OUT/'deployment.json').exists(),'Already deployed'
        assert (backup/'configs/agent.json').read_bytes()==(PINNED/'configs/agent.json').read_bytes(),'Deployment outcome changed; reconcile first'
        for folder in ('pcm_eval','tools'):
            assert all((PINNED/folder/p.name).read_bytes()==p.read_bytes() for p in (backup/folder).glob('*.py')),'Desktop changed since the failed preflight'
    schedule('Stop')
    try:
        if not backup.exists():shutil.copytree(PINNED,backup,ignore=shutil.ignore_patterns('__pycache__'))
        result=remote(prefix+DEPLOY,timeout=65)
        for name in ('pcm_eval','tests'):
            for p in (ROOT/name).glob('*.py'):shutil.copy2(p,PINNED/name/p.name) if (PINNED/name).exists() else None
        for name in TOOLS:shutil.copy2(ROOT/'tools'/name,PINNED/'tools'/name)
        shutil.copy2(OUT/'agent-config.json',PINNED/'configs/agent.json')
        (OUT/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:schedule('Start')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','deploy'));main(p.parse_args().action)
