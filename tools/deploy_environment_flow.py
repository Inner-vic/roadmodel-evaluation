"""Stage a fresh release; preserve shared Bot routing and hardware admission."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from connect_active_entry import call
from pcm_eval.common import load_json,save_json

OLD='/mnt/data/bench-eval/service/roadmodel/evaluation-native-20261105-r2'
NEW='/mnt/data/bench-eval/service/roadmodel/evaluation-environment-20261106-r3'
STAGE='/mnt/data/bench-eval/roadmodel/deployment/evaluation-environment-20261106-r3'
ENVROOT='/mnt/data/bench-eval/roadmodel/deployment/environment-init-20261106'
LOCAL=ROOT/'evidence/environment-init-20261106'


def remote(code,payload=b'',timeout=120):
    return json.loads(call('OLD='+repr(OLD)+'\nNEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\nENVROOT='+repr(ENVROOT)+'\n'+code,payload,timeout=timeout))


def prepare():
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        files=[p.relative_to(ROOT).as_posix() for p in (ROOT/'pcm_eval').glob('*.py')]
        files+=['tools/evaluation_agent.py','tools/evaluation_space_actor.py','tools/evaluation_worker.py','tools/delivery_demo_agent.py','tools/model_space_desktop.py','tests/test_environment_flow.py','schemas/space-preparation-result.schema.json','skills/roadmodel-space-preparation/SKILL.md']
        for name in files:tar.add(ROOT/name,arcname=name)
        raw=json.dumps(load_json(LOCAL/'new-profile.json')).encode();i=tarfile.TarInfo('configs/environment-profile.json');i.size=len(raw);tar.addfile(i,io.BytesIO(raw))
    r=remote('''import pathlib,sys,json,shutil,tarfile,io,ast,subprocess,re,hashlib
old=pathlib.Path(OLD);new=pathlib.Path(NEW);stage=pathlib.Path(STAGE);stage.mkdir(exist_ok=True)
assert not (stage/'deployment.json').exists(),'Release already deployed'
if not new.exists():shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();p=(new/member.name).resolve();p.relative_to(new);raw=tar.extractfile(member).read()
  if p.suffix=='.py':ast.parse(raw)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
cfg=json.loads((old/'configs/evaluation.json').read_text());cfg['profile']=json.loads((new/'configs/environment-profile.json').read_text())
cfg['model_space']={'host':cfg['profile']['ssh_host'],'port':cfg['profile']['ssh_port'],'user':'root','identity_file':cfg['profile']['identity_file']}
bundle=json.loads((pathlib.Path(ENVROOT)/'model-environment/bundle.json').read_text());assert bundle['dataset_files_included'] is False
cfg['native_profile']['execution']['environment_bundle']=bundle
cfg.update(execution_scope='validation',validation_selection={'scene_count':1,'frames_per_scene':2},bench_available=False)
(new/'configs/evaluation.json').write_text(json.dumps(cfg,indent=2))
sys.path.insert(0,str(new));from pcm_eval.lpai_environment import ensure
ready=ensure(bundle,cfg['model_space']);(stage/'environment-ready.json').write_text(json.dumps(ready,indent=2))
tests=[]
for module in ('tests.test_environment_flow','tests.test_lpai_native_model','tests.test_evaluation','tests.test_delivery_workflow','tests.test_delivery_combined','tests.test_publication','tests.test_strict_model_template'):
 p=subprocess.run([sys.executable,'-m','unittest',module,'-q'],cwd=new,capture_output=True,text=True,timeout=60);out=p.stdout+p.stderr
 (stage/(module+'.log')).write_text(out);assert p.returncode==0,out[-2500:];tests.append({'module':module,'count':int(re.search(r'Ran (\\d+) tests?',out)[1])})
receipt={'state':'PREPARED','release':NEW,'tests':tests,'test_count':sum(t['count'] for t in tests),'environment':ready,'bench_contacted':False,'files':{str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('pcm_eval','tools','configs','schemas','skills') for p in (new/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2))
agent={'release':NEW,'state_root':'/mnt/data/bench-eval/state/roadmodel/evaluation','owner':cfg['owner'],'agent_id':cfg['agent_id'],'python':sys.executable,'profile':cfg['profile'],'native_profile':cfg['native_profile'],'compile_profile':cfg['compile_profile'],'rpc_timeout':110,'queue_kind':'evaluation','paused':False,'browser_backend_verified':False}
print(json.dumps({'receipt':{k:v for k,v in receipt.items() if k!='files'},'agent':agent}))
''',buf.getvalue(),timeout=360)
    save_json(LOCAL/'r3-preparation.json',r['receipt'],overwrite=(LOCAL/'r3-preparation.json').exists())
    cfg={**r['agent'],'local_state':str(ROOT/'.runtime/evaluation-flow-agent'),'lpai_cli':'C:/Users/developer/AppData/Local/bin/lpai-agent.exe'}
    save_json(LOCAL/'r3-agent-config.json',cfg,overwrite=(LOCAL/'r3-agent-config.json').exists())
    print(json.dumps(r['receipt']))


def deploy():
    r=remote('''import pathlib,json,hashlib,sqlite3,shutil,os,sys,time,signal
new=pathlib.Path(NEW);stage=pathlib.Path(STAGE);assert not (stage/'deployment.json').exists(),'Already deployed'
prepared=json.loads((stage/'preparation.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(new/name)==h for name,h in prepared['files'].items()),'Candidate changed after tests'
base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path.insert(0,str(base))
from bench_gateway.config import load
from bench_gateway.deployment import status
from bench_gateway.core import Store
cfg=load(base);before=status(cfg);assert before['active'] and before['phase']=='RUNNING'
with Store(cfg['state_root']).connect() as db:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if db.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not db.execute('select 1 from '+table+" where state in ('QUEUED','RUNNING','STARTING','PREPARING')").fetchone(),'Shared queue must drain first'
plugin=base/'bench_gateway/roadmodel_plugin.py';assert sha(plugin)=='9ed06dacacfdecf48affd02f299068c50f6c030c139198fcbe17fe5e7adb4d49','Inspect concurrent routing changes first'
body=plugin.read_bytes();prior=("ROOT = Path('"+OLD+"')").encode();assert body.count(prior)==1
body=body.replace(prior,("ROOT = Path('"+NEW+"')").encode());compile(body,str(plugin),'exec')
database=pathlib.Path(cfg['state_root'])/'roadmodel/evaluation/runtime.sqlite3'
with sqlite3.connect(database) as source:
 assert not source.execute("select 1 from tasks where state in ('QUEUED','PREPARING','READY','RUNNING','WAITING_REPORT')").fetchone(),'RoadModel active work must drain'
 with sqlite3.connect(stage/'before-runtime.sqlite3') as target:source.backup(target)
shutil.copy2(plugin,stage/'before-plugin.py');tmp=plugin.with_suffix('.r3.tmp');tmp.write_bytes(body);os.replace(tmp,plugin)
worker=before['children']['worker'];bridge=before['children']['bridge'];assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker['pid']),'cmdline').read_bytes()
os.kill(worker['pid'],signal.SIGTERM)
for _ in range(30):
 time.sleep(1);after=status(cfg);assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge['pid'],'Shared bridge must remain unchanged'
 w=after['children']['worker']
 if w['pid']!=worker['pid'] and w['ready']:break
else:raise ValueError('Worker reload needs review')
receipt={'state':'DEPLOYED','release':NEW,'plugin_sha256':sha(plugin),'shared_bridge_pid_unchanged':True,'worker_reloaded':True,'bench_available':False,'execution_scope':'validation','new_inference_frames':0,'real_card_callback_verified':False}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
''',timeout=50)
    save_json(LOCAL/'r3-deployment.json',r)
    print(json.dumps(r))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',choices=('prepare','deploy'),default='prepare');args=parser.parse_args()
    prepare() if args.action=='prepare' else deploy()
