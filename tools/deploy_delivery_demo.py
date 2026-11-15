"""Prepare, verify and install the Demo alongside the existing Bot routes."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import load_json, save_json
from pcm_eval.planning import ssh_command

RELEASE = '/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r1'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/delivery-demo-20261103-r1'
PYTHON = '/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'
LOCAL = ROOT/'evidence/delivery-demo-20261103'


def call(code, body=b'', timeout=150):
    command = ssh_command('ubuntu-workstation', [PYTHON, '-c', code])
    command[0] = 'C:/Windows/System32/OpenSSH/ssh.exe'
    p = subprocess.run(command, input=body, capture_output=True, timeout=timeout)
    if p.returncode:
        raise ValueError(p.stderr.decode('utf-8', errors='replace')[-3000:])
    return json.loads(p.stdout)


def prepare():
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'pcm_eval').glob('*.py')]
    names += ['tools/delivery_demo_worker.py', 'tools/delivery_demo_plugin.py',
              'tests/test_delivery_demo.py', 'tests/test_delivery_workflow.py',
              'tests/__init__.py', 'tests/test_two_link_runtime.py',
              'tests/test_model_ticket.py', 'tests/test_publication.py', 'tests/test_strict_model_template.py',
              'configs/delivery_catalog.json', 'configs/report_template.json',
              'evidence/input-contract-review-20261027/release-record.json',
              'evidence/model-evaluation/model-reference-20261027/model-input-binding.json',
              'evidence/model-evaluation/model-reference-20261027/native-gt-preparation.json',
              'evidence/requested-dataset/data_link_26-10-20-1_train859_eval135_v7.json']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'examples').glob('*.json')]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name in sorted(set(names)):
            tar.add(ROOT/name, arcname=name)
    code = r'''import pathlib,sys,json,io,tarfile,shutil,subprocess,uuid,hashlib,os,ast
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Already installed; do not replace a running release'
stage.mkdir(parents=True,exist_ok=True);release.mkdir(parents=True,exist_ok=True)
old=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/model-tickets-20261103-r2')
for name in ['pcm_eval','configs','schemas','active']:
 if (old/name).is_dir() and not (release/name).exists():shutil.copytree(old/name,release/name)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile();p=(release/m.name).resolve();p.relative_to(release)
  raw=tar.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.py':ast.parse(raw,filename=m.name)
  p.write_bytes(raw)
oldcfg=json.loads((old/'configs/model_ticket_catalog.json').read_text());runtime=oldcfg['runtime_config'];owner=next(iter(oldcfg['delivery_by_owner']))
profile=runtime['space_preparation']['by_owner'][owner]
cfg={'owner':owner,'agent_id':uuid.uuid4().hex,'python':sys.executable,'bench_available':False,
     'model_space':{'host':profile['ssh_host'],'port':profile['ssh_port'],'identity_file':profile['identity_file'],'user':'root'},
     'reference_file':runtime['reference_file'],'lark_cli':runtime['lark_cli'],'lark_config':runtime['lark_config']}
(release/'configs/delivery_demo.json').write_text(json.dumps(cfg,indent=2))
sys.path.insert(0,str(release))
from pcm_eval.delivery_demo import DemoRuntime
from pcm_eval.delivery_demo_entry import render
rt=DemoRuntime(stage/'preview-state');card=render(rt.open(owner,'feishu:oc_preview:'+owner))
(stage/'card-preview.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
tests=[]
for pattern in ['test_delivery*.py','test_model_ticket.py','test_publication.py','test_strict_model_template.py']:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=release,capture_output=True,text=True,timeout=90)
 (stage/(pattern.replace('*','all')+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,p.stderr[-2500:];tests.append(pattern)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'state':'PREPARED','release':str(release),'tests':tests,'new_inference_frames':0,
 'files':{str(p.relative_to(release)):sha(p) for name in ['pcm_eval','tools','configs'] for p in (release/name).glob('*') if p.is_file()},
 'agent':{'release':str(release),'state_root':'/mnt/data/bench-eval/state/roadmodel/delivery-demo',
          'owner':owner,'agent_id':cfg['agent_id'],'python':sys.executable}}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code, stream.getvalue())
    LOCAL.mkdir(parents=True, exist_ok=True)
    save_json(LOCAL/'preparation.json', result)
    save_json(LOCAL/'agent-config.json', result['agent'])
    print(json.dumps({'state': result['state'], 'release': RELEASE, 'tests': result['tests']}))


def deploy():
    code = r'''import pathlib,sys,json,sqlite3,shutil,hashlib,os,time
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);base=pathlib.Path('/mnt/data/bench-eval/service/gateway')
assert not (stage/'deployment.json').exists(),'Demo already deployed'
prep=json.loads((stage/'preparation.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(release/n)==h for n,h in prep['files'].items())
sys.path[:0]=[str(base),str(release)]
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
cfg=load(base);store=Store(cfg['state_root']);before=status(cfg);assert before['active']
with store.connect() as c:
 for table in ['jobs','arrival_analyses','archive_jobs']:
  if c.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():
   assert not c.execute('select 1 from '+table+" where state in ('QUEUED','RUNNING')").fetchone(),'Shared work must drain first'
ticket=pathlib.Path(cfg['state_root'])/'roadmodel/model-tickets/tickets.sqlite3'
if ticket.exists():
 with sqlite3.connect(ticket) as c:assert not c.execute("select 1 from tickets where state in ('CHECKING','PREPARING','QUEUED','RUNNING')").fetchone(),'Existing work order active'
backup=stage/'before';backup.mkdir(exist_ok=False)
plugin=base/'bench_gateway/roadmodel_plugin.py';shutil.copy2(plugin,backup/'roadmodel_plugin.py')
with store.connect() as src:
 with sqlite3.connect(backup/'gateway.sqlite3') as dst:src.backup(dst)
stopped=stop(cfg,wait_seconds=40);assert stopped['phase']=='STOPPED' and not stopped.get('active')
try:
 body=(release/'tools/delivery_demo_plugin.py').read_bytes();compile(body,str(plugin),'exec')
 tmp=plugin.with_suffix('.demo.tmp');tmp.write_bytes(body);os.replace(tmp,plugin)
 start(cfg)
 for _ in range(40):
  current=status(cfg)
  if current.get('active') and current.get('phase')=='RUNNING' and all(x.get('ready') for x in current['children'].values()):break
  time.sleep(1)
 else:raise ValueError('Bot readiness not confirmed')
except Exception:
 if not status(cfg).get('active'):shutil.copy2(backup/'roadmodel_plugin.py',plugin);start(cfg)
 raise
receipt={'state':'DEPLOYED','release':str(release),'service_instance':current['instance'],
 'plugin_sha256':sha(plugin),'single_existing_connection':True,'new_inference_frames':0,
 'scope':{'model_occ_frames':2,'compile_tensor_frames':1},'bench_available':False,
 'user_callback_and_new_inference_verified':False}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'deployment.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'deploy'])
    args = p.parse_args()
    prepare() if args.action == 'prepare' else deploy()
