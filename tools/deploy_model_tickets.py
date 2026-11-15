"""Stage/test and deploy the model-only work-order entry; never run an evaluation."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from connect_active_entry import call

ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/model-tickets-20261103-r1'
STAGE='/mnt/data/bench-eval/roadmodel/deployment/model-tickets-20261103-r1'
FILES=['pcm_eval/model_ticket.py','pcm_eval/model_ticket_entry.py','pcm_eval/strict_model_template.py',
       'tools/model_ticket_worker.py','tools/model_ticket_plugin.py','tests/test_model_ticket.py','tests/test_strict_model_template.py']


def prepare():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as t:
        for name in FILES:t.add(ROOT/name,arcname=name)
    raw=stream.getvalue();signature=hashlib.sha256(raw).hexdigest()
    folder=ROOT/'inspection/model-tickets-20261102-r1';folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'code.tar.gz';archive.write_bytes(raw)
    remote='/mnt/data/bench-eval/roadmodel/deployment/model-tickets-code-'+signature+'.tar.gz'
    batch=folder/'upload.sftp';batch.write_text('put "'+archive.as_posix()+'" "'+remote+'"\n',encoding='utf8')
    p=subprocess.run([shutil.which('sftp') or 'sftp','-o','BatchMode=yes','-o','ConnectTimeout=15','-b',str(batch),'ubuntu-workstation'],capture_output=True,timeout=100)
    if p.returncode:raise ValueError(p.stderr.decode(errors='replace')[-1000:])
    code=r'''import pathlib,json,hashlib,tarfile,shutil,sys,subprocess,os,ast
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);old=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=pathlib.Path(ARCHIVE);assert sha(archive)==SIGNATURE
assert not (stage/'deployment.json').exists(),'Already deployed; do not replace the running release'
stage.mkdir(parents=True,exist_ok=True)
if not release.exists():
 release.mkdir()
 for name in ['pcm_eval','configs','schemas','active']:
  if (old/name).is_dir():shutil.copytree(old/name,release/name)
with tarfile.open(archive) as tar:
 for m in tar:
  assert m.isfile();p=(release/m.name).resolve();p.relative_to(release);p.parent.mkdir(parents=True,exist_ok=True)
  body=tar.extractfile(m).read();ast.parse(body,filename=m.name);p.write_bytes(body)
base=pathlib.Path('/mnt/data/bench-eval');done=base/'roadmodel/formal/model-timing-v322-20261102-r1';cfg=json.loads((done/'run-config.json').read_text());req=json.loads((done/'request.json').read_text())
registry=stage/'registry';registry.mkdir(exist_ok=True)
main=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1');index=main/'data_link_26-10-20-1_train859_eval135_v7.json';assert sha(index)=='e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c'
sys.path.insert(0,str(release));from pcm_eval.model_ticket import EXAMPLE_INPUTS as DEFAULTS,resolve
catalog={'models':[{'url':DEFAULTS['model_delivery_url'],'version':'v3.2.2','evaluation_heads':['occ','vision']}],
 'datasets':[{'id':'occ-26-10-20-1','version':'26-10-20-1','index_path':str(index),'index_sha256':sha(index),
   'aliases':[DEFAULTS['dataset_ref'],'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/components/detail/project-bf8f6b06-f0e7-4d72-aa58-0c2731157df1/record/81e922d6-ee86-483c-9366-d5e3f6aca940',str(main),'工作站:'+str(main)]},
  ],
 'delivery_by_owner':{'ou_00000000000000000000000000000001':{'document_id':'DEMO00000000000000000000002'}},
 'python':sys.executable,'worker':str(release/'tools/model_ticket_worker.py'),
 'runtime_config':{'native_release':'/mnt/data/bench-eval/service/roadmodel/model-timing-v322-20261102-r1-setup-fix2',
   'space_parent':'/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027','old_space':'/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029',
   'timed_space':cfg['space_root'],'identity_file':cfg['identity_file'],'gpu_selection':json.loads((done/'model/gpu-selection.json').read_text()),
   'vision_cache_receipt':str(done/'report/verification.json'),'baseline_model_report':cfg['baseline_model_report'],
   'reference_file':'/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/reference.json',
   'lark_cli':'/mnt/data/bench-eval/service/roadmodel/lark-cli/node_modules/@larksuite/cli/bin/lark-cli',
   'lark_config':'/mnt/data/bench-eval/service/gateway/runtime/roadmodel/lark-cli-config',
   'native_dataset_bindings':{'occ':'occ-26-10-20-1','vision':'vision-26-08-14-0'}}}
(release/'configs/model_ticket_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
assert not resolve(DEFAULTS,catalog)['ready'],'No silent visual fallback'
r=resolve(DEFAULTS,catalog);assert r['single_dataset'] and r['scope']=='full' and r['heads']['occ']['frame_count']==8337 and r['heads']['vision']['frame_count']==0
for name,total in [('formal_model_controller.py',len(r['heads']['occ']['frames'])),('formal_vision_controller.py',len(r['heads']['vision']['frames']))]:
 s=(pathlib.Path(catalog['runtime_config']['native_release'])/'tools'/name).read_text();assert "SPACE = '"+cfg['space_root']+"'" in s
 s=s.replace("SPACE = '"+cfg['space_root']+"'","SPACE = '/private-test'").replace('8337' if 'model_controller' in name else '11543',str(total));ast.parse(s)
env={**os.environ,'PYTHONPATH':str(release)}
tests=[]
for pattern in ['test_model_ticket.py','test_strict_model_template.py']:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=release,env=env,capture_output=True,text=True,timeout=60)
 (stage/(pattern+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-2500:];tests.append(pattern)
receipt={'state':'PREPARED','new_inference_frames':0,'tests':tests,'default_missing_visual_gt_blocks':True,
         'single_dataset_only':True,'all_configured_items':True,'scope':'full','source_scope':{k:h['frame_count'] for k,h in r['heads'].items()},
         'files':{str(p.relative_to(release)):sha(p) for folder in ['pcm_eval','tools','tests'] for p in (release/folder).glob('*.py')},
         'catalog_sha256':sha(release/'configs/model_ticket_catalog.json')}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
'''
    print(call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\nARCHIVE='+repr(remote)+'\nSIGNATURE='+repr(signature)+'\n'+code,timeout=200))


def deploy():
    code=r'''import pathlib,json,hashlib,shutil,sqlite3,sys,os,time
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);base=pathlib.Path('/mnt/data/bench-eval/service/gateway')
assert not (stage/'deployment.json').exists(),'Already deployed; inspect existing receipt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prepared=json.loads((stage/'preparation.json').read_text());assert prepared['state']=='PREPARED'
assert all(sha(release/n)==h for n,h in prepared['files'].items());assert sha(release/'configs/model_ticket_catalog.json')==prepared['catalog_sha256']
sys.path[:0]=[str(base),str(release)];from bench_gateway.config import load;from bench_gateway.core import Store;from bench_gateway.deployment import status,stop,start
cfg=load(base);store=Store(cfg['state_root']);before=status(cfg);assert before['active']
with store.connect() as c:
 for t in ['jobs','arrival_analyses','archive_jobs']:
  if c.execute('select 1 from sqlite_master where name=?',(t,)).fetchone():assert not c.execute('select 1 from '+t+" where state in ('QUEUED','RUNNING')").fetchone(),'Wait for shared work to drain'
with sqlite3.connect(pathlib.Path(cfg['state_root'])/'roadmodel/runtime.sqlite3') as c:
 assert not c.execute("select 1 from tasks where state in ('QUEUED','RUNNING')").fetchone(),'Old RoadModel task active'
ticket_db=pathlib.Path(cfg['state_root'])/'roadmodel/model-tickets/tickets.sqlite3'
if ticket_db.exists():
 with sqlite3.connect(ticket_db) as c:assert not c.execute("select 1 from tickets where state in ('CHECKING','PREPARING','QUEUED','RUNNING')").fetchone(),'Model work order active; wait before migration'
backup=stage/'before';backup.mkdir();plugin=base/'bench_gateway/roadmodel_plugin.py';shutil.copy2(plugin,backup/'roadmodel_plugin.py')
with store.connect() as src:
 with sqlite3.connect(backup/'gateway.sqlite3') as dst:src.backup(dst)
with sqlite3.connect(pathlib.Path(cfg['state_root'])/'roadmodel/runtime.sqlite3') as src:
 with sqlite3.connect(backup/'roadmodel.sqlite3') as dst:src.backup(dst)
if ticket_db.exists():
 with sqlite3.connect(ticket_db) as src:
  with sqlite3.connect(backup/'model-tickets.sqlite3') as dst:src.backup(dst)
stopped=stop(cfg,wait_seconds=40);assert stopped['phase']=='STOPPED' and not stopped.get('active')
try:
 body=(release/'tools/model_ticket_plugin.py').read_bytes();compile(body,str(plugin),'exec');temp=plugin.with_suffix('.ticket.tmp');temp.write_bytes(body);os.replace(temp,plugin)
 start(cfg)
 for _ in range(40):
  s=status(cfg)
  if s.get('active') and s.get('phase')=='RUNNING' and all(c.get('ready') for c in s['children'].values()):break
  time.sleep(1)
 else:raise ValueError('Bot readiness not confirmed')
except Exception:
 if not status(cfg).get('active'):shutil.copy2(backup/'roadmodel_plugin.py',plugin);start(cfg)
 raise
receipt={'state':'DEPLOYED','release':str(release),'plugin_sha256':sha(plugin),'service_instance':s['instance'],
         'existing_bridge_reused':True,'new_inference_frames':0,'compile_process_untouched':True,'old_task_database_preserved':True}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    print(call('RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+code,timeout=150))


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else deploy()
