"""Version-checked, backed-up migration of the existing RoadModel bot plugin."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'inspection/gateway-migration-20261030'
PACKAGE=ROOT/'evidence/trial-entry-20261030/package'
RELEASE='/mnt/data/bench-eval/service/roadmodel/trial-20261030'
GATEWAY='/mnt/data/bench-eval/service/gateway'
PY='/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'

def call(code,payload=b'',timeout=90):
    p=subprocess.run(ssh_command('ubuntu-workstation',[PY,'-c',code]),input=payload,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2200:])
    return p.stdout.decode()

def patches():
    originals={n:(SNAP/n).read_text(encoding='utf8') for n in ['bench_gateway/cc_agent.py','bench_gateway/worker.py']}
    agent=originals['bench_gateway/cc_agent.py'];worker=originals['bench_gateway/worker.py']
    marker='        # The CC adapter adds project/platform preambles on initial turns.'
    assert agent.count(marker)==1 and 'roadmodel_handle' not in agent
    hook="""        from .roadmodel_plugin import handle as roadmodel_handle
        roadmodel_answer=roadmodel_handle(service,session,text)
        if roadmodel_answer is not None:
            emit({'type':'item.completed','item':{'type':'agent_message','text':roadmodel_answer}})
            emit({'type':'turn.completed','usage':{'input_tokens':0,'cached_input_tokens':0,'output_tokens':0}})
            return
"""
    agent=agent.replace(marker,hook+marker)
    marker='        recover_arrival(store)\n';assert worker.count(marker)==1
    worker=worker.replace(marker,marker+'        from .roadmodel_plugin import recover as recover_roadmodel\n        recover_roadmodel(cfg)\n')
    marker='    # Card API failures must not starve final results or required user actions.\n';assert worker.count(marker)==1
    worker=worker.replace(marker,"    if not expired():\n        from .roadmodel_plugin import deliver as deliver_roadmodel\n        deliver_roadmodel(cfg,store)\n"+marker)
    marker='        try:\n            while True:\n                # Drain active work';assert worker.count(marker)==1
    hook="""        def roadmodel_processing():
            from .roadmodel_plugin import tick as roadmodel_tick
            while not stopped.is_set():
                if os.environ.get('BENCH_GATEWAY_MANAGED')=='1' and (Path(cfg['project_root'])/'runtime/service/drain.json').exists():return
                try:roadmodel_tick(cfg)
                except Exception:print(json.dumps({'component':'roadmodel','event':'dispatch_failed'}),flush=True)
                stopped.wait(1)
        roadmodel_worker=None
        if not once:
            roadmodel_worker=threading.Thread(target=roadmodel_processing,daemon=True);roadmodel_worker.start()
"""
    worker=worker.replace(marker,hook+marker)
    marker='                if once:\n';assert worker.count(marker)==1
    worker=worker.replace(marker,marker+'                    from .roadmodel_plugin import tick as roadmodel_tick\n                    roadmodel_tick(cfg)\n')
    marker='            if preparer:preparer.join()\n';assert worker.count(marker)==1
    worker=worker.replace(marker,marker+'            if roadmodel_worker:roadmodel_worker.join()\n')
    plugin='"""Workstation-only RoadModel plugin on the existing authenticated bridge."""\nimport sys\nfrom pathlib import Path\nROOT=Path('+repr(RELEASE)+')\nif str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))\nfrom pcm_eval.feishu_entry import handle,recover,tick,deliver\n'
    changed={'bench_gateway/cc_agent.py':agent,'bench_gateway/worker.py':worker,'bench_gateway/roadmodel_plugin.py':plugin}
    for n,t in changed.items():compile(t,n,'exec')
    return originals,changed

def prepare():
    originals,changed=patches()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as t:
        for folder in ('pcm_eval','schemas','configs'):
            for p in (ROOT/folder).glob('*'):
                if p.is_file():t.add(p,arcname='release/'+p.relative_to(ROOT).as_posix())
        for p in PACKAGE.rglob('*'):
            if p.is_file() and p.relative_to(PACKAGE).as_posix()!='configs/two_link_catalog.json':t.add(p,arcname='release/'+p.relative_to(PACKAGE).as_posix())
        # Use the reviewed retained-result catalog, never the old live PC binding.
        p=PACKAGE/'configs/two_link_catalog.json';t.add(p,arcname='catalog.json')
        raw=json.dumps({'expected':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in originals.items()},'files':changed}).encode()
        m=tarfile.TarInfo('patch.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    code="""import pathlib,sys,tarfile,io,json
release=pathlib.Path(%r);stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030');stage.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  if m.name=='catalog.json':p=release/'configs/two_link_catalog.json'
  elif m.name=='patch.json':p=stage/'patch.json'
  else:
   assert m.name.startswith('release/');p=(release/m.name[len('release/'):]).resolve();p.relative_to(release)
  body=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() and p.read_bytes()!=body and m.name!='catalog.json':raise ValueError('Different staged file')
  p.write_bytes(body)
sys.path.insert(0,str(release));from pcm_eval.two_link import resolve_pair
catalog=json.loads((release/'configs/two_link_catalog.json').read_text());inputs={'model_delivery_url':catalog['models'][0]['url'],'dataset_url':catalog['datasets'][0]['url']};r=resolve_pair(inputs,'rm-stage-verification');assert r['ready'],r
print(json.dumps({'state':'PREPARED','chains':len(r['jobs']),'new_inference_frames':0}))
""" % RELEASE
    return call(code,stream.getvalue())

def deploy():
    code="""import pathlib,sys,json,hashlib,shutil,sqlite3,time,subprocess,os
base=pathlib.Path(%r);stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030');patch=json.loads((stage/'patch.json').read_text());sys.path.insert(0,str(base))
from bench_gateway.config import load,save
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
cfg=load(base);store=Store(cfg['state_root']);old=status(cfg);assert old['active'] and old['phase']=='RUNNING'
for n,h in patch['expected'].items():assert hashlib.sha256((base/n).read_text().encode()).hexdigest()==h,'Gateway code changed; re-review before deployment'
def idle():
 with store.connect() as c:
  for table in ('jobs','preparations','arrival_analyses','archive_jobs'):
   if c.execute("SELECT 1 FROM sqlite_master WHERE name=?",(table,)).fetchone():
    assert not c.execute("SELECT 1 FROM "+table+" WHERE state IN ('QUEUED','RUNNING') LIMIT 1").fetchone(),'Existing work must drain first: '+table
idle();backup=pathlib.Path('/mnt/data/bench-eval/backups')/('roadmodel-entry-'+time.strftime('%%Y%%m%%d-%%H%%M%%S'));backup.mkdir(parents=True,exist_ok=False)
for n in patch['files']:
 p=base/n
 if p.exists():d=backup/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
config_hash=hashlib.sha256((base/'config/gateway.json').read_bytes()).hexdigest()
with store.connect() as source:
 with sqlite3.connect(backup/'gateway.sqlite3') as dest:source.backup(dest)
receipt={'state':'DRAINING','backup':str(backup),'compile_process_untouched':True,'new_inference_frames':0};save(stage/'deployment.json',receipt)
stopped=stop(cfg,wait_seconds=40);assert not stopped.get('active') and stopped['phase']=='STOPPED','Normal drain not complete; no code changed';idle()
for n,h in patch['expected'].items():assert hashlib.sha256((base/n).read_text().encode()).hexdigest()==h
for n,body in patch['files'].items():
 compile(body,n,'exec');p=base/n;tmp=p.with_suffix('.roadmodel.tmp');tmp.write_text(body);os.replace(tmp,p)
assert hashlib.sha256((base/'config/gateway.json').read_bytes()).hexdigest()==config_hash
started=start(cfg)
for _ in range(45):
 current=status(cfg)
 if current.get('active') and current.get('phase')=='RUNNING' and current['children']['worker'].get('ready') and current['children']['bridge'].get('ready'):break
 time.sleep(1)
else:raise ValueError('Service not ready; preserve deployment and backup')
assert current['instance']!=old['instance']
receipt.update(state='DEPLOYED',service_instance=current['instance'],bridge_ready=True,worker_ready=True,config_unchanged=True,extra_feishu_connections=0,files={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in patch['files']});save(stage/'deployment.json',receipt);print(json.dumps(receipt))
""" % GATEWAY
    return call(code,timeout=110)

if __name__=='__main__':
    print(prepare() if sys.argv[1]=='prepare' else deploy())
