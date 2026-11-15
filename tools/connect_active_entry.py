"""Replace the report-only trial branch with scope-verified native task adoption."""
from pathlib import Path
import io,json,os,subprocess,sys,tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command
ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/trial-core-20261030'
BASE='/mnt/data/bench-eval/service/gateway'
TASK='/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01'
COMPILE='/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'
PY='/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'
def call(code,payload=b'',timeout=100):
    options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    p=subprocess.run(ssh_command('ubuntu-workstation',[PY,'-c',code]),input=payload,capture_output=True,timeout=timeout,**options)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2200:])
    return p.stdout.decode()

def prepare():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as t:
        for folder in ('pcm_eval','configs','schemas'):
            for p in (ROOT/folder).glob('*'):
                if p.is_file():t.add(p,arcname=p.relative_to(ROOT).as_posix())
    code="""import pathlib,sys,tarfile,io,json,hashlib
release=pathlib.Path(%r)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  p=(release/m.name).resolve();p.relative_to(release);raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() and p.read_bytes()!=raw:raise ValueError('Different release contents; choose new release')
  p.write_bytes(raw)
sys.path.insert(0,str(release));from pcm_eval.common import save_json,digest;from pcm_eval.two_link import resolve_pair
cat=json.loads((release/'configs/two_link_catalog.json').read_text());cat['material_root']=str(release)
for kind,task in [('model_accuracy',%r),('compile_alignment',%r)]:
 frozen=pathlib.Path(task)/'request.json';request=json.loads(frozen.read_text())
 bound={'schema':'roadmodel.active-execution/v1','kind':kind,'native_task_id':request['task_id'],'version':request['model_version'],
        'model_delivery_url':cat['models'][0]['url'],'dataset_url':cat['datasets'][0]['url'],'dataset_id':cat['datasets'][0]['id'],
        'expected_frames':request['expected_frames'],'execution_root':task,'frozen_request':{'path':str(frozen),'sha256':digest(frozen)},
        'delivery_receipt':%r+'/full-report/publication/verification.json'}
 name='active/'+kind+'.json';save_json(release/name,bound)
 cat['models'][0]['chains'][kind]={'version':bound['version'],'dataset_id':bound['dataset_id'],'active_execution':{'path':name,'sha256':digest(release/name)}}
save_json(release/'configs/two_link_catalog.json',cat,overwrite=True)
r=resolve_pair({'model_delivery_url':cat['models'][0]['url'],'dataset_url':cat['datasets'][0]['url']},'rm-core-preflight');assert r['ready'],r
from pcm_eval.active_execution import verify,poll
out=[]
for j in r['jobs']:
 b=verify(j['material_root'],j['active_execution'],j['kind'],j['version'],j['dataset_id']);d=poll(b);assert d['state']=='RUNNING',d;out.append({'kind':j['kind'],'state':d['state'],'completed_frames':d['completed_frames']})
print(json.dumps({'state':'PREPARED','chains':out}))
""" % (RELEASE,TASK,COMPILE,TASK)
    return call(code,stream.getvalue())

def deploy():
    code="""import pathlib,sys,json,time,hashlib,sqlite3,shutil,os
base=pathlib.Path(%r);release=pathlib.Path(%r);sys.path[:0]=[str(base),str(release)]
from bench_gateway.config import load,save
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
from pcm_eval.runtime import Runtime,execute_atomic
from pcm_eval.two_link import resolve_pair
cfg=load(base);store=Store(cfg['state_root']);prior=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030');installed=json.loads((prior/'deployment.json').read_text())
for n,h in installed['files'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,'Gateway changed; inspect first'
with store.connect() as c:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if c.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not c.execute("select 1 from "+table+" where state in ('QUEUED','RUNNING')").fetchone(),'Drain shared work first'
old=status(cfg);assert old['active'];backup=pathlib.Path('/mnt/data/bench-eval/backups')/('roadmodel-core-'+time.strftime('%%Y%%m%%d-%%H%%M%%S'));backup.mkdir(parents=True,exist_ok=False)
with store.connect() as src:
 with sqlite3.connect(backup/'gateway.sqlite3') as dst:src.backup(dst)
rtroot=pathlib.Path(cfg['state_root'])/'roadmodel'
with sqlite3.connect(rtroot/'runtime.sqlite3') as src:
 with sqlite3.connect(backup/'roadmodel.sqlite3') as dst:src.backup(dst)
shutil.copy2(base/'bench_gateway/roadmodel_plugin.py',backup/'roadmodel_plugin.py')
stopped=stop(cfg,wait_seconds=40);assert stopped['phase']=='STOPPED' and not stopped.get('active')
plugin=chr(10).join(['import sys','from pathlib import Path','ROOT=Path('+repr(str(release))+')','if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))','from pcm_eval.feishu_entry import handle,recover,tick,deliver',''])
compile(plugin,'roadmodel_plugin.py','exec');p=base/'bench_gateway/roadmodel_plugin.py';q=p.with_suffix('.core.tmp');q.write_text(plugin);os.replace(q,p)
rt=Runtime(rtroot,catalog_file=release/'configs/two_link_catalog.json');adopted=[]
with rt.connect() as c:rows=c.execute("select * from tasks where state='SUCCEEDED' and event_id is not null").fetchall()
for row in rows:
 before=rt.get(row['id'],row['owner'])
 if not all(x['detail'].get('mode')=='reuse_verified_results' for x in before['chains']):continue
 resolved=resolve_pair(json.loads(row['inputs']),row['id'],catalog_file=rt.catalog_file)
 assert resolved['ready'] and all(j.get('active_execution') for j in resolved['jobs'])
 history=rtroot/'runs'/row['id']/'report-only-completion.before-core.json';assert not history.exists();save(history,before)
 details={}
 for job in resolved['jobs']:details[job['kind']]=execute_atomic(job,rtroot/'runs'/row['id']/'native-evaluation'/job['kind'])
 assert all(d['state']=='RUNNING' for d in details.values())
 with rt.connect() as c:
  c.execute('BEGIN IMMEDIATE');current=c.execute('select state,event_id from tasks where id=?',(row['id'],)).fetchone();assert current['state']==row['state'] and current['event_id']==row['event_id']
  for kind,detail in details.items():c.execute('update chains set state=?,detail=? where task=? and kind=?',(detail['state'],json.dumps(detail,ensure_ascii=False),row['id'],kind))
  c.execute("update tasks set state='RUNNING',notice=?,sent_digest='',updated=? where id=?",('纠正此前报告复用的完成状态：现已关联同输入的真实评测，继续未完成范围；不重复已完成帧。',time.time(),row['id']))
 adopted.append(row['id'])
start(cfg)
for _ in range(45):
 s=status(cfg)
 if s.get('active') and s.get('phase')=='RUNNING' and all(x.get('ready') for x in s['children'].values()):break
 time.sleep(1)
else:raise ValueError('Gateway readiness not confirmed')
receipt={'state':'CORE_EXECUTIONS_ATTACHED','service_instance':s['instance'],'adopted_tasks':adopted,'release':str(release),'backup':str(backup),'new_or_repeated_inference_dispatched_by_migration':0,'compile_process_untouched':True,'real_user_submit_previously_verified':True,'full_execution_and_report_completed':False,'files':{n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in installed['files']}}
save(prior/'core-deployment.json',receipt);print(json.dumps(receipt))
""" % (BASE,RELEASE)
    return call(code,timeout=110)

if __name__=='__main__':print(prepare() if sys.argv[1]=='prepare' else deploy())
