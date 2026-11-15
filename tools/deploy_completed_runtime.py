"""Prepare updated runtime, bind automated report stages, preserve native work."""
import io,json,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command
ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4'
PY='/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'
def call(code,payload=b'',timeout=100):
    p=subprocess.run(ssh_command('ubuntu-workstation',[PY,'-c',code]),input=payload,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2400:])
    return p.stdout.decode()

def prepare():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as t:
        for folder in ('pcm_eval','configs','schemas'):
            for p in (ROOT/folder).glob('*'):
                if p.is_file():t.add(p,arcname=p.relative_to(ROOT).as_posix())
        t.add(ROOT/'tools/run_entry_pipeline.py',arcname='tools/run_entry_pipeline.py')
    code="""import pathlib,sys,tarfile,io,json
b=pathlib.Path(%r)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  p=(b/m.name).resolve();p.relative_to(b);raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw) if not p.exists() else None
print(json.dumps({'state':'PROGRAM_PREPARED','release':str(b)}))
""" % RELEASE
    return call(code,stream.getvalue())

def wire():
    code="""import pathlib,sys,json,sqlite3,copy
b=pathlib.Path(%r);sys.path.insert(0,str(b))
from pcm_eval.common import load_json,save_json,digest
from pcm_eval.runtime import Runtime,execute_atomic
from pcm_eval.report_dispatch import start_report_pipeline
old=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/trial-core-20261030');valid=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3');full=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');compiled=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
baseline='/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/model-report.json';reference='/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/reference.json'
def coordinator(config,path):
 save_json(path,config)
 return {'config':str(path),'config_sha256':digest(path),'script':str(b/'tools/run_entry_pipeline.py'),
         'script_sha256':digest(b/'tools/run_entry_pipeline.py'),'dispatch_receipt':str(path.with_suffix('.dispatch.json')),
         'python':%r,'cwd':str(b)}
# Resume report generation from both already completed native validation results.
rt=Runtime(valid/'runtime',catalog_file=b/'configs/validation-catalog.json')
with rt.connect() as c:row=c.execute('SELECT id,owner FROM tasks').fetchone()
before=rt.get(row['id'],row['owner']);save_json(valid/'before-report-recovery.json',before)
bindings=[copy.deepcopy(x['detail']['binding']) for x in before['chains']]
for x in bindings:x['delivery_receipt']=str(valid/'result-report-r4/publication/verification.json')
config=load_json(valid/'pipeline-admission-02.json');config['bindings']=copy.deepcopy(bindings);config['output']=str(valid/'result-report-r4');config['report']['output']=config['output'];config['runtime_root']=str(valid/'runtime');config['runtime_catalog']=str(b/'configs/validation-catalog.json')
desc=coordinator(config,valid/'pipeline-report-r4.json')
cat=load_json('/mnt/data/bench-eval/service/roadmodel/entry-live-20261030-r3/configs/two_link_catalog.json');cat['material_root']=str(b)
for binding in bindings:
 binding['report_pipeline']=desc;binding['report_coordinator_failure']=str((valid/'pipeline-report-r4.json').with_suffix('.failure.json'))
 path=save_json(b/('validation/'+binding['kind']+'.json'),binding)
 cat['models'][0]['chains'][binding['kind']]={'version':'v3.2.2','dataset_id':binding['dataset_id'],'active_execution':{'path':str(path.relative_to(b)),'sha256':digest(path)}}
save_json(b/'configs/validation-catalog.json',cat)
for binding in bindings:
 spec=cat['models'][0]['chains'][binding['kind']]
 job={'kind':binding['kind'],'version':'v3.2.2','dataset_id':binding['dataset_id'],'material_root':str(b),'active_execution':spec['active_execution']}
 detail=execute_atomic(job,valid/'report-recovery-r4'/binding['kind'])
 assert detail['inference_dispatch']=='adopt_existing_exact_input_task' and detail['repeated_completed_frames']==0
 with rt.connect() as c:c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',(detail['state'],json.dumps(detail,ensure_ascii=False),row['id'],binding['kind']))
with rt.connect() as c:c.execute("UPDATE tasks SET state='RUNNING',notice=? WHERE id=?",('原生推理已完成，仅恢复报告生成和机器人回填。',row['id']))
# Prepare the current full-run automatic completion pipeline; native controllers remain running.
cat=load_json(old/'configs/two_link_catalog.json');cat['material_root']=str(b);bindings=[]
for kind in ('model_accuracy','compile_alignment'):
 spec=cat['models'][0]['chains'][kind]['active_execution'];binding=load_json(pathlib.Path(old)/spec['path']);bindings.append(binding)
request=load_json(full/'request.json');report={'output':str(full/'full-report'),'native_task_id':request['task_id'],'document_id':request['report_document_id'],'scope':'full',
 'baseline_model':baseline,'reference':reference,'model_request':str(full/'request.json'),'compile_request':str(compiled/'request.json'),
 'compile_execution_request':request['compile_request'],'frozen_requests':{x['kind']:x['frozen_request']['sha256'] for x in bindings}}
config={'scope':'full','bindings':copy.deepcopy(bindings),'output':str(full/'full-report'),'report':report,'model_task_root':str(full),'compile_task_root':str(compiled),
        'runtime_root':'/mnt/data/bench-eval/state/roadmodel','runtime_catalog':str(b/'configs/two_link_catalog.json')}
desc=coordinator(config,full/'full-report-coordinator.json')
for binding in bindings:
 binding['report_pipeline']=desc;binding['report_coordinator_failure']=str((full/'full-report-coordinator.json').with_suffix('.failure.json'))
 path=save_json(b/('active/'+binding['kind']+'.json'),binding)
 cat['models'][0]['chains'][binding['kind']]['active_execution']={'path':str(path.relative_to(b)),'sha256':digest(path)}
save_json(b/'configs/two_link_catalog.json',cat,overwrite=True)
started=start_report_pipeline(bindings[0]);assert started['state']=='STARTED'
save_json(full/'automatic-report-activation.json',{'state':'STARTED','report_coordinator':started,'native_inference_restarted':False,'full_workers_unchanged':True})
print(json.dumps({'state':'REPORT_PIPELINES_WIRED','validation_native_reruns':0,'full_report_coordinator':started}))
""" % (RELEASE,PY)
    return call(code)

def activate():
    code="""import pathlib,sys,json,sqlite3,time,shutil,hashlib,os
b=pathlib.Path(%r);base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path[:0]=[str(b),str(base)]
from bench_gateway.config import load,save
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
from pcm_eval.runtime import Runtime,execute_atomic
from pcm_eval.two_link import resolve_pair
from pcm_eval.common import load_json,save_json
valid=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3')
vr=Runtime(valid/'runtime',catalog_file=b/'configs/validation-catalog.json');vr.refresh_active()
with vr.connect() as c:row=c.execute('SELECT id,owner FROM tasks').fetchone()
proof=vr.get(row['id'],row['owner']);assert proof['state']=='SUCCEEDED',proof
verified=load_json(valid/'result-report-r4/publication/verification.json');assert verified['five_tables_match'] and verified['scope']=='selected_validation'
cfg=load(base);store=Store(cfg['state_root']);prior=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030');installed=load_json(prior/'core-deployment.json')
for name,h in installed['files'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h,'Gateway source changed: '+name
def idle():
 with store.connect() as c:
  for table in ('jobs','arrival_analyses','archive_jobs'):
   if c.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not c.execute("select 1 from "+table+" where state in ('QUEUED','RUNNING')").fetchone(),'Drain '+table+' first'
idle();old=status(cfg);assert old['active']
stopped=stop(cfg,wait_seconds=40);assert stopped['phase']=='STOPPED' and not stopped.get('active');idle()
backup=pathlib.Path('/mnt/data/bench-eval/backups')/('roadmodel-report-runtime-'+time.strftime('%%Y%%m%%d-%%H%%M%%S'));backup.mkdir(parents=True,exist_ok=False)
rtroot=pathlib.Path(cfg['state_root'])/'roadmodel';rt=Runtime(rtroot,catalog_file=b/'configs/two_link_catalog.json')
for name,connect in [('gateway.sqlite3',store.connect),('roadmodel.sqlite3',rt.connect)]:
 with connect() as src:
  with sqlite3.connect(backup/name) as dst:src.backup(dst)
plugin=base/'bench_gateway/roadmodel_plugin.py';shutil.copy2(plugin,backup/plugin.name)
try:
 with rt.connect() as c:row=c.execute('SELECT * FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone()
 assert row and row['event_id']=='6bc45e97cb885cb060f64bf7921e51e8' and row['state']=='RUNNING'
 before=rt.get(row['id'],row['owner']);save_json(backup/'production-task-before.json',before)
 resolved=resolve_pair(json.loads(row['inputs']),row['id'],catalog_file=rt.catalog_file);assert resolved['ready']
 details={}
 for job in resolved['jobs']:
  d=execute_atomic(job,rtroot/'runs'/row['id']/'report-runtime-r4'/job['kind'])
  assert d['inference_dispatch']=='adopt_existing_exact_input_task' and d['repeated_completed_frames']==0 and d['state']=='RUNNING',d
  prior_dispatch=load_json(pathlib.Path(d['binding']['report_pipeline']['dispatch_receipt']))
  assert d['report_coordinator']==prior_dispatch and prior_dispatch['state']=='STARTED',d
  details[job['kind']]=d
 with rt.connect() as c:
  c.execute('BEGIN IMMEDIATE');assert c.execute('SELECT state FROM tasks WHERE id=?',(row['id'],)).fetchone()['state']=='RUNNING'
  for kind,d in details.items():c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',(d['state'],json.dumps(d,ensure_ascii=False),row['id'],kind))
  c.execute('UPDATE tasks SET notice=?,sent_digest=?,updated=? WHERE id=?',('真实评测继续运行；两条链路完成后自动生成、回填并校验同一份报告。','',time.time(),row['id']))
 text=chr(10).join(['import sys','from pathlib import Path','ROOT=Path('+repr(str(b))+')','if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))','from pcm_eval.feishu_entry import handle,recover,tick,deliver',''])
 compile(text,'roadmodel_plugin.py','exec');temp=plugin.with_suffix('.r4.tmp');temp.write_text(text);os.replace(temp,plugin)
except Exception:
 shutil.copy2(backup/plugin.name,plugin);start(cfg);raise
start(cfg)
for _ in range(45):
 s=status(cfg)
 if s.get('active') and s.get('phase')=='RUNNING' and s['instance']!=old['instance'] and all(x.get('ready') for x in s['children'].values()):break
 time.sleep(1)
else:raise ValueError('Gateway readiness needs review; native workers were not changed')
receipt={'state':'REAL_EXECUTION_AND_REPORT_RUNTIME_DEPLOYED','release':str(b),'backup':str(backup),'service_instance':s['instance'],
 'production_task':row['id'],'real_user_callback_previously_verified':True,'internal_validation_task':proof['id'],'validation_report':verified,
 'model_validation_frames':37,'compile_validation_frames':1,'new_inference_during_deployment':0,'full_native_workers_unchanged':True,
 'files':{name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in installed['files']}}
save_json(prior/'report-runtime-r4-deployment.json',receipt);print(json.dumps(receipt))
""" % RELEASE
    return call(code,timeout=115)

if __name__=='__main__':print({'prepare':prepare,'wire':wire,'activate':activate}[sys.argv[1]]())
