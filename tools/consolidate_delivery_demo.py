"""Install the one-entry/one-report Demo and update its existing verified document."""
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from deploy_delivery_demo import call
from pcm_eval.common import save_json

OLD = '/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r2'
NEW = '/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r3'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/delivery-combined-20261103-r1'
PRIOR = '/mnt/data/bench-eval/roadmodel/deployment/delivery-demo-20261103-r1'
LOCAL = ROOT/'evidence/delivery-combined-20261103'


def remote(code, body=b'', timeout=150):
    return call('OLD='+repr(OLD)+'\nNEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\nPRIOR='+repr(PRIOR)+'\n'+code,
                body, timeout=timeout)


def prepare():
    files = ['pcm_eval/combined_delivery.py', 'pcm_eval/combined_publication.py',
             'pcm_eval/delivery_demo.py', 'pcm_eval/delivery_demo_entry.py',
             'tools/delivery_demo_worker.py', 'tests/test_delivery_combined.py']
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name in files:
            tar.add(ROOT/name, arcname=name)
    code = r'''import pathlib,sys,json,shutil,io,tarfile,ast,subprocess,hashlib,re
old=pathlib.Path(OLD);new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Already deployed; preserve the installed release'
stage.mkdir(parents=True,exist_ok=True)
if not new.exists():shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile();p=(new/m.name).resolve();p.relative_to(new);raw=tar.extractfile(m).read();ast.parse(raw);p.write_bytes(raw)
plugin=new/'tools/delivery_demo_plugin.py';text=plugin.read_text()
if OLD in text:
 assert text.count(OLD)==1;plugin.write_text(text.replace(OLD,NEW))
assert NEW in plugin.read_text()
counts={}
for pattern in ['test_delivery*.py','test_model_ticket.py','test_publication.py','test_strict_model_template.py']:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=new,capture_output=True,text=True,timeout=90)
 text=p.stdout+p.stderr;(stage/(pattern.replace('*','all')+'.log')).write_text(text)
 assert p.returncode==0,text[-3000:];counts[pattern]=int(re.search(r'Ran ([0-9]+) tests?',text).group(1))
sys.path[:0]=[str(new),str(new/'tools')]
from pcm_eval.common import load_json,digest,save_json
from pcm_eval.combined_delivery import write_combined
from pcm_eval.feishu_update import table_values
import xml.etree.ElementTree as ET
run=pathlib.Path(PRIOR)/'live-execution-validation';workflow=load_json(run/'effective/workflow.json');result=load_json(run/'progress.json')
cfg=load_json(new/'configs/delivery_demo.json')
assert not cfg['bench_available']
result['chains'][1]['issues']=['等待台架可用；未派发编译推理。']
report=write_combined(workflow,result,stage/'preview',reference_file=cfg['reference_file'],template_file=new/'configs/report_template.json')
old_xml=load_json(run/'effective/model_accuracy/report-receipt.json')['report_xml']
values=table_values(ET.fromstring('<root>'+pathlib.Path(report['report_xml']).read_text()+'</root>'))
assert len(values)==5 and values[:3]==table_values(ET.fromstring('<root>'+pathlib.Path(old_xml).read_text()+'</root>'))
assert not report['all_evaluations_complete']
assert '等待台架' in pathlib.Path(report['report_xml']).read_text()
save_json(stage/'preview.json',report)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'state':'PREPARED','release':NEW,'workstation_tests':counts,'test_count':sum(counts.values()),
 'original_three_model_tables_unchanged':True,'combined_table_count':5,'new_inference_frames':0,
 'files':{str(p.relative_to(new)):sha(p) for name in ['pcm_eval','tools','configs'] for p in (new/name).glob('*') if p.is_file()}}
save_json(stage/'preparation.json',receipt);print(json.dumps(receipt))
'''
    result = remote(code, stream.getvalue())
    save_json(LOCAL/'preparation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


def publish():
    code = r'''import pathlib,sys,json,os,xml.etree.ElementTree as ET
sys.path[:0]=[NEW,NEW+'/tools'];os.chdir('/mnt/data/bench-eval')
from pcm_eval.common import load_json,digest,save_json
from delivery_demo_worker import finish_combined
from pcm_eval.feishu_update import table_values
stage=pathlib.Path(STAGE);run=pathlib.Path(PRIOR)/'live-execution-validation';cfg=load_json(pathlib.Path(NEW)/'configs/delivery_demo.json')
workflow=load_json(run/'effective/workflow.json');result=load_json(run/'progress.json')
native=run/'effective/model_accuracy/inference.json';native_before=digest(native)
model_result=pathlib.Path(load_json(native)['receipt']['result_file']);result_before=digest(model_result)
result['chains'][1]['issues']=['等待台架可用；未派发编译推理。']
old=load_json(pathlib.Path(PRIOR)/'live-model-verification.json');doc=old['model_report_url'].rstrip('/').split('/')[-1]
value=finish_combined(run,cfg,workflow,result,document_id=doc)
assert digest(native)==native_before and digest(model_result)==result_before
report=value['combined_report'];assert report['state']=='VERIFIED' and report['document_id']==doc
assert report['table_count']==5 and not report['all_evaluations_complete']
proof={'state':'VERIFIED','report_url':report['report_url'],'revision_id':report.get('revision_id'),
 'same_document_as_previous_demo':True,'combined_table_count':5,'all_metric_cells_match':True,
 'model_frames':2,'model_metrics':len(load_json(model_result)['metrics']),
 'compile_state':'WAITING_BENCH','new_inference_frames':0,'native_result_sha256':result_before}
save_json(stage/'publication-verification.json',proof);print(json.dumps(proof))
'''
    result = remote(code, timeout=180)
    save_json(LOCAL/'publication-verification.json', result)
    print(json.dumps(result))


def deploy():
    code = r'''import pathlib,sys,json,sqlite3,hashlib,shutil,os,time
new=pathlib.Path(NEW);stage=pathlib.Path(STAGE);base=pathlib.Path('/mnt/data/bench-eval/service/gateway')
assert not (stage/'deployment.json').exists()
prep=json.loads((stage/'preparation.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(new/n)==h for n,h in prep['files'].items())
assert json.loads((stage/'publication-verification.json').read_text())['state']=='VERIFIED'
sys.path[:0]=[str(base),str(new)]
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
cfg=load(base);store=Store(cfg['state_root'])
with store.connect() as c:
 for table in ['jobs','arrival_analyses','archive_jobs']:
  if c.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not c.execute('select 1 from '+table+" where state in ('QUEUED','RUNNING')").fetchone(),'Shared work active'
demo=pathlib.Path(cfg['state_root'])/'roadmodel/delivery-demo/runtime.sqlite3'
with sqlite3.connect(demo) as c:assert not c.execute("select 1 from tasks where state='RUNNING'").fetchone(),'Evaluation active; wait'
backup=stage/'before';backup.mkdir();plugin=base/'bench_gateway/roadmodel_plugin.py';shutil.copy2(plugin,backup/'roadmodel_plugin.py')
with sqlite3.connect(demo) as src:
 with sqlite3.connect(backup/'demo.sqlite3') as dst:src.backup(dst)
assert stop(cfg,wait_seconds=40)['phase']=='STOPPED'
try:
 with sqlite3.connect(demo) as c:assert not c.execute("select 1 from tasks where state='RUNNING'").fetchone(),'Evaluation started during drain'
 tmp=plugin.with_suffix('.combined.tmp');tmp.write_bytes((new/'tools/delivery_demo_plugin.py').read_bytes());os.replace(tmp,plugin)
 start(cfg)
 for _ in range(40):
  s=status(cfg)
  if s.get('active') and s['phase']=='RUNNING' and all(c.get('ready') for c in s['children'].values()):break
  time.sleep(1)
 else:raise ValueError('Bot readiness not confirmed')
except Exception:
 if not status(cfg).get('active'):shutil.copy2(backup/'roadmodel_plugin.py',plugin);start(cfg)
 raise
receipt={'state':'DEPLOYED','release':NEW,'service_instance':s['instance'],
 'entry':'RoadModel评测','one_report_per_task':True,'new_inference_frames':0}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = remote(code)
    save_json(LOCAL/'deployment.json', result)
    print(json.dumps(result))


def health():
    code = r'''import sys,pathlib,json,time
sys.path[:0]=[NEW,'/mnt/data/bench-eval/service/gateway']
from bench_gateway.config import load
from bench_gateway.deployment import status
from pcm_eval.delivery_demo import DemoRuntime
from pcm_eval.delivery_demo_entry import render
cfg=load(pathlib.Path('/mnt/data/bench-eval/service/gateway'));s=status(cfg)
assert s['active'] and s['phase']=='RUNNING' and all(c.get('ready') for c in s['children'].values())
plugin=pathlib.Path('/mnt/data/bench-eval/service/gateway/bench_gateway/roadmodel_plugin.py').read_text();assert NEW in plugin
rt=DemoRuntime(pathlib.Path(cfg['state_root'])/'roadmodel/delivery-demo')
with rt.connect() as c:age=time.time()-max(r[0] for r in c.execute('select heartbeat from source_agent'))
assert age<45,age
publication=json.loads((pathlib.Path(STAGE)/'publication-verification.json').read_text())
task={'state':'PARTIAL','inputs':{},'notice':'','report_url':publication['report_url'],
 'chains':[{'kind':k,'state':'BLOCKED','detail':{}} for k in ['model_accuracy','compile_alignment']]}
card=render(task);value=json.dumps(card,ensure_ascii=False)
assert value.count(publication['report_url'])==1 and card['header']['title']['content']=='RoadModel评测'
receipt={'state':'HEALTHY','release':NEW,'entry':'RoadModel评测','single_report_button':True,
 'report_url':publication['report_url'],'source_agent_heartbeat_age_seconds':round(age,2),
 'bridge_ready':True,'worker_ready':True,'real_user_card_callback_verified':False}
(pathlib.Path(STAGE)/'health.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = remote(code)
    save_json(LOCAL/'health.json', result, overwrite=True)
    print(json.dumps(result))


if __name__ == '__main__':
    {'prepare': prepare, 'publish': publish, 'deploy': deploy, 'health': health}[sys.argv[1]]()
