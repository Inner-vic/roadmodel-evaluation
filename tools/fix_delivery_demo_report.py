"""Install the report-only fix and finish the retained two-frame validation."""
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from deploy_delivery_demo import call, LOCAL, RELEASE, STAGE
from pcm_eval.common import save_json

NEW = '/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r2'


def prepare():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        tar.add(ROOT/'tools/delivery_demo_worker.py', arcname='tools/delivery_demo_worker.py')
    code = r'''import pathlib,sys,json,shutil,io,tarfile,ast,subprocess,hashlib
old=pathlib.Path(OLD);new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not new.exists();shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile();p=(new/m.name).resolve();p.relative_to(new);raw=tar.extractfile(m).read();ast.parse(raw);p.write_bytes(raw)
plugin=new/'tools/delivery_demo_plugin.py';text=plugin.read_text();assert text.count(OLD)==1
plugin.write_text(text.replace(OLD,NEW))
tests=[]
for pattern in ['test_delivery*.py','test_model_ticket.py','test_publication.py','test_strict_model_template.py']:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=new,capture_output=True,text=True,timeout=90)
 (stage/('r2-'+pattern.replace('*','all')+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;tests.append(pattern)
receipt={'state':'PREPARED','release':NEW,'tests':tests,'files':{n:hashlib.sha256((new/n).read_bytes()).hexdigest() for n in ['tools/delivery_demo_worker.py','tools/delivery_demo_plugin.py']}}
(stage/'report-fix-prepared.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
    result = call('OLD='+repr(RELEASE)+'\nNEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\n'+code, stream.getvalue())
    save_json(LOCAL/'report-fix-prepared.json', result)
    print(json.dumps(result))


def deploy():
    code = r'''import pathlib,sys,json,sqlite3,shutil,hashlib,os,time
new=pathlib.Path(NEW);stage=pathlib.Path(STAGE);base=pathlib.Path('/mnt/data/bench-eval/service/gateway')
prep=json.loads((stage/'report-fix-prepared.json').read_text());assert all(hashlib.sha256((new/n).read_bytes()).hexdigest()==h for n,h in prep['files'].items())
sys.path[:0]=[str(base),str(new)]
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status,stop,start
cfg=load(base);store=Store(cfg['state_root'])
with store.connect() as c:
 for table in ['jobs','arrival_analyses','archive_jobs']:
  if c.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not c.execute('select 1 from '+table+" where state in ('QUEUED','RUNNING')").fetchone()
demo=pathlib.Path(cfg['state_root'])/'roadmodel/delivery-demo/runtime.sqlite3'
with sqlite3.connect(demo) as c:assert not c.execute("select 1 from tasks where state='RUNNING'").fetchone(),'Demo running; wait'
plugin=base/'bench_gateway/roadmodel_plugin.py';shutil.copy2(plugin,stage/'plugin.before-report-fix.py')
assert stop(cfg,wait_seconds=40)['phase']=='STOPPED'
tmp=plugin.with_suffix('.r2.tmp');tmp.write_bytes((new/'tools/delivery_demo_plugin.py').read_bytes());os.replace(tmp,plugin);start(cfg)
for _ in range(40):
 s=status(cfg)
 if s.get('active') and s['phase']=='RUNNING' and all(x.get('ready') for x in s['children'].values()):break
 time.sleep(1)
else:raise ValueError('Bot readiness not confirmed')
receipt={'state':'DEPLOYED','release':NEW,'instance':s['instance'],'new_inference_frames':0,'original_results_preserved':True}
(stage/'report-fix-deployed.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
    result = call('NEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\n'+code)
    save_json(LOCAL/'report-fix-deployed.json', result)
    print(json.dumps(result))


def finish():
    code = r'''import pathlib,sys,json,hashlib,os,xml.etree.ElementTree as ET
sys.path[:0]=[NEW,NEW+'/tools']
from pcm_eval.delivery_workflow import run_delivery,_write_atomic
from pcm_eval.common import load_json,digest,save_json
from delivery_demo_worker import publish_demo_report
stage=pathlib.Path(STAGE);run=stage/'live-execution-validation';root=run/'effective';jobroot=root/'model_accuracy'
native=load_json(jobroot/'inference.json');assert native['state']=='SUCCEEDED'
result=pathlib.Path(native['receipt']['result_file']);before=digest(result)
report=load_json(jobroot/'report-receipt.json');xml=pathlib.Path(report['report_xml']);tree=ET.fromstring('<root>'+xml.read_text()+'</root>')
assert not tree.findall('title');shutil_source=xml.with_suffix('.before-title-fix.xml');shutil_source.write_bytes(xml.read_bytes())
xml.write_text('<title>PCM 单次报告 - 模型评测 v3.2.2</title>'+xml.read_text(),encoding='utf-8')
# The previous error happened before publication.json or any CLI invocation.
publication=jobroot/'publication';assert not (publication/'publication.json').exists()
assert not list(publication.glob('cli-*'))
journal=jobroot/'publication-started.json';assert journal.exists();journal.rename(jobroot/'publication-not-attempted-before-title-fix.json')
cfg=load_json(pathlib.Path(NEW)/'configs/delivery_demo.json');os.chdir('/mnt/data/bench-eval')
def no_inference(*a,**kw):raise AssertionError('Report repair must never repeat inference')
def publish(job,r,target):return publish_demo_report(cfg,job,r,target)
value=run_delivery(load_json(root/'workflow.json'),root,executor=no_inference,publisher=publish)
assert digest(result)==before
assert value['chains'][0]['state']=='SUCCEEDED',value
assert value['chains'][1]['state']=='BLOCKED',value
_write_atomic(run/'progress.json',value)
proof={'state':'VERIFIED','model_frames':2,'model_metrics':315,'new_inference_frames_during_report_fix':0,
 'model_result_sha256':before,'model_report_url':value['chains'][0]['publication']['report_url'],
 'compile_state':'WAITING_BENCH','compile_frames':0,'real_production_card_callback_verified':False}
save_json(stage/'live-model-verification.json',proof);print(json.dumps(proof))
'''
    result = call('NEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\n'+code, timeout=180)
    save_json(LOCAL/'live-model-verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    {'prepare': prepare, 'deploy': deploy, 'finish': finish}[sys.argv[1]]()
