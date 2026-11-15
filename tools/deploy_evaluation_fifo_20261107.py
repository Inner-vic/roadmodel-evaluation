"""Admit the new evaluation card through the original shared hardware worker."""
import io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]

def main():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name in ('pcm_eval/compile_hardware_queue.py','tools/evaluation_worker.py','tests/test_compile_hardware_queue.py'):
            tar.add(ROOT/name,arcname=name)
    result=remote('''import pathlib,sys,io,tarfile,ast,json,hashlib,subprocess,os,signal,time,shutil
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
base=pathlib.Path('/mnt/data/bench-eval/service/gateway')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107/fifo-admission');stage.mkdir(exist_ok=True)
assert not (stage/'deployment.json').exists(),'Already deployed'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path[:0]=[str(base),str(release)]
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status
gateway=load(base);before=status(gateway);assert before['active'] and before['phase']=='RUNNING'
with Store(gateway['state_root']).connect() as db:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if db.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():
   assert not db.execute('select 1 from '+table+" where state in ('QUEUED','RUNNING','STARTING','PREPARING')").fetchone(),'Shared work must drain before reload'
plugin=base/'bench_gateway/roadmodel_plugin.py';queue=base/'bench_gateway/roadmodel_queue.py'
assert sha(plugin)=='12d1ab199481967077bcc56a52fce3f0c4d0633d1b1e75d18c70ae1a978304c4','Inspect concurrent routing changes'
old=plugin.read_text();needle="                    if json.loads((ROOT/'configs'/config).read_text()).get('bench_available') is not False:"+chr(10)+"                        raise ValueError('This board execution path is not admitted to the shared queue')"
replacement="                    if json.loads((ROOT/'configs'/config).read_text()).get('bench_available') is not False:"+chr(10)+"                        if name!='evaluation_entry':"+chr(10)+"                            raise ValueError('This board execution path is not admitted to the shared queue')"+chr(10)+"                        backend('compile_hardware_queue').validate_admission_config(json.loads((ROOT/'configs'/config).read_text()))"
assert old.count(needle)==1
newplugin=old.replace(needle,replacement)
oldqueue=queue.read_text();needleq="    dispatch={'entered':False}"
assert oldqueue.count(needleq)==1
newqueue=oldqueue.replace(needleq,"    if queued['payload'].get('entry_runtime')=='roadmodel-evaluation/v1':"+chr(10)+"        from .roadmodel_plugin import backend"+chr(10)+"        return backend('compile_hardware_queue').execute_queued(cfg,store,queued)"+chr(10)+needleq)
ast.parse(newplugin);ast.parse(newqueue)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();path=(release/member.name).resolve();path.relative_to(release)
  raw=tar.extractfile(member).read();ast.parse(raw)
  if path.exists():
   backup=stage/member.name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
tests=subprocess.run([sys.executable,'-m','unittest','tests.test_compile_hardware_queue','tests.test_evaluation','tests.test_compile_execution','-q'],cwd=release,capture_output=True,text=True,timeout=80)
(stage/'tests.log').write_text(tests.stdout+tests.stderr)
assert tests.returncode==0,(tests.stdout+tests.stderr)[-2200:]
shutil.copy2(plugin,stage/'before-plugin.py');shutil.copy2(queue,stage/'before-queue.py')
config=release/'configs/evaluation.json';shutil.copy2(config,stage/'before-evaluation.json');cfg=json.loads(config.read_text())
assert cfg['bench_available'] is True and cfg['execution_scope']=='validation'
cfg['shared_hardware_queue']={'enabled':True,'action':'roadmodel_hardware','adapter_sha256':sha(release/'pcm_eval/compile_hardware_queue.py')}
def replace(path,text):
 tmp=path.with_suffix('.fifo.tmp');tmp.write_text(text);os.replace(tmp,path)
replace(config,json.dumps(cfg,indent=2));replace(queue,newqueue);replace(plugin,newplugin)
from pcm_eval.compile_hardware_queue import validate_admission_config
validate_admission_config(cfg)
# Reload only the idle shared worker, keeping the real Bot connection intact.
worker=before['children']['worker'];bridge=before['children']['bridge'];assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker['pid']),'cmdline').read_bytes()
os.kill(worker['pid'],signal.SIGTERM)
for _ in range(30):
 time.sleep(1);after=status(gateway)
 assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge['pid']
 if after['children']['worker']['pid']!=worker['pid'] and after['children']['worker']['ready']:break
else:raise ValueError('Reload needs review')
receipt={'state':'DEPLOYED','test_output':(tests.stdout+tests.stderr)[-500:],
 'plugin_sha256':sha(plugin),'queue_sha256':sha(queue),'adapter_sha256':cfg['shared_hardware_queue']['adapter_sha256'],
 'bridge_pid_unchanged':bridge['pid'],'worker_pid':after['children']['worker']['pid'],
 'existing_card_continues':'rm-aeedd04d020548c6aa58','scope':'validation','full_not_resumed':True}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
''',stream.getvalue(),timeout=140)
    out=ROOT/'evidence/new-card-dual-20261107/fifo-deployment.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
