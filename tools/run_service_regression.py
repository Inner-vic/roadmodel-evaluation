"""Run dispatch/publication unit tests on the deployed Linux interpreter.

All child launchers and Feishu operations in these tests are mocked. No native
evaluation or message is dispatched. Keep evidence outside immutable releases.
"""
import io,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
ROOT=Path(__file__).resolve().parents[1]
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tar:
    for name in ('tests/test_retained_runtime.py','tests/test_native_dispatch.py','tests/test_report_dispatch.py',
                 'tests/test_feishu_update.py','examples/two_link_submission.json'):
        tar.add(ROOT/name,arcname=name)
code="""import pathlib,sys,tarfile,io,subprocess,os,json
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/r4-service-tests');stage.mkdir(parents=True,exist_ok=False);(stage/'evidence').mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  p=(stage/m.name).resolve();p.relative_to(stage);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
env={**os.environ,'PYTHONPATH':'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4'}
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*dispatch.py','-q'],cwd=stage,env=env,capture_output=True)
q=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_feishu_update.py','-q'],cwd=stage,env=env,capture_output=True)
value={'state':'PASSED' if p.returncode==q.returncode==0 else 'FAILED','platform':'deployed-linux-python312','native_inference':0,'feishu_writes':0,'dispatch_tests':(p.stdout+p.stderr).decode(),'publication_tests':(q.stdout+q.stderr).decode()}
(stage/'receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value))
assert value['state']=='PASSED'
"""
if __name__=='__main__':print(call(code,stream.getvalue()))
