"""Deploy a GT-only capability and bounded validation; do not resume native work."""
import io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/parallel-gt-20261030-r2'
OUT='/mnt/data/bench-eval/roadmodel/validation/parallel-gt-20261030-r2'


def prepare():
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in (ROOT/'pcm_eval').glob('*.py'):tar.add(p,arcname='pcm_eval/'+p.name)
        for name in ('benchmark_parallel_gt.py',):tar.add(ROOT/'tools'/name,arcname='tools/'+name)
        for name in ('test_parallel_gt.py','test_memory_gt.py'):tar.add(ROOT/'tests'/name,arcname='tests/'+name)
    code="""import pathlib,tarfile,io,sys,json,subprocess
b=pathlib.Path(%r);d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/parallel-gt-20261030-r2');d.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(b/m.name).resolve();p.relative_to(b);raw=tar.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw
  if not p.exists():p.write_bytes(raw)
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*gt.py','-q'],cwd=b,capture_output=True)
(d/'unit-tests.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr).decode()
print(json.dumps({'state':'PREPARED','unit_tests':(p.stdout+p.stderr).decode(),'native_inference_frames':0}))
"""%RELEASE
    return call(code,stream.getvalue())


def start():
    code="""import pathlib,subprocess,sys,os,json
b=pathlib.Path(%r);out=pathlib.Path(%r);d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/parallel-gt-20261030-r2')
assert not out.exists() and not (d/'dispatch.json').exists(),'Do not repeat an existing validation'
with (d/'validation.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(b/'tools/benchmark_parallel_gt.py'),str(out)],cwd=b,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'STARTED','pid':p.pid,'output':str(out),'native_inference_frames':0,'full_evaluation_resumed':False};(d/'dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
"""%(RELEASE,OUT)
    return call(code)


def status():
    code="""import pathlib,json,os
out=pathlib.Path(%r);d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/parallel-gt-20261030-r2');r={}
for name in ('progress.json','verification.json'):
 if (out/name).exists():r[name]=json.loads((out/name).read_text())
if (d/'validation.log').exists():r['log_tail']=(d/'validation.log').read_text(errors='replace')[-3000:]
print(json.dumps(r))
"""%OUT
    return call(code)

if __name__=='__main__':print({'prepare':prepare,'start':start,'status':status}[sys.argv[1]]())
