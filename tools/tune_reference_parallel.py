"""Run a finite reference-session comparison; it cannot launch board inference."""
import io
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-reference-parallel-20261030-r1'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-reference-parallel-20261030-r1'


def start():
    stream = io.BytesIO(); root = Path(__file__).resolve().parents[1]
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        tar.add(root / 'pcm_eval/reference_workers.py', arcname='reference_workers.py')
        tar.add(root / 'tools/tune_reference_parallel_remote.py', arcname='compare.py')
    return call("""import pathlib,sys,tarfile,io,subprocess,json,os
out=pathlib.Path(%r);d=pathlib.Path(%r);assert not out.exists() and not d.exists(),'Never repeat this experiment'
assert json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-ort-threads-20261030-r1/verification.json').read_text())['state']=='VERIFIED'
d.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(d/m.name).resolve();p.relative_to(d);p.write_bytes(tar.extractfile(m).read())
with (d/'comparison.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(d/'compare.py'),str(out)],cwd=d,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'CPU_ONLY_PARALLEL_COMPARISON_STARTED','pid':p.pid,'fresh_frames':4,'native_inference_frames':0};(d/'dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
""" % (OUT, DEPLOY), stream.getvalue())


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);d=pathlib.Path(%r);r={}
for n in ('progress.json','verification.json'):
 if (out/n).exists():
  x=json.loads((out/n).read_text())
  for c in x.get('cases',[]):c.pop('frames',None)
  r[n]=x
r['log_tail']=(d/'comparison.log').read_text(errors='replace')[-1800:];print(json.dumps(r))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'start':start, 'status':status}[sys.argv[1]]())
