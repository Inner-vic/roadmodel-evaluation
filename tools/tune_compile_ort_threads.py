"""Dispatch a finite CPU-only tuning comparison; no board execution API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-ort-threads-20261030-r1'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-ort-threads-20261030-r1'


def start():
    return call("""import pathlib,sys,subprocess,json,os,py_compile
out=pathlib.Path(%r);d=pathlib.Path(%r)
assert not out.exists() and not d.exists(),'Never repeat the existing comparison'
native=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-batch-20261030-r1/verification.json')
assert json.loads(native.read_text())['state']=='VERIFIED','Do not interfere with active native timing'
d.mkdir(parents=True);program=d/'compare.py';program.write_bytes(sys.stdin.buffer.read());py_compile.compile(str(program),doraise=True)
with (d/'comparison.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(program),str(out)],cwd=d,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'CPU_ONLY_COMPARISON_STARTED','pid':p.pid,'new_frames':2,'candidate_forward_passes':4,'native_inference_frames':0,'full_evaluation_resumed':False}
(d/'dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
""" % (OUT, DEPLOY), Path(__file__).with_name('tune_compile_ort_threads_remote.py').read_bytes())


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);d=pathlib.Path(%r);r={}
for name in ('progress.json','verification.json'):
 if (out/name).exists():
  value=json.loads((out/name).read_text())
  for row in value.get('cases',[]):row.pop('frames',None)
  r[name]=value
r['log_tail']=(d/'comparison.log').read_text(errors='replace')[-1800:]
print(json.dumps(r))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'start':start, 'status':status}[sys.argv[1]]())
