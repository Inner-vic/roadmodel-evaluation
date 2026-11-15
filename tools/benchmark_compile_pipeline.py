"""Reserve and dispatch exactly six new frames; no full-run resume operation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

ROOT = Path(__file__).resolve().parents[1]
OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-pipeline-timing-20261030-r1'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-pipeline-timing-20261030-r1'


def prepare():
    code = """import pathlib,sys,json,hashlib,subprocess,py_compile
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-memory-gt-20261030-r1');sys.path.insert(0,str(release))
from pcm_eval.common import load_json,save_json,digest
import formal_compile_remote as c
out=pathlib.Path(%r);deploy=pathlib.Path(%r)
assert not out.exists() and not deploy.exists(),'Use the existing attempt; never repeat native inference'
ready=load_json(pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-memory-gt-20261030-r1/candidate-ready.json'))
assert ready['state']=='VERIFIED_INACTIVE_CANDIDATE' and digest(release/'formal_compile_remote.py')==ready['entrypoint_sha256']
assert load_json(c.c.ROOT/'progress.json')['state']=='STOPPED_AT_BOUNDARY' and load_json(c.c.ROOT/'progress.json')['completed_frames']==2151
assert (c.FULL/'STOP_AFTER_FRAME').exists() and not (c.c.ROOT/'controller.lock').exists()
indices=[2152,2153,2154,2163,2164,2165]
for i in indices:
 assert not (c.c.ROOT/'frames'/('frame_%%06d'%%i)).exists(),'Selected frame has prior work'
 assert not (c.old.BASE/'external-bench-leases'/('pcm-v322-subnets-full-%%06d-optv1'%%i)).exists(),'Selected frame has prior native lease'
q=json.loads(subprocess.run([str(c.old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,timeout=20,check=True).stdout)
assert not q.get('active') and not q.get('queued') and not q.get('waiting') and not (c.old.RUNNER/'locks/bench.lock').exists()
out.mkdir(parents=True);deploy.mkdir(parents=True)
source=sys.stdin.buffer.read();(deploy/'benchmark.py').write_bytes(source);py_compile.compile(str(deploy/'benchmark.py'),doraise=True)
r={'schema':'roadmodel.bounded-pipeline-benchmark/v1','indices':indices,'frames':[c.FRAMES[i] for i in indices],
 'checkpoint_sha256':digest(c.c.ROOT/'checkpoint.npz'),'progress_sha256':digest(c.c.ROOT/'progress.json'),
 'stop_marker_sha256':digest(c.FULL/'STOP_AFTER_FRAME'),'frozen_request_sha256':digest(c.FULL/'request.json'),
 'benchmark_sha256':digest(deploy/'benchmark.py'),'maximum_native_frames':6,'full_evaluation_resumed':False,
 'candidate_ready_sha256':digest(pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-memory-gt-20261030-r1/candidate-ready.json'))}
save_json(out/'reservation.json',r);print(json.dumps(r))
""" % (OUT, DEPLOY)
    return call(code, (ROOT / 'tools/benchmark_compile_pipeline_remote.py').read_bytes())


def start():
    return call("""import pathlib,subprocess,sys,json,os,hashlib
out=pathlib.Path(%r);deploy=pathlib.Path(%r);r=json.loads((out/'reservation.json').read_text())
assert hashlib.sha256((deploy/'benchmark.py').read_bytes()).hexdigest()==r['benchmark_sha256']
assert not (deploy/'dispatch.json').exists() and not (out/'started.json').exists(),'Unknown or completed attempt must never be replayed'
with (deploy/'dispatch.json').open('x') as f:json.dump({'state':'DISPATCHING','maximum_native_frames':6},f)
with (deploy/'benchmark.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(deploy/'benchmark.py'),str(out)],cwd=deploy,
  env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'BOUNDED_VALIDATION_STARTED','pid':p.pid,'maximum_native_frames':6,'full_evaluation_resumed':False}
(deploy/'dispatch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
""" % (OUT, DEPLOY))


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);deploy=pathlib.Path(%r);result={}
for name in ('progress.json','failure.json','verification.json'):
 if (out/name).exists():result['historical_failure' if name=='failure.json' and (out/'recovery.json').exists() else name]=json.loads((out/name).read_text())
result['log_tail']=(deploy/'benchmark.log').read_text(errors='replace')[-2300:] if (deploy/'benchmark.log').exists() else ''
print(json.dumps(result))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'prepare':prepare, 'start':start, 'status':status}[sys.argv[1]]())
