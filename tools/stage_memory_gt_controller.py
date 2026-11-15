"""Stage the validated controller without restarting any native task or gateway."""
import io
import json
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

ROOT = Path(__file__).resolve().parents[1]
RELEASE = '/mnt/data/bench-eval/service/roadmodel/compile-memory-gt-20261030-r1'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-memory-gt-20261030-r1'
OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-memory-gt-20261030-r1'
PROOF = '/mnt/data/bench-eval/roadmodel/validation/parallel-gt-20261030-r2/verification.json'


def prepare():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for p in (ROOT / 'pcm_eval').glob('*.py'):tar.add(p, arcname='pcm_eval/' + p.name)
        for source, target in (('fast_compile_controller.py', 'formal_compile_remote.py'),
                               ('fast_subnet_driver.py', 'fast_subnet_driver.py'),
                               ('board_validate_subnet.py', 'board_validate_subnet.py'),
                               ('verify_memory_gt_controller.py', 'tools/verify_memory_gt_controller.py')):
            tar.add(ROOT / 'tools' / source, arcname=target)
        for name in ('test_parallel_gt.py', 'test_memory_gt.py', 'test_fast_compile.py'):
            tar.add(ROOT / 'tests' / name, arcname='tests/' + name)
    code = """import pathlib,sys,tarfile,io,hashlib,json,subprocess
b=pathlib.Path(%r);d=pathlib.Path(%r);d.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(b/m.name).resolve();p.relative_to(b);raw=tar.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==raw,'Choose an immutable new release'
  if not p.exists():p.write_bytes(raw)
sys.path.insert(0,str(b));from pcm_eval.common import load_json,save_json,digest
proof=pathlib.Path(%r);verified=load_json(proof);assert verified['state']=='VERIFIED'
reference_kernel=proof.parents[3]/'service/roadmodel/parallel-gt-20261030-r2/pcm_eval/memory_gt.py'
assert digest(reference_kernel)==digest(b/'pcm_eval/memory_gt.py'),'Staged kernel differs from benchmark'
config={'schema':'roadmodel.gt-runtime/v1','backend':'memory_optimized','workers':verified['selected_workers'],
 'threads_per_worker':1,'verification_path':str(proof),'verification_sha256':digest(proof),
 'memory_kernel_sha256':digest(b/'pcm_eval/memory_gt.py'),
 'frozen_request_sha256':digest(pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01/request.json'))}
if (b/'gt-runtime.json').exists():assert load_json(b/'gt-runtime.json')==config
else:save_json(b/'gt-runtime.json',config)
from pcm_eval.gt_runtime import validated_settings;validated_settings(b/'gt-runtime.json')
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-q'],cwd=b,capture_output=True)
(d/'unit-tests.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr).decode()
receipt={'state':'STAGED_NOT_STARTED','release':str(b),'native_inference_frames':0,'full_evaluation_resumed':False,
 'source':{str(p.relative_to(b)):digest(p) for p in b.rglob('*.py')},'config_sha256':digest(b/'gt-runtime.json'),
 'unit_tests':(p.stdout+p.stderr).decode()}
if (d/'staging.json').exists():assert load_json(d/'staging.json')==receipt
else:save_json(d/'staging.json',receipt)
print(json.dumps({'state':receipt['state'],'release':str(b),'unit_tests':receipt['unit_tests']}))
""" % (RELEASE, DEPLOY, PROOF)
    return call(code, stream.getvalue())


def validate():
    code = """import pathlib,subprocess,sys,os,json
b=pathlib.Path(%r);d=pathlib.Path(%r);out=pathlib.Path(%r)
assert not out.exists() and not (d/'validation-dispatch.json').exists(),'Never repeat an unknown validation'
with (d/'validation.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(b/'tools/verify_memory_gt_controller.py'),str(out)],cwd=b,
  env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'GT_ONLY_VALIDATION','pid':p.pid,'native_inference_frames':0,'full_evaluation_resumed':False}
(d/'validation-dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
""" % (RELEASE, DEPLOY, OUT)
    return call(code)


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);d=pathlib.Path(%r)
print(json.dumps({'verification':json.loads((out/'verification.json').read_text()) if (out/'verification.json').exists() else None,
 'log_tail':(d/'validation.log').read_text(errors='replace')[-3500:] if (d/'validation.log').exists() else ''}))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'prepare':prepare, 'validate':validate, 'status':status}[sys.argv[1]]())
