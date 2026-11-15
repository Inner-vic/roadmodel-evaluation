"""Stage/test an immutable batched driver, then reserve six new frames only."""
import io
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

ROOT = Path(__file__).resolve().parents[1]
RELEASE = '/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r2'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-batch-20261030-r2'
OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-batch-20261030-r2'


def prepare():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for p in (ROOT / 'pcm_eval').glob('*.py'):archive.add(p, arcname='pcm_eval/' + p.name)
        for name in ('fast_subnet_driver.py', 'board_validate_subnet.py', 'batch_subnet_driver.py', 'board_batch_subnet.py'):
            archive.add(ROOT / 'tools' / name, arcname=name)
        archive.add(ROOT / 'tools/fast_compile_controller.py', arcname='formal_compile_remote.py')
        archive.add(ROOT / 'tools/benchmark_compile_pipeline_remote.py', arcname='tools/benchmark.py')
        for name in ('test_batch_subnets.py', 'test_fast_compile.py', 'test_memory_gt.py', 'test_parallel_gt.py', 'test_reference_prefetch.py'):
            archive.add(ROOT / 'tests' / name, arcname='tests/' + name)
        for name in ('compile_output_semantics.json', 'report_template.json'):
            archive.add(ROOT / 'configs' / name, arcname='configs/' + name)
        archive.add(ROOT / 'schemas/single_eval.schema.json', arcname='schemas/single_eval.schema.json')
    code = """import pathlib,sys,tarfile,io,json,subprocess
b=pathlib.Path(%r);d=pathlib.Path(%r);d.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  p=(b/member.name).resolve();p.relative_to(b);raw=tar.extractfile(member).read();p.parent.mkdir(parents=True,exist_ok=True)
  assert not p.exists() or p.read_bytes()==raw,'Choose a new immutable release'
  if not p.exists():p.write_bytes(raw)
sys.path.insert(0,str(b));from pcm_eval.common import digest,load_json,save_json
previous=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-memory-gt-20261030-r1')
for name in ('memory_gt.py','parallel_gt.py','gt_runtime.py'):assert digest(previous/'pcm_eval'/name)==digest(b/'pcm_eval'/name)
config=previous/'gt-runtime.json'
if (b/'gt-runtime.json').exists():assert (b/'gt-runtime.json').read_bytes()==config.read_bytes()
else:(b/'gt-runtime.json').write_bytes(config.read_bytes())
from pcm_eval.gt_runtime import validated_settings;validated_settings(b/'gt-runtime.json')
proof=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-reference-parallel-20261030-r2/verification.json')
value=load_json(proof);assert value['state']=='VERIFIED' and value['all_selected18_bytes_exact']
assert digest(b/'pcm_eval/reference_workers.py')==digest(pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-reference-parallel-20261030-r2/reference_workers.py'))
reference={'workers':value['selected_workers'],'onnx_fp16_sha256':value['reference_onnx_sha256'],'verification_path':str(proof),'verification_sha256':digest(proof)}
if (b/'reference-runtime.json').exists():assert load_json(b/'reference-runtime.json')==reference
else:save_json(b/'reference-runtime.json',reference)
from pcm_eval.reference_prefetch import settings;settings(b/'reference-runtime.json',value['frozen_request_sha256'])

production=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
for name in ('configs/compile_output_semantics.json','configs/report_template.json','schemas/single_eval.schema.json'):
 assert digest(b/name)==digest(production/name),'Original report contract changed'
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-q'],cwd=b,capture_output=True)
(d/'unit-tests.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr).decode()
receipt={'state':'STAGED_NOT_STARTED','source':{str(p.relative_to(b)):digest(p) for p in b.rglob('*') if p.is_file() and p.suffix in ('.py','.json')},'unit_tests':(p.stdout+p.stderr).decode(),'native_inference_frames':0}
if not (d/'staging.json').exists():save_json(d/'staging.json',receipt)
else:assert load_json(d/'staging.json')['source']==receipt['source']
print(json.dumps({'state':receipt['state'],'unit_tests':receipt['unit_tests']}))
""" % (RELEASE, DEPLOY)
    return call(code, stream.getvalue())


def reserve():
    return call("""import pathlib,sys,json,subprocess
b=pathlib.Path(%r);d=pathlib.Path(%r);out=pathlib.Path(%r);sys.path.insert(0,str(b))
from pcm_eval.common import load_json,save_json,digest
import formal_compile_remote as c
assert not out.exists() and not (d/'dispatch.json').exists(),'Never replay an existing attempt'
for name,h in load_json(d/'staging.json')['source'].items():assert digest(b/name)==h
assert load_json(c.c.ROOT/'progress.json')['state']=='STOPPED_AT_BOUNDARY' and load_json(c.c.ROOT/'progress.json')['completed_frames']==2151
assert (c.FULL/'STOP_AFTER_FRAME').exists() and not (c.c.ROOT/'controller.lock').exists()
indices=[2161,2162,2172,2173,2174,2175]
for i in indices:
 assert not (c.c.ROOT/'frames'/('frame_%%06d'%%i)).exists(),'Previously prepared/executed slot'
 assert not (c.old.BASE/'external-bench-leases'/('pcm-v322-subnets-full-%%06d-optv1'%%i)).exists(),'Previously attempted native slot'
q=json.loads(subprocess.run([str(c.old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,timeout=20,check=True).stdout)
assert not q.get('active') and not q.get('queued') and not q.get('waiting') and not (c.old.RUNNER/'locks/bench.lock').exists()
out.mkdir(parents=True)
r={'indices':indices,'frames':[c.FRAMES[i] for i in indices],'checkpoint_sha256':digest(c.c.ROOT/'checkpoint.npz'),
 'progress_sha256':digest(c.c.ROOT/'progress.json'),'stop_marker_sha256':digest(c.FULL/'STOP_AFTER_FRAME'),
 'frozen_request_sha256':digest(c.FULL/'request.json'),'benchmark_sha256':digest(b/'tools/benchmark.py'),
 'maximum_native_frames':6,'full_evaluation_resumed':False}
save_json(out/'reservation.json',r);print(json.dumps(r))
""" % (RELEASE, DEPLOY, OUT))


def start():
    return call("""import pathlib,sys,subprocess,json,os
b=pathlib.Path(%r);d=pathlib.Path(%r);out=pathlib.Path(%r);sys.path.insert(0,str(b))
from pcm_eval.common import load_json,digest
assert digest(b/'tools/benchmark.py')==load_json(out/'reservation.json')['benchmark_sha256']
assert not (out/'started.json').exists()
with (d/'dispatch.json').open('x') as f:json.dump({'state':'DISPATCHING','maximum_native_frames':6},f)
with (d/'validation.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(b/'tools/benchmark.py'),str(out)],cwd=b,
  env={**os.environ,'PYTHONUNBUFFERED':'1','ROADMODEL_PIPELINE_RELEASE':str(b),'ROADMODEL_PIPELINE_BATCH':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'BOUNDED_BATCH_VALIDATION_STARTED','pid':p.pid,'maximum_native_frames':6,'full_evaluation_resumed':False}
(d/'dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
""" % (RELEASE, DEPLOY, OUT))


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);d=pathlib.Path(%r);r={}
for name in ('progress.json','failure.json','verification.json'):
 if (out/name).exists():r[name]=json.loads((out/name).read_text())
r['log_tail']=(d/'validation.log').read_text(errors='replace')[-2800:] if (d/'validation.log').exists() else ''
print(json.dumps(r))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'prepare':prepare, 'reserve':reserve, 'start':start, 'status':status}[sys.argv[1]]())
