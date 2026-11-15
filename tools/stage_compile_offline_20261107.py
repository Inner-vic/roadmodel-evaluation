"""Stage and verify a candidate without changing Bot routing or touching the board."""
import io
import json
from pathlib import Path
import tarfile

from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
SUITE=('tests.test_compile_execution','tests.test_batch_subnets','tests.test_fast_compile',
       'tests.test_evaluation','tests.test_delivery_combined','tests.test_delivery_workflow',
       'tests.test_subnet_contract','tests.test_lpai_native_model','tests.test_environment_flow',
       'tests.test_compile_report_atomic')

REMOTE='''import pathlib,sys,json,io,tarfile,shutil,subprocess,hashlib,time
candidate=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-offline-20261107-r4');stage.mkdir(parents=True,exist_ok=True)
assert not (stage/'deployment.json').exists(),'Release is active; create a new candidate instead of modifying it'
checkpoint=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01/compile/checkpoint.npz')
plugin=pathlib.Path('/mnt/data/bench-eval/service/gateway/bench_gateway/roadmodel_plugin.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={'checkpoint':sha(checkpoint),'plugin':sha(plugin)}
if not candidate.exists():shutil.copytree(NEW,candidate,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for member in archive:
  assert member.isfile();path=(candidate/member.name).resolve();path.relative_to(candidate)
  raw=archive.extractfile(member).read()
  if path.suffix=='.py':compile(raw,str(path),'exec')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
guard=stage/'network-denied';guard.mkdir(exist_ok=True)
(guard/'sitecustomize.py').write_text("import socket\\ndef denied(*args,**kwargs):raise AssertionError('Network disabled during offline compile validation')\\nsocket.socket.connect=denied\\nsocket.socket.connect_ex=denied\\nsocket.create_connection=denied\\n")
environment=dict(__import__('os').environ,PYTHONPATH=str(guard)+':'+str(candidate)+':'+str(candidate/'tools'),ROADMODEL_OFFLINE_VALIDATION='1')
guard_check=subprocess.run([sys.executable,'-c',"import socket;assert getattr(socket.socket.connect,'__name__','')=='denied';assert getattr(socket.create_connection,'__name__','')=='denied'"],cwd=candidate,env=environment,capture_output=True,text=True,timeout=15)
assert guard_check.returncode==0,guard_check.stderr
suite=json.loads((candidate/'offline-suite.json').read_text())
child=subprocess.run([sys.executable,'-m','unittest',*suite,'-q'],cwd=candidate,env=environment,capture_output=True,text=True,timeout=180)
(stage/'tests.stdout.txt').write_text(child.stdout);(stage/'tests.stderr.txt').write_text(child.stderr)
assert child.returncode==0,(child.stdout+child.stderr)[-5000:]
validator=candidate/'tools/validate_compile_scope_offline.py'
validation_id=hashlib.sha256(''.join(sha(p) for folder in ('pcm_eval','tools') for p in sorted((candidate/folder).glob('*.py'))).encode()).hexdigest()[:16]
check=subprocess.run([sys.executable,str(validator),'--release',str(candidate),'--output',str(stage/'scope-check'/validation_id)],cwd=candidate,env=environment,capture_output=True,text=True,timeout=120)
(stage/'scope-check.stdout.txt').write_text(check.stdout);(stage/'scope-check.stderr.txt').write_text(check.stderr)
assert check.returncode==0,(check.stdout+check.stderr)[-5000:]
assert before=={'checkpoint':sha(checkpoint),'plugin':sha(plugin)},'Live checkpoint or Bot routing changed'
receipt={'state':'OFFLINE_VERIFIED','candidate':str(candidate),'production_release_unchanged':True,
         'live_checkpoint_unchanged':True,'bench_contacted':False,'native_inference_frames':0,
         'offline_suite':suite,'tests_summary':child.stderr.strip().splitlines()[-4:],
         'scope_check':json.loads(check.stdout),'source_files':{str(p.relative_to(candidate)):sha(p) for folder in ('pcm_eval','tools') for p in (candidate/folder).glob('*.py')}}
(stage/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''

if __name__=='__main__':
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'pcm_eval').glob('*.py')]
    names += ['tools/'+name for name in ('evaluation_worker.py','evaluation_agent.py','fast_compile_controller.py',
              'fast_subnet_driver.py','batch_subnet_driver.py','board_batch_subnet.py','board_validate_subnet.py',
              'formal_compile_remote.py','validate_compile_scope_offline.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('*.py')]
    names += ['examples/compile_v322_subnets_small.json',
              'evidence/input-contract-review-20261027/release-record.json',
              'evidence/feishu-entry-20261029/v322-inventory.json',
              'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/report-delivery-02/collected/result.json',
              'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/report-delivery-02/collected/cosine_summary.json']
    prefix='evidence/actual-runs/pcm-v7-multiscene-001/'
    names += [prefix+p for p in ('collected/cosine_summary.json','gt-adapted-20261027/result.json',
                                 'gt-adapted-20261027/metrics_board.json','gt-adapted-20261027/metrics_onnx.json')]
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name in names:archive.add(ROOT/name,arcname=name)
        raw=json.dumps(SUITE).encode();item=tarfile.TarInfo('offline-suite.json');item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
    result=remote(REMOTE,stream.getvalue(),timeout=330)
    local=ROOT/'evidence/compile-offline-20261107';local.mkdir(exist_ok=True)
    (local/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'}))
