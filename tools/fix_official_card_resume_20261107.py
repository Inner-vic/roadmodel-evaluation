"""Verify the startup readback fix in isolation, then resume the same pre-inference card."""
import io
import json
from pathlib import Path
import shutil
import tarfile
from deploy_environment_flow import remote
from deploy_model_core_20261107 import schedule

ROOT=Path(__file__).resolve().parents[1]
TASK=json.loads((ROOT/'evidence/official-card-e2e-20261107/reservation.json').read_text())['task']
FILES=['pcm_eval/space_lifecycle.py','tests/test_space_api.py']
CODE='''import pathlib,sys,json,tarfile,io,shutil,ast,unittest,hashlib,time
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107');candidate=stage/'resume-candidate'
assert not candidate.exists();shutil.copytree(release,candidate,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.name in FILES and m.isfile();body=tar.extractfile(m).read();ast.parse(body);(candidate/m.name).write_bytes(body)
sys.path.insert(0,str(candidate));suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_space_api','tests.test_environment_flow','tests.test_evaluation'])
with (stage/'resume-tests.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful(),[(str(c),t[-1800:]) for c,t in result.failures+result.errors]
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation',config_file=release/'configs/evaluation.json');run=r.root/'runs'/TASK
assert not (run/'workflow.json').exists()
backup=stage/'before-resume-patch';backup.mkdir()
for name in FILES:
 p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(release/name,p);shutil.copy2(candidate/name,release/name)
with r.connect() as db:
 task=dict(db.execute('select * from tasks where id=?',(TASK,)).fetchone());material=dict(db.execute('select * from materials where task=?',(TASK,)).fetchone())
 assert task['state']=='BLOCKED' and material['state']=='PREPARED'
 (backup/'blocked-material.json').write_text(json.dumps(material,indent=2))
 db.execute("update materials set state='QUEUED',updated=? where task=?",(time.time(),TASK))
 db.execute("update tasks set state='PREPARING',notice=?,sent_digest='',updated=? where id=?",('官方启动已确认，接续实际挂载核验',time.time(),TASK))
proof={'state':'VERIFIED_AND_RESUMED','task':TASK,'tests':result.testsRun,'platform_start_requests_sent_by_fix':0,'inference_replayed':0,'files':{n:hashlib.sha256((release/n).read_bytes()).hexdigest() for n in FILES}}
(stage/'resume-patch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

if __name__=='__main__':
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name in FILES:archive.add(ROOT/name,arcname=name)
    result=remote('TASK='+repr(TASK)+'\nFILES='+repr(FILES)+'\n'+CODE,buf.getvalue(),timeout=100)
    pinned=ROOT/'.runtime/evaluation-environment-desktop-20261106-r3'
    for name in FILES:
        (pinned/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,pinned/name)
    (ROOT/'evidence/official-card-e2e-20261107/resume-patch.json').write_text(json.dumps(result,indent=2))
    schedule('Start')
    print(json.dumps(result))
