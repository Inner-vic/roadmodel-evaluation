"""Isolated Linux regression and skill validation; no platform mutations."""
import io
import json
from pathlib import Path
import tarfile
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
MODULES=['tests.test_installation','tests.test_installation_cli','tests.test_space_api',
    'tests.test_notebook_sdk_worker',
    'tests.test_environment_flow','tests.test_evaluation','tests.test_dataset_mount',
    'tests.test_delivery_workflow','tests.test_delivery_combined','tests.test_publication',
    'tests.test_strict_model_template']
CODE='''import pathlib,sys,tarfile,io,shutil,ast,json,unittest,hashlib,subprocess,importlib.util
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/initialization-20261109');stage.mkdir(parents=True,exist_ok=True)
candidate=stage/'candidate'
if not candidate.exists():shutil.copytree('/mnt/data/bench-eval/service/roadmodel/evaluation-space-api-20261107-r6',candidate,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();p=(candidate/member.name).resolve();p.relative_to(candidate);body=tar.extractfile(member).read()
  if p.suffix=='.py':ast.parse(body)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
sys.path.insert(0,str(candidate));suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
with (stage/'tests.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
skill={'state':'UNAVAILABLE_PYYAML'}
if importlib.util.find_spec('yaml'):
 p=subprocess.run([sys.executable,str(candidate/'tools/skill_quick_validate.py'),str(candidate/'skills/roadmodel-space-preparation')],capture_output=True,text=True,timeout=20)
 skill={'state':'VERIFIED' if p.returncode==0 else 'FAILED','output':(p.stdout+p.stderr)[-1200:]}
proof={'state':'VERIFIED' if result.wasSuccessful() and skill['state']=='VERIFIED' else 'FAILED','tests':result.testsRun,
 'failures':len(result.failures),'errors':len(result.errors),'skill':skill,'candidate':str(candidate),
 'platform_create_calls_by_tests':0,'production_changed':False,'new_inference_frames':0,
 'files':{str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('pcm_eval','tests') for p in (candidate/folder).glob('*.py')},
 'failed_cases':[{'test':str(c),'trace':t[-1800:]} for c,t in result.failures+result.errors]}
(stage/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

if __name__=='__main__':
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for folder in ('pcm_eval','tests'):
            for p in (ROOT/folder).glob('*.py'):archive.add(p,arcname=p.relative_to(ROOT).as_posix())
        for name in ['tools/evaluation_agent.py','tools/evaluation_space_actor.py','tools/initialize_evaluation_environment.py','tools/notebook_sdk_worker.py',
            'skills/roadmodel-space-preparation/SKILL.md','schemas/installation-browser-result.schema.json']:
            archive.add(ROOT/name,arcname=name)
        archive.add('D:/codex/skills/.system/skill-creator/scripts/quick_validate.py',arcname='tools/skill_quick_validate.py')
    proof=remote('MODULES='+repr(MODULES)+'\n'+CODE,buf.getvalue(),timeout=110)
    (ROOT/'evidence/initialization-20261109/verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in proof.items() if k!='files'},ensure_ascii=True))
    raise SystemExit(proof['state']!='VERIFIED')
