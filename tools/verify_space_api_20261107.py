"""Isolated regression for official API lifecycle; no platform writes."""
import io
import json
from pathlib import Path
import sys
import tarfile

from deploy_environment_flow import remote
from verify_model_core_20261107 import MODULES

ROOT = Path(__file__).resolve().parents[1]
TARGET = '/mnt/data/bench-eval/roadmodel/inspection/space-api-20261107/candidate'
CODE = '''import pathlib,json,sys,shutil,tarfile,io,ast,unittest,hashlib
candidate=pathlib.Path(TARGET);base=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5')
if not candidate.exists():shutil.copytree(base,candidate,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile();p=(candidate/m.name).resolve();p.relative_to(candidate);body=tar.extractfile(m).read();ast.parse(body)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
sys.path.insert(0,str(candidate));suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
stage=candidate.parent
with (stage/'tests.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
proof={'state':'VERIFIED' if result.wasSuccessful() else 'FAILED','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'candidate':str(candidate),'modules':MODULES,'source_files':{str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('pcm_eval','tests','tools') for p in (candidate/d).glob('*.py')},
 'platform_write_requests':0,'new_inference_frames':0,'production_changed':False,
 'failed_cases':[{'test':str(c),'trace':t[-2200:]} for c,t in result.failures+result.errors]}
(stage/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

if __name__ == '__main__':
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for folder in ('pcm_eval', 'tests', 'tools'):
            for p in (ROOT/folder).glob('*.py'):
                archive.add(p, arcname=p.relative_to(ROOT).as_posix())
    result = remote('TARGET='+repr(TARGET)+'\nMODULES='+repr(MODULES+['tests.test_space_api'])+'\n'+CODE, buf.getvalue(), timeout=150)
    out = ROOT/'evidence/sdk-space-20261107';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_files'}, ensure_ascii=True))
    raise SystemExit(result['state'] != 'VERIFIED')
