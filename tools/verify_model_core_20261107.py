"""Verify core changes in an isolated workstation checkout; no model inference."""
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
TARGET = '/mnt/data/bench-eval/roadmodel/inspection/model-core-20261107/candidate'
MODULES = ['tests.test_vision_preparation', 'tests.test_vision_gt_binding',
           'tests.test_lpai_native_model', 'tests.test_dataset_mount', 'tests.test_evaluation',
           'tests.test_delivery_workflow', 'tests.test_delivery_combined', 'tests.test_environment_flow',
           'tests.test_publication', 'tests.test_strict_model_template', 'tests.test_model_report']


def main():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for folder in ('pcm_eval', 'tests', 'tools'):
            for path in (ROOT/folder).glob('*.py'):
                archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    code = '''import pathlib,json,sys,shutil,tarfile,io,ast,unittest,hashlib
candidate=pathlib.Path(TARGET);base=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
if not candidate.exists():shutil.copytree(base,candidate,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile();p=(candidate/m.name).resolve();p.relative_to(candidate);body=tar.extractfile(m).read();ast.parse(body)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
sys.path.insert(0,str(candidate));suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
stage=candidate.parent
with (stage/'tests.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
receipt={'state':'VERIFIED' if result.wasSuccessful() else 'FAILED','tests':result.testsRun,
 'failures':len(result.failures),'errors':len(result.errors),'modules':MODULES,'candidate':str(candidate),
 'source_files':{str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('pcm_eval','tests','tools') for p in (candidate/d).glob('*.py')},
 'new_inference_frames':0,'bench_contacted':False,'production_changed':False,
 'failed_cases':[{'test':str(case),'trace':trace[-2500:]} for case,trace in result.failures+result.errors]}
(stage/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = remote('TARGET='+repr(TARGET)+'\nMODULES='+repr(MODULES)+'\n'+code, stream.getvalue(), timeout=150)
    out = ROOT/'evidence/core-development-20261107';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_files'}))
    raise SystemExit(result['state'] != 'VERIFIED')


if __name__ == '__main__':
    main()
