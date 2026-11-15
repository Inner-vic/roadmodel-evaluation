"""Stage/test/reserve an isolated two-frame experiment without dispatching hardware."""
import io
import json
from pathlib import Path
import tarfile
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
STAGE='/mnt/data/bench-eval/roadmodel/deployment/multiframe-twoframe-20261109-r1'
CODE=r'''import pathlib,json,sys,tarfile,io,shutil,subprocess,hashlib,numpy as np
stage=pathlib.Path(STAGE);source=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-resume-20261109-r1/controller')
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
assert (full/'STOP_AFTER_FRAME').exists() and not (full/'compile/controller.lock').exists()
progress=json.loads((full/'compile/progress.json').read_text());assert progress['state']=='STOPPED_AT_BOUNDARY'
with np.load(full/'compile/checkpoint.npz',allow_pickle=False) as cp:completed=int(cp['completed'])
assert completed==2998==progress['completed_frames']
stage.mkdir(exist_ok=False);code=stage/'code';shutil.copytree(source,code,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for member in archive:
  assert member.isfile() and pathlib.PurePosixPath(member.name).name==member.name
  (code/member.name).write_bytes(archive.extractfile(member).read())
test=subprocess.run([sys.executable,str(code/'test_board_multiframe_subnet.py'),'-v'],cwd=code,capture_output=True,text=True,timeout=40)
(stage/'tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,(test.stdout+test.stderr)[-2500:]
indices=[]
for i in range(completed,completed+64):
 root=full/'compile/frames'/f'frame_{i:06d}'
 if not root.exists():indices.append(i)
 if len(indices)==2:break
assert len(indices)==2
req=json.loads((full/'request.json').read_text());out=stage/'validation';out.mkdir()
value={'state':'RESERVED_NOT_STARTED','indices':indices,'frames':[req['frames'][i] for i in indices],
 'checkpoint_completed':completed,'checkpoint_sha256':hashlib.sha256((full/'compile/checkpoint.npz').read_bytes()).hexdigest(),
 'pause_marker_sha256':hashlib.sha256((full/'STOP_AFTER_FRAME').read_bytes()).hexdigest(),
 'native_previously_executed':False,'new_inference_frames':0,'full_evaluation_resumed':False,'test_count':5,
 'code_sha256':{n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in ('board_multiframe_subnet.py','multiframe_validation_remote.py')}}
(out/'reservation.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
if __name__=='__main__':
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for path in ('tools/board_multiframe_subnet.py','tools/multiframe_validation_remote.py','tests/test_board_multiframe_subnet.py'):
            tar.add(ROOT/path,arcname=Path(path).name)
    value=remote('STAGE='+repr(STAGE)+'\n'+CODE,stream.getvalue(),timeout=80)
    (ROOT/'evidence/multiframe-exploration-20261109/twoframe-reservation.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value))
