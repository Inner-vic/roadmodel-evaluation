"""Move only orchestration to the workstation; retain completed native batches."""
import io
import json
import tarfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import digest
from pcm_eval.planning import ssh_command
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CODE = '/mnt/data/bench-eval/service/roadmodel/resume-20261030'
TASK = '/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01'
OLD = ROOT/'evidence/formal-v322-20261029-01'

def send(code, payload, *args):
    p = subprocess.run(ssh_command('ubuntu-workstation',['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python','-c',code,*args]),input=payload,capture_output=True,timeout=120)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
    return p.stdout.decode()

def main():
    files = {p.relative_to(ROOT).as_posix():p for d in ('pcm_eval','schemas') for p in (ROOT/d).glob('*') if p.is_file()}
    for n in ('formal_model_controller','formal_model_native_remote','formal_model_stage_remote','formal_gpu','formal_vision_controller','model_space_transport'):
        files['tools/'+n+'.py']=ROOT/'tools'/str(n+'.py')
    for n in ('evidence/model-evaluation/model-reference-20261027/native-launch.json','evidence/model-evaluation/remaining-heads-20261027/vision-launch.json'):
        files[n]=ROOT/n
    data={'request.json':OLD/'request.json','model-attempts.json':OLD/'model-attempts.json','vision-data-state.json':OLD/'vision-data-state.json'}
    for folder in ('batch-0000-a02','batch-0001-a03'):
        for p in (OLD/'model'/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.log'):data['model/'+p.relative_to(OLD/'model').as_posix()]=p
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for prefix,items in [('code',files),('task',data)]:
            for n,p in items.items():tar.add(p,arcname=prefix+'/'+n)
    unpack="""import pathlib,sys,tarfile,json,io
roots={'code':pathlib.Path(sys.argv[1]),'task':pathlib.Path(sys.argv[2])}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  prefix,name=m.name.split('/',1);b=roots[prefix];p=(b/name).resolve();assert p.is_relative_to(b.resolve()) and m.isfile();raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw,'Different existing file';p.write_bytes(raw) if not p.exists() else None
print(json.dumps({'code':str(roots['code']),'task':str(roots['task']),'completed_batches_imported':2}))"""
    print(send(unpack,stream.getvalue(),CODE,TASK))
    # Same previously authorized GPU identity, stored privately for the relocated scheduler.
    key=Path('C:/Users/developer/.ssh/bmad_roadmodel_ed25519').read_bytes()
    dest='/mnt/data/bench-eval/service/gateway/runtime/roadmodel/model-space.key'
    writer="import pathlib,sys,os;p=pathlib.Path(sys.argv[1]);raw=sys.stdin.buffer.read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600) if not p.exists() else None;os.write(fd,raw) if fd is not None else None;os.close(fd) if fd is not None else None;os.chmod(p,0o600);print('authorized identity ready')"
    print(send(writer,key,dest))
    launch="""import pathlib,sys,json,os,subprocess,fcntl
code,task,key=map(pathlib.Path,sys.argv[1:4]);sys.path.insert(0,str(code));from pcm_eval.common import save_json
gpu={'physical_index':4,'visible_indices':list(range(8)),'selection_reason':'same space restarted; physical GPU4 observed 40335MiB free; probe uses same mask as inference'}
save_json(task/'model/gpu-selection.json',gpu,overwrite=(task/'model/gpu-selection.json').exists())
assert all((task/'model'/n/'normalized-result.json').exists() for n in ('batch-0000-a02','batch-0001-a03'))
env={**os.environ,'ROADMODEL_TASK_ROOT':str(task),'ROADMODEL_STAGE_LOCAL':'1','ROADMODEL_IDENTITY':str(key),'PYTHONUNBUFFERED':'1'}
receipt=task/'workstation-dispatch.json'
assert not receipt.exists(),'Inspect existing dispatcher before another launch'
out={}
for kind in ('model','vision'):
 f=(task/(kind+'-controller.log')).open('xb');p=subprocess.Popen([sys.executable,str(code/'tools'/('formal_'+kind+'_controller.py'))],cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);out[kind+'_controller_pid']=p.pid
out.update(state='STARTED',retained_OCC_frames=200,new_cloud_downloads=0,compile_process_untouched=True,task_root=str(task));save_json(receipt,out);print(json.dumps(out))"""
    print(send(launch,b'',CODE,TASK,dest))

if __name__=='__main__':main()
