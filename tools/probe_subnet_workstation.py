"""Read-only workstation inputs, original Runner lock API, and model audit."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command

code=r'''
import json,pathlib,subprocess
base=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval')
runner=pathlib.Path('/home/lixiang/bench-eval')
q=subprocess.run([str(runner/'bin/bench-eval'),'queue'],capture_output=True,text=True,timeout=15)
queue=json.loads(q.stdout) if q.returncode==0 else {}
out={'queue':{'readable':q.returncode==0,'active':queue.get('active'),'queued_count':len(queue.get('queued',[])),
 'waiting_count':len(queue.get('waiting',[]))},'lock_exists':(runner/'locks/bench.lock').exists(),
 'automation_files':[p.name for p in (runner/'current/automation').iterdir()],
 'source_roots':[str(p) for p in (runner/'current').iterdir() if p.is_dir()],
 'sample_dirs':[str(p) for p in (base/'runs/pcm-v7-multiscene-001').glob('frame_*')][:2],
 'model_files':[{'path':str(p),'size':p.stat().st_size} for p in (base/'package/benchEval-skill-20261020/examples/assets').iterdir() if p.is_file()],
 'existing_frame_files':[]}
for frame in list((base/'runs/pcm-v7-multiscene-001').glob('frame_*'))[:1]:
 out['existing_frame_files']=[str(p.relative_to(frame)) for p in frame.rglob('*') if p.is_file()][:45]
print(json.dumps(out))
'''
if __name__=='__main__':
    proc=subprocess.run(ssh_command('ubuntu-workstation',['python3','-']),input=code.encode(),capture_output=True,timeout=45)
    if proc.returncode:raise RuntimeError(proc.stderr.decode(errors='replace')[-1500:])
    value=json.loads(proc.stdout)
    out=Path('evidence/compile-subnet-review-20261029/workstation-probe.json');out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False,indent=2))
