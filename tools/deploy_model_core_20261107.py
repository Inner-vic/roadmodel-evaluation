"""Activate tested core code after queues drain; preserve every existing task."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4'
RELEASE = '/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/model-core-20261107-r5'
CHANGED = ['combined_delivery.py', 'lpai_model_transport.py', 'vision_gt_binding.py',
           'lpai_native_model.py', 'delivery_workflow.py', 'model_report.py', 'vision_preparation.py']
TASK = 'RoadModelEvaluationAgent-20261106-r3'

CODE = '''import pathlib,json,sys,shutil,hashlib,sqlite3,os,time,signal
source=pathlib.Path(SOURCE);release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE)
candidate=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/model-core-20261107/candidate')
proof=json.loads((candidate.parent/'verification.json').read_text());assert proof['state']=='VERIFIED'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(candidate/name)==value for name,value in proof['source_files'].items()),'Candidate changed after tests'
diff=[p.name for p in (candidate/'pcm_eval').glob('*.py') if not (source/'pcm_eval'/p.name).exists() or sha(p)!=sha(source/'pcm_eval'/p.name)]
assert set(diff)==set(CHANGED),'Unexpected pending source changes'
stage.mkdir(parents=True,exist_ok=True)
if not release.exists():shutil.copytree(candidate,release,ignore=shutil.ignore_patterns('__pycache__'))
assert all(sha(release/name)==value for name,value in proof['source_files'].items())
cfgbody=(source/'configs/evaluation.json').read_bytes();assert (release/'configs/evaluation.json').read_bytes()==cfgbody
evalcfg=json.loads(cfgbody);assert evalcfg['execution_scope']=='validation' and evalcfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}
base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path.insert(0,str(base))
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.deployment import status
cfg=load(base);before=status(cfg);assert before['active'] and before['children']['worker']['ready'] and before['children']['bridge']['ready']
with Store(cfg['state_root']).connect() as db:
 for table in ('jobs','arrival_analyses','archive_jobs'):
  if db.execute('select 1 from sqlite_master where name=?',(table,)).fetchone():assert not db.execute("select 1 from "+table+" where state in ('QUEUED','RUNNING','STARTING','PREPARING')").fetchone(),'Shared work active'
state=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3')
with sqlite3.connect(state) as db:
 assert not db.execute("select 1 from tasks where state in ('QUEUED','READY','RUNNING','PREPARING','WAITING_REPORT')").fetchone(),'RoadModel work active'
 assert not db.execute("select 1 from materials where state='READING'").fetchone()
 assert not db.execute("select 1 from publications where state='WRITING'").fetchone()
 with sqlite3.connect(stage/'before-runtime.sqlite3') as dst:db.backup(dst)
plugin=base/'bench_gateway/roadmodel_plugin.py';old=plugin.read_bytes()
assert sha(plugin)=='fb6ae4834b6fd7912fdbb1698c6cd4a6ceeecdc292d5122f63db67dee1e19062','Review concurrent plugin changes'
assert not (stage/'deployment.json').exists(),'Already deployed'
line=("ROOT = Path('"+SOURCE+"')").encode();assert old.count(line)==1
shutil.copy2(plugin,stage/'before-plugin.py')
new=old.replace(line,("ROOT = Path('"+RELEASE+"')").encode());compile(new,str(plugin),'exec')
tmp=plugin.with_suffix('.model-core.tmp');tmp.write_bytes(new);os.replace(tmp,plugin)
worker=before['children']['worker']['pid'];bridge=before['children']['bridge']['pid']
try:
 assert b'bench_gateway.cli' in pathlib.Path('/proc',str(worker),'cmdline').read_bytes()
 os.kill(worker,signal.SIGTERM)
 for _ in range(35):
  time.sleep(1);after=status(cfg)
  assert after['instance']==before['instance'] and after['children']['bridge']['pid']==bridge,'Shared bridge changed'
  if after['children']['worker']['pid']!=worker and after['children']['worker']['ready']:break
 else:raise ValueError('New worker not ready')
except Exception:
 tmp.write_bytes(old);os.replace(tmp,plugin)
 current=status(cfg)['children']['worker']['pid']
 if current!=worker and b'bench_gateway.cli' in pathlib.Path('/proc',str(current),'cmdline').read_bytes():os.kill(current,signal.SIGTERM)
 raise
result={'state':'DEPLOYED','release':RELEASE,'tests':proof['tests'],'changed_modules':CHANGED,
 'worker_ready':after['children']['worker']['ready'],'bridge_pid_unchanged':True,
 'execution_scope':evalcfg['execution_scope'],'validation_selection':evalcfg['validation_selection'],
 'bench_available_unchanged':evalcfg['bench_available'],'native_inference_frames':0,
 'existing_tasks_resumed':False,'report_template_unchanged':True,'plugin_sha256':sha(plugin)}
(stage/'deployment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''


def schedule(action):
    if action not in ('Stop', 'Start'):
        raise ValueError('Unsupported agent lifecycle action')
    # The minute trigger can restart the task while deployment copies files.
    # Stop-ScheduledTask alone can also leave its pythonw child holding the lock.
    script_path = str(ROOT/'.runtime/evaluation-environment-desktop-20261106-r3/tools/evaluation_agent.py')
    quoted = "'"+script_path.replace("'", "''")+"'"
    if action == 'Stop':
        command = ("$ErrorActionPreference='Stop'; Disable-ScheduledTask -TaskName "+repr(TASK)+" | Out-Null; "
                   "Stop-ScheduledTask -TaskName "+repr(TASK)+"; "
                   "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and "
                   "$_.CommandLine -and $_.CommandLine.Contains("+quoted+") } | "
                   "ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction Stop }")
    else:
        command = ("$ErrorActionPreference='Stop'; Enable-ScheduledTask -TaskName "+repr(TASK)+" | Out-Null; "
                   "Start-ScheduledTask -TaskName "+repr(TASK))
    return subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                           command], check=True, capture_output=True,
                          creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    proof = json.loads((ROOT/'evidence/core-development-20261107/verification.json').read_text(encoding='utf-8'))
    assert proof['state'] == 'VERIFIED'
    assert all(hashlib.sha256((ROOT/'pcm_eval'/name).read_bytes()).hexdigest() == proof['source_files']['pcm_eval/'+name] for name in CHANGED)
    pinned = ROOT/'.runtime/evaluation-environment-desktop-20261106-r3'
    backup = ROOT/'.runtime/evaluation-desktop-before-model-core-r5'
    assert not backup.exists(), 'Preserve previous deployment backup'
    schedule('Stop')
    try:
        shutil.copytree(pinned, backup, ignore=shutil.ignore_patterns('__pycache__'))
        result = remote('SOURCE='+repr(SOURCE)+'\nRELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\nCHANGED='+repr(CHANGED)+'\n'+CODE, timeout=65)
        for name in CHANGED:
            shutil.copy2(ROOT/'pcm_eval'/name, pinned/'pcm_eval'/name)
        cfgpath = pinned/'configs/agent.json'
        cfg = json.loads(cfgpath.read_text(encoding='utf-8'));cfg['release'] = RELEASE
        cfgpath.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        result['desktop_release_binding'] = RELEASE
        target = ROOT/'evidence/core-development-20261107/deployment.json'
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        schedule('Start')


if __name__ == '__main__':
    main()
