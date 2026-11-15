"""Prepare (or explicitly start) a fresh, measured model-only full evaluation."""
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from connect_active_entry import call

ROOT = Path(__file__).resolve().parents[1]
ID = 'model-timing-v322-20261102-r1'
TASK = '/mnt/data/bench-eval/roadmodel/formal/' + ID
RELEASE = '/mnt/data/bench-eval/service/roadmodel/' + ID
SPACE_PARENT = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
SPACE = SPACE_PARENT + '/' + ID
OLD_SPACE = SPACE_PARENT + '/formal-20261029'
OLD_TASK = '/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01'


def once(text, before, after):
    assert text.count(before) == 1, repr(before)
    return text.replace(before, after)


def render(name, source):
    if name == 'formal_model_native_remote.py':
        source = once(source, 'import time\n', 'import time\nfrom model_rerun_timing import span\n')
        source = once(source, 'try:\n    env =', "timing_sink = rd / 'native-timing.jsonl'\ntry:\n    env =")
        start = source.index('    for name, expected in ')
        end = source.index("    pre = r'''", start)
        source = source[:start] + "    with span(timing_sink, 'weights_and_input_validation'):\n" + ''.join('    '+line if line.strip() else line for line in source[start:end].splitlines(True)) + source[end:]
        before = "    with (rd / 'preflight-console.log').open('w') as log:\n        p = subprocess.run([sys.executable, '-c', pre, str(rd)], env=env, stdout=log, stderr=subprocess.STDOUT)\n    if p.returncode: raise RuntimeError('Native preflight failed')"
        source = once(source, before, "    with span(timing_sink, 'strict_model_and_loader_preflight'):\n" + '\n'.join('    '+line for line in before.splitlines()))
        before = "    with (rd / 'eval-console.log').open('w') as log:\n        p = subprocess.run(launch['argv'], cwd=launch['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT)"
        source = once(source, before, "    with span(timing_sink, 'native_eval_process'):\n" + '\n'.join('    '+line for line in before.splitlines()))
    elif name in ('formal_model_controller.py', 'formal_vision_controller.py'):
        source = once(source, "SPACE = '"+OLD_SPACE+"'", 'SPACE = '+repr(SPACE))
        if name == 'formal_model_controller.py':
            assert source.count("SPACE.rsplit('/formal-20261029', 1)[0]") == 2
            source = source.replace("SPACE.rsplit('/formal-20261029', 1)[0]", repr(SPACE_PARENT))
            source = once(source, "run_id=f'model-v322-full-occ-{index:04d}'", 'run_id=f"{request[\'task_id\']}-occ-{index:04d}"')
        else:
            source = once(source, "run_id='model-v322-full-vision-20261029'", "run_id=request['task_id']+'-vision'")
            source = once(source, "DATA_LINK_FILE=SPACE + '/vision-input/full-data-link.json'", "DATA_LINK_FILE="+repr(OLD_SPACE+'/vision-input/full-data-link.json'))
            source = once(source, "DATASET_PATH=SPACE + '/vision-input/DataReorganized-VisionHead,' + SPACE + '/vision-input/DataReorganized-VisionHead-standard'", 'DATASET_PATH='+repr(OLD_SPACE+'/vision-input/DataReorganized-VisionHead,'+OLD_SPACE+'/vision-input/DataReorganized-VisionHead-standard'))
    ast.parse(source, filename=name)
    return source


def bundle():
    files = {}
    for name in ['formal_model_controller.py', 'formal_vision_controller.py', 'formal_model_native_remote.py']:
        files[name] = render(name, (ROOT/'tools'/name).read_text(encoding='utf8')).encode()
    for name in ['model_rerun_timing.py', 'timed_model_rerun.py']:
        files[name] = (ROOT/'tools'/name).read_bytes()
    files['vision-cache-manifest.json']=(ROOT/'evidence/formal-v322-20261029-01/vision-files.json').read_bytes()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, body in files.items():
            item = tarfile.TarInfo(('' if name.endswith('.json') else 'tools/')+name); item.size = len(body)
            tar.addfile(item, io.BytesIO(body))
    return stream.getvalue()


def prepare():
    code = '''import pathlib,json,shutil,tarfile,io,sys,hashlib,datetime,ast
release=pathlib.Path(%r);task=pathlib.Path(%r);old=pathlib.Path(%r)
assert not task.exists(),'Inspect existing rerun; never overwrite results'
base=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/resume-20261030')
compile_root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pause={n:sha(compile_root/n) for n in ['STOP_AFTER_FRAME','compile/checkpoint.npz','compile/progress.json']}
assert json.loads((compile_root/'compile/progress.json').read_text())['state']=='STOPPED_AT_BOUNDARY'
assert json.loads((old/'model/progress.json').read_text())['state']=='OCC_SUCCEEDED'
assert json.loads((old/'vision/progress.json').read_text())['state']=='SUCCEEDED'
archive=pathlib.Path(%r);raw=archive.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r
if not release.exists():shutil.copytree(base,release)
else:
 assert release.is_dir() and not release.is_symlink() and release.name==%r
 # Reconcile the inspected incomplete code-only copy. No task/results exist.
 assert all(sha(release/p.relative_to(base))==sha(p) for p in (base/'pcm_eval').glob('*.py'))
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for member in tar:
  p=release/member.name;assert member.isfile() and p.resolve().is_relative_to(release)
  body=tar.extractfile(member).read()
  if member.name.endswith('.py'):ast.parse(body,filename=member.name)
  if p.exists():
   original=base/member.name
   assert p.read_bytes()==body or (original.is_file() and p.read_bytes()==original.read_bytes()),'Unexpected existing release content'
  p.write_bytes(body)
report_base=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
for relative in ['pcm_eval/model_aggregate.py','schemas/single_eval.schema.json','configs/report_template.json']:
 source=report_base/relative;target=release/relative;assert source.is_file(),'Missing frozen report dependency'
 target.parent.mkdir(parents=True,exist_ok=True)
 assert not target.exists() or target.read_bytes()==source.read_bytes(),'Report dependency differs; reconcile before inference'
 if not target.exists():shutil.copy2(source,target)
for folder in ['model','vision']: (task/folder).mkdir(parents=True)
req=json.loads((old/'request.json').read_text());req.update(task_id=%r,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='PREPARED_FRESH_MODEL_ONLY',training=False)
req.pop('compile_request',None);req['rerun_of']=str(old);req['old_result_reuse']=False
(task/'request.json').write_text(json.dumps(req,indent=2))
shutil.copy2(release/'vision-cache-manifest.json',task/'vision-cache-manifest.json')
assert len(req['frames'])==8337 and len(req['vision']['frames'])==11543
data=json.loads((old/'vision-data-state.json').read_text());assert data['state']=='READY'
(task/'vision-data-state.json').write_text(json.dumps(data))
old_receipts=[old/'model'/f'batch-{i:04d}-{json.loads((old/"model-attempts.json").read_text()).get(str(i),"a02")}'/'normalized-result.json' for i in range(84)]+[old/'vision/normalized-result.json']
assert all(p.is_file() for p in old_receipts)
archive={'state':'ARCHIVED_EXCLUDED_FROM_FRESH_RUN','new_task_id':req['task_id'],'retained_original_paths':[{'path':str(p),'sha256':sha(p)} for p in old_receipts],'reason':'User requested fresh model rerun for actual end-to-end timing; original immutable evidence preserved; no old result imported'}
(task/'prior-results-archive.json').write_text(json.dumps(archive,indent=2))
report_config=json.loads((old/'full-report-coordinator.json').read_text())
baseline=report_config['report']['baseline_model'];assert pathlib.Path(baseline).is_file()
cfg={'task_id':req['task_id'],'task_root':str(task),'space_root':%r,'vision_cache_root':%r,'identity_file':'/mnt/data/bench-eval/service/gateway/runtime/roadmodel/model-space.key','baseline_model_report':baseline,'compile_pause_hashes':pause,'compile_root':str(compile_root)}
(task/'run-config.json').write_text(json.dumps(cfg,indent=2))
proof={'state':'PREPARED_NOT_STARTED','task_id':req['task_id'],'release':str(release),'old_results_imported':0,'expected_OCC_frames':8337,'expected_vision_frames':11543,'request_sha256':sha(task/'request.json'),'files':{str(p.relative_to(release)):sha(p) for p in (release/'tools').glob('*.py')},'compile_pause_hashes':pause}
(task/'preparation.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
''' % (RELEASE,TASK,OLD_TASK,upload_bundle()[0],upload_bundle_digest(),ID,ID,SPACE,OLD_SPACE+'/vision-input')
    return call(code, timeout=100)


_UPLOADED=None


def upload_bundle_digest():
    assert _UPLOADED is not None
    return _UPLOADED[1]


def upload_bundle():
    global _UPLOADED
    if _UPLOADED is not None:return _UPLOADED
    payload=bundle();sha=hashlib.sha256(payload).hexdigest()
    folder=ROOT/'inspection'/ID;folder.mkdir(parents=True,exist_ok=True)
    local=folder/('deployment-'+sha+'.tar.gz');local.write_bytes(payload)
    remote='/mnt/data/bench-eval/roadmodel/deployment/'+ID+'-'+sha+'.tar.gz'
    # Use SFTP for the bundle; Windows OpenSSH stdin streaming timed out before EOF.
    batch=folder/'upload.sftp'
    batch.write_text('put "'+local.as_posix()+'" "'+remote+'"\n',encoding='utf8')
    result=subprocess.run([shutil.which('sftp') or 'sftp','-o','BatchMode=yes','-o','ConnectTimeout=15','-b',str(batch),'ubuntu-workstation'],capture_output=True,timeout=100)
    if result.returncode:raise RuntimeError('SFTP bundle upload failed: '+result.stderr.decode(errors='replace')[-1000:])
    _UPLOADED=(remote,sha)
    return _UPLOADED


def start():
    # This is explicit, never automatically retried. No work runs during prepare.
    probe = 'import subprocess,pathlib;print(subprocess.run(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout);assert pathlib.Path('+repr(OLD_SPACE+'/vision-input/full-data-link.json')+').is_file()'
    code = '''import pathlib,json,sys,os,subprocess,datetime,hashlib,shlex
task=pathlib.Path(%r);release=pathlib.Path(%r);cfg=json.loads((task/'run-config.json').read_text())
assert not (task/'dispatch.json').exists() and not (task/'timing-start.json').exists(),'Inspect prior dispatch first'
for name,expected in cfg['compile_pause_hashes'].items():assert hashlib.sha256((pathlib.Path(cfg['compile_root'])/name).read_bytes()).hexdigest()==expected
probe=%r
key=cfg['identity_file'];p=subprocess.run(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','31017','-i',key,'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',probe])],capture_output=True,text=True,timeout=35)
assert p.returncode==0,p.stderr[-1000:];print('GPU_PROBE',p.stdout)
gpus=[]
for line in p.stdout.splitlines():
 if not line.strip():continue
 parts=[int(v.strip()) for v in line.split(',')];assert len(parts)==3
 if parts[1]>=10000 and parts[2]<=5:gpus.append(parts)
assert gpus,'No idle GPU with sufficient free memory; do not touch other processes'
gpu=next((g for g in gpus if g[0]==4),max(gpus,key=lambda g:g[1]))
(task/'model/gpu-selection.json').write_text(json.dumps({'physical_index':gpu[0],'visible_indices':list(range(8)),'selection_reason':'fresh read-only nvidia-smi probe, >=10GB free and <=5 percent utilization','free_MiB_at_dispatch':gpu[1]}))
assert not list((task/'model').glob('batch-*')) and not (task/'vision/normalized-result.json').exists()
env={**os.environ,'PYTHONUNBUFFERED':'1','ROADMODEL_TASK_ROOT':str(task),'ROADMODEL_STAGE_LOCAL':'1','ROADMODEL_IDENTITY':key}
log=(task/'controller.log').open('xb');p=subprocess.Popen([sys.executable,str(release/'tools/timed_model_rerun.py'),str(task/'run-config.json')],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'STARTED','task_id':cfg['task_id'],'pid':p.pid,'dispatched_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_OCC_frames':8337,'fresh_vision_frames':11543,'old_results_imported':0,'compile_paused':True}
(task/'dispatch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
''' % (TASK,RELEASE,probe)
    return call(code, timeout=80)


if __name__ == '__main__':
    action=sys.argv[1]
    if action=='check':
        body=bundle();print(json.dumps({'state':'LOCAL_SOURCE_PARSED','bundle_bytes':len(body)}))
    elif action=='prepare':print(prepare())
    elif action=='start':print(start())
    else:raise ValueError(action)
