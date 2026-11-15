"""Dispatch installed code to LPAI and collect only native logs and result metadata."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile
import time

from .common import digest, load_json, save_json, utc_now
from .planning import ssh_command
from .parsers import collect_model
from .model_aggregate import aggregate_occ

ROOT = Path(__file__).resolve().parents[1]


def completed_heads(plan, native):
    """Collect successful head scopes only, never candidate or failed frames."""
    if native.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('LPAI返回的执行计划不同')
    heads = native.get('executed_frames_by_head', plan['heads'])
    if set(heads) - set(plan['heads']) or heads.get('occ') != plan['heads']['occ']:
        raise ValueError('LPAI返回的已完成范围与冻结计划不同')
    for kind, frames in heads.items():
        allowed = plan.get('vision_candidates', plan['heads'].get(kind, [])) if kind == 'vision' else plan['heads'][kind]
        if frames and frames != allowed:
            raise ValueError('不能收集部分帧或超出本次计划的模型结果')
        status = native.get('head_statuses', {}).get(kind)
        if status and (status['state'] != 'SUCCEEDED' or status['completed_frames'] != len(frames)):
            raise ValueError('模型头的完成状态与返回范围不一致')
    return heads


def call(space, args, payload=None, timeout=90):
    options = {k: space[k] for k in ('user', 'port', 'identity_file')}
    command = ssh_command(space['host'], args, options=options)
    # New LPAI spaces receive a new port. Use the existing preparation
    # transport's TOFU policy; changed known keys still fail verification.
    command[1:1] = ['-o', 'StrictHostKeyChecking=accept-new']
    proc = subprocess.run(command, input=payload,
                          capture_output=True, timeout=timeout)
    if proc.returncode:
        raise ValueError('LPAI原生执行连接失败，需核对原收据：'+proc.stderr.decode(errors='replace')[-300:])
    return proc.stdout


def execute(plan, output, space):
    space = plan['profile'].get('space_access', space)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if plan['profile'].get('environment_bundle'):
        from .lpai_environment import ensure
        ready = ensure(plan['profile']['environment_bundle'], space)
        save_json(output/'environment-ready.json', ready,
                  overwrite=(output/'environment-ready.json').exists())
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for folder, pattern in [('pcm_eval', '*.py'), ('schemas', 'single_eval.schema.json')]:
            for path in (ROOT/folder).glob(pattern):
                tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    archive = stream.getvalue()
    runtime = '/lpai/roadmodel-evaluation/runtimes/'+hashlib.sha256(archive).hexdigest()[:20]
    remote = '/lpai/roadmodel-evaluation/runs/'+plan['run_id']
    installer = '''import pathlib,sys,tarfile,io,hashlib
root=pathlib.Path(sys.argv[1]);raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();path=(root/member.name).resolve();path.relative_to(root)
  body=tar.extractfile(member).read();path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==body
  else:path.write_bytes(body)
print('INSTALLED')'''
    started = utc_now()
    call(space, ['python3', '-c', installer, runtime, hashlib.sha256(archive).hexdigest()], archive)
    writer = '''import pathlib,sys,json
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);raw=sys.stdin.buffer.read();p=root/'request.json'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print('BOUND')'''
    call(space, ['python3', '-c', writer, remote], json.dumps(plan).encode())
    dispatch_file = output/'dispatch.json'
    if not dispatch_file.exists():
        save_json(dispatch_file, {'state': 'DISPATCHING', 'plan_sha256': plan['plan_sha256'], 'remote': remote})
        starter = '''import pathlib,subprocess,sys,json,os
root=pathlib.Path(sys.argv[1]);runtime=sys.argv[2];intent=root/'driver.json'
if intent.exists():raise ValueError('Prior native dispatch requires reconciliation')
intent.write_text(json.dumps({'state':'STARTING','plan_sha256':sys.argv[3]}))
code="import sys,json,pathlib;sys.path.insert(0,sys.argv[1]);from pcm_eval.lpai_native_model import run;p=pathlib.Path(sys.argv[2]);run(json.loads((p/'request.json').read_text()),p)"
with (root/'driver.log').open('xb') as log:child=subprocess.Popen([sys.executable,'-c',code,runtime,str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
intent.write_text(json.dumps({'state':'STARTED','pid':child.pid,'plan_sha256':sys.argv[3]}));print(intent.read_text())'''
        value = json.loads(call(space, ['python3', '-c', starter, remote, runtime, plan['plan_sha256']]))
        save_json(dispatch_file, {**value, 'remote': remote}, overwrite=True)
    elif load_json(dispatch_file).get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('原派发计划不同，不重复执行')
    observer = '''import pathlib,sys,json,os
root=pathlib.Path(sys.argv[1]);final=root/'result.json';driver=json.loads((root/'driver.json').read_text());pid=driver.get('pid')
progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
alive=False
if pid:
 try:alive=str(root).encode() in pathlib.Path('/proc',str(pid),'cmdline').read_bytes()
 except OSError:pass
print(json.dumps({'finished':final.exists(),'alive':alive,'progress':progress,'error_log':(root/'driver.log').read_text(errors='replace')[-1500:] if not alive and not final.exists() else ''}))'''
    while True:
        state = json.loads(call(space, ['python3', '-c', observer, remote]))
        save_json(output/'native-progress.json', state, overwrite=(output/'native-progress.json').exists())
        if state['finished']:
            break
        if not state['alive']:
            raise ValueError('LPAI原生任务已退出，保留日志待核验：'+state['error_log'])
        time.sleep(5)
    exporter = '''import pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log')]
assert sum(p.stat().st_size for p in files)<100*1024*1024,'Collect metadata explicitly in bounded batches'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path in files:tar.add(path,arcname=path.relative_to(root).as_posix())'''
    raw = call(space, ['python3', '-c', exporter, remote], timeout=150)
    collected = output/'native'
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        for member in tar:
            if not member.isfile():
                raise ValueError('Unsafe result archive')
            path = (collected/member.name).resolve()
            path.relative_to(collected.resolve())
            body = tar.extractfile(member).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != body:
                raise ValueError('已收集原生结果发生变化')
            if not path.exists():
                path.write_bytes(body)
    native = load_json(collected/'result.json')
    outputs = {}
    for kind, frames in completed_heads(plan, native).items():
        if not frames:
            continue
        files = []
        batch_size = plan['batch_size'] if kind == 'occ' else len(frames)
        for index, start in enumerate(range(0, len(frames), batch_size)):
            folder = collected/kind/f'batch-{index:04d}'
            logs = list((folder/'output').glob('*/*.log'))
            if len(logs) != 1:
                raise ValueError('已回收的原生日志不唯一')
            result = collect_model(folder, plan['profile']['version'], expected_frames=len(frames[start:start+batch_size]), log_file=logs[0])
            path = folder/'recollected-result.json'
            save_json(path, result)
            files.append(path)
        if kind == 'occ':
            merged = output/'aggregation'
            aggregate_occ(files, frames, merged, run_id=plan['run_id'])
            outputs[kind] = str(merged/'normalized-result.json')
        else:
            outputs[kind] = str(files[0])
    return {'state': 'succeeded', 'run_id': plan['run_id'], 'result_file': outputs['occ'],
            'result_sha256': digest(Path(outputs['occ'])), 'started_at': started, 'finished_at': utc_now(),
            'incomplete_heads': native['incomplete_heads'], 'head_result_files': [p for k, p in outputs.items() if k != 'occ'],
            'head_statuses': native.get('head_statuses', {}),
            'head_datasets': native.get('head_datasets', {}),
            'executed_frames_by_head': native.get('executed_frames_by_head', {}),
            'execution_location': 'lpai', 'raw_data_collected': False}
