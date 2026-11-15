"""Run the full native vision set once, after OCC releases the shared GPU."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from pcm_eval.common import load_json, save_json, utc_now, digest
from pcm_eval.parsers import collect_model
from model_space_transport import run, put
from formal_gpu import PROBE, selected_gpu

TASK = Path(os.environ.get('ROADMODEL_TASK_ROOT', ROOT / 'evidence/formal-v322-20261029-01'))
OUT = TASK / 'vision'
SPACE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029'


def state(phase, **fields):
    save_json(OUT / 'progress.json', {'state': phase, 'expected_frames': 11543, 'updated_at': utc_now(), **fields}, overwrite=True)


def main():
    OUT.mkdir(exist_ok=True)
    state('WAITING_OCC_AND_DATA')
    while True:
        occ = load_json(TASK / 'model/progress.json') if (TASK / 'model/progress.json').exists() else {}
        data = load_json(TASK / 'vision-data-state.json') if (TASK / 'vision-data-state.json').exists() else {}
        if occ.get('state') == 'NEEDS_REVIEW' or data.get('state') == 'NEEDS_REVIEW':
            state('WAITING_PREREQUISITE_REVIEW', model_state=occ.get('state'), data_state=data.get('state'))
        if occ.get('state') == 'OCC_SUCCEEDED' and data.get('state') == 'READY': break
        time.sleep(30)
    request = load_json(TASK / 'request.json')
    launch = deepcopy(load_json(ROOT / 'evidence/model-evaluation/remaining-heads-20261027/vision-launch.json'))
    rd = SPACE + '/vision-full'
    launch.update(run_id='model-v322-full-vision-20261029', expected_frames=11543, selected_frames=request['vision']['frames'])
    launch['environment'].update(DATA_LINK_FILE=SPACE + '/vision-input/full-data-link.json',
                                 DATASET_PATH=SPACE + '/vision-input/DataReorganized-VisionHead,' + SPACE + '/vision-input/DataReorganized-VisionHead-standard')
    launch['environment']['CUDA_VISIBLE_DEVICES'] = selected_gpu(TASK)
    launch['argv'] = ['python3', 'eval.py', '--py-config', launch['py_config'], '--work-dir', rd + '/output',
                      '--resume-from', launch['main_checkpoint'], '--classifier-resume', launch['classifier_checkpoint'], '--zip-vis', 'none']
    save_json(OUT / 'launch.json', launch)
    put(OUT / 'launch.json', rd + '/launch.json')
    code = "import pathlib,subprocess,sys,json;b=pathlib.Path(sys.argv[1]);assert not (b/'execution_receipt.json').exists();f=(b/'worker-console.log').open('xb');p=subprocess.Popen([sys.executable,%r,str(b)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid}))" % (SPACE + '/native-worker.py')
    while True:
        mem = json.loads(run(['python3','-c',PROBE,launch['environment']['CUDA_VISIBLE_DEVICES']]).decode())
        if mem['free_GiB'] >= 6: break
        state('WAITING_GPU_MEMORY',physical_GPU=mem['physical_GPU'],free_GPU_GiB=mem['free_GiB'])
        time.sleep(60)
    dispatch = json.loads(run(['python3', '-c', code, rd]).decode())
    save_json(OUT / 'native-dispatch.json', dispatch)
    while True:
        code = "import pathlib,sys,json,re;b=pathlib.Path(sys.argv[1]);p=b/'execution_receipt.json';r=json.loads(p.read_text()) if p.exists() else {'state':'starting'};l=(b/'eval-console.log').read_text(errors='replace')[-12000:] if (b/'eval-console.log').exists() else '';m=re.findall(r'(\\d+)/11543',l);print(json.dumps({'state':r['state'],'observed':int(m[-1]) if m else 0}))"
        status = json.loads(run(['python3', '-c', code, rd]).decode())
        state('RUNNING', native_state=status['state'], completed_frames=status['observed'])
        if status['state'] in ('succeeded', 'failed'): break
        time.sleep(30)
    if status['state'] != 'succeeded': raise RuntimeError('Full vision run failed; preserve outputs')
    code = "import pathlib,sys,tarfile;b=pathlib.Path(sys.argv[1]);files=[p for p in b.rglob('*') if p.is_file() and p.suffix in ('.json','.log')];\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:\n for p in files:t.add(p,arcname=p.relative_to(b).as_posix())"
    raw = run(['python3', '-c', code, rd], timeout=180)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for m in tar:
            p = OUT / m.name
            assert m.isfile() and p.resolve().is_relative_to(OUT.resolve())
            body = tar.extractfile(m).read(); p.parent.mkdir(parents=True, exist_ok=True)
            assert not p.exists() or p.read_bytes() == body
            if not p.exists(): p.write_bytes(body)
    console = (OUT / 'eval-console.log').read_text(errors='replace')
    assert re.search(r'100%[^\r\n]*\b11543/11543\b', console) and 'Total samples used: 11543' in console
    save_json(OUT / 'coverage_receipt.json', {'schema':'roadmodel.native-coverage/v1','state':'verified','run_id':launch['run_id'],
         'expected_frames':11543,'observed_frames':11543,'completed_batches':11543,'batch_size':1,'selected_frames':launch['selected_frames'],'console_sha256':digest(OUT/'eval-console.log'),'loader_preflight_sha256':digest(OUT/'model-preflight.json')}, overwrite=(OUT/'coverage_receipt.json').exists())
    logs = list((OUT / 'output').glob('*/*.log')); assert len(logs) == 1
    result = collect_model(OUT, 'v3.2.2', expected_frames=11543, log_file=logs[0])
    assert result['collection']['status'] == 'complete'
    save_json(OUT / 'normalized-result.json', result)
    state('SUCCEEDED', completed_frames=11543, metric_records=len(result['metrics']))


if __name__ == '__main__':
    try: main()
    except Exception as e:
        state('NEEDS_REVIEW', error_type=type(e).__name__, error=str(e)[:300]); raise
