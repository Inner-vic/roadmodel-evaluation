"""Wait for shared GPU memory, then retry the verified zero-completion OOM batch."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from pcm_eval.common import load_json, save_json, utc_now
from model_space_transport import run, put
from formal_gpu import PROBE, selected_gpu

TASK = ROOT / 'evidence/formal-v322-20261029-01'
OLD = TASK / 'model/batch-0001-a02'
NEW = TASK / 'model/batch-0001-a03'
SPACE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029'
REMOTE_OLD = SPACE + '/occ/batch-0001-a02'
REMOTE_NEW = SPACE + '/occ/batch-0001-a03'


def progress(phase, **fields):
    save_json(TASK / 'model/progress.json', {'state': phase, 'completed_OCC_frames': 100, 'expected_OCC_frames': 8337,
               'updated_at': utc_now(), 'failed_batch': 1, **fields}, overwrite=True)


def main():
    inspect = "import pathlib,json,re,sys;b=pathlib.Path(sys.argv[1]);r=json.loads((b/'execution_receipt.json').read_text());s=(b/'eval-console.log').read_text(errors='replace');assert r['state']=='failed' and r['exit_code']==1 and 'CUDA out of memory' in s;done=[int(x) for x in re.findall(r'(\\d+)/100',s)];assert done and max(done)==0;print(json.dumps({'state':'confirmed_OOM_before_any_completed_frame','completed_frames':0,'receipt':r}))"
    proof = json.loads(run(['python3', '-c', inspect, REMOTE_OLD]).decode())
    proof_file = TASK / 'model/oom-zero-completion-proof.json'
    if proof_file.exists():
        assert load_json(proof_file) == proof, 'Original failed-attempt proof changed'
    else:
        save_json(proof_file, proof)
    mask = selected_gpu(TASK)
    while True:
        d = json.loads(run(['python3','-c',PROBE,mask]).decode())
        progress('WAITING_GPU_MEMORY', free_GPU_GiB=d['free_GiB'], required_GPU_GiB=6, physical_GPU=mask,
                 reason='Shared GPU OOM in batch 1; first 100 completed frames retained; other workloads unchanged')
        if d['free_GiB'] >= 6: break
        time.sleep(60)
    # Only failed, zero-completion input copies are linked into a fresh attempt.
    stage = "import pathlib,sys,shutil,os,json;old=pathlib.Path(sys.argv[1]);new=pathlib.Path(sys.argv[2]);root=pathlib.Path(%r);assert old.resolve().is_relative_to(root.resolve()) and new.resolve().is_relative_to(root.resolve());assert not new.exists();new.mkdir(parents=True);shutil.copytree(old/'input',new/'input',copy_function=os.link);print(json.dumps({'state':'inputs_reused'}))" % SPACE
    run(['python3', '-c', stage, REMOTE_OLD, REMOTE_NEW], timeout=180)
    launch = deepcopy(load_json(OLD / 'launch.json'))
    launch['run_id'] += '-a03'
    launch['environment']['CUDA_VISIBLE_DEVICES'] = mask
    launch['environment'].update(DATASET_PATH=REMOTE_NEW + '/input', DATA_LINK_FILE=REMOTE_NEW + '/input/batch-data-link.json')
    launch['argv'][launch['argv'].index('--work-dir') + 1] = REMOTE_NEW + '/output'
    NEW.mkdir(exist_ok=False)
    save_json(NEW / 'launch.json', launch)
    put(NEW / 'launch.json', REMOTE_NEW + '/launch.json')
    start = "import pathlib,subprocess,sys,json;b=pathlib.Path(sys.argv[1]);assert not (b/'execution_receipt.json').exists();f=(b/'worker-console.log').open('xb');p=subprocess.Popen([sys.executable,%r,str(b)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid}))" % (SPACE + '/native-worker.py')
    native = json.loads(run(['python3', '-c', start, REMOTE_NEW]).decode())
    save_json(TASK / 'model-attempts.json', {'1': 'a03'})
    f = (TASK / 'model-controller-after-oom.log').open('xb')
    p = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'tools/formal_model_controller.py')], cwd=ROOT,
                         stdout=f, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    f.close()
    save_json(TASK / 'model/oom-recovery-dispatch.json', {'native': native, 'controller_pid': p.pid, 'repeated_completed_frames': 0,
               'failed_attempt_preserved': REMOTE_OLD, 'new_attempt': REMOTE_NEW})
    print(json.dumps({'state': 'RESUMED', 'retained_frames': 100, 'native': native}), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as e:
        progress('NEEDS_REVIEW', error_type=type(e).__name__, error=str(e)[:400]); raise
