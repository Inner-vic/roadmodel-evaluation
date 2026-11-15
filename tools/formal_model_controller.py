"""Stream all 8337 OCC frames in native batches; keep every batch result and progress."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from pcm_eval.common import load_json, save_json, utc_now, digest
from pcm_eval.planning import ssh_command
from pcm_eval.parsers import collect_model
from model_space_transport import command, run, put
from formal_gpu import PROBE, selected_gpu

TASK = Path(os.environ.get('ROADMODEL_TASK_ROOT', ROOT / 'evidence/formal-v322-20261029-01'))
OUT = TASK / 'model'
SPACE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029'
WORKSTATION = '/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'


def state(phase, completed, **fields):
    save_json(OUT / 'progress.json', {'state': phase, 'completed_OCC_frames': completed, 'expected_OCC_frames': 8337,
              'updated_at': utc_now(), **fields}, overwrite=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    request = load_json(OUT.parent / 'request.json')
    base = load_json(ROOT / 'evidence/model-evaluation/model-reference-20261027/native-launch.json')
    state('PREPARING', 0)
    put(ROOT / 'tools/formal_model_native_remote.py', SPACE + '/native-worker.py')
    completed = 0
    for index, start in enumerate(range(0, 8337, 100)):
        count = min(100, 8337 - start)
        attempts = load_json(OUT.parent / 'model-attempts.json') if (OUT.parent / 'model-attempts.json').exists() else {}
        suffix = attempts.get(str(index), 'a02')
        rd = SPACE + f'/occ/batch-{index:04d}-{suffix}'
        local = OUT / f'batch-{index:04d}-{suffix}'
        if (local / 'normalized-result.json').exists():
            completed += count; continue
        if (local / 'launch.json').exists():
            launch = load_json(local / 'launch.json')
            assert launch['selected_frames'] == request['frames'][start:start + count]
            check = "import pathlib,sys,json;p=pathlib.Path(sys.argv[1]);r=json.loads((p/'execution_receipt.json').read_text());assert r['state'] in ('preflight','running','succeeded');print(json.dumps({'state':r['state'],'selected_frames':r['selected_frames']}))"
            retained = json.loads(run(['python3', '-c', check, rd]).decode())
            assert retained['selected_frames'] == launch['selected_frames']
            state('RESUMING_RESULT_COLLECTION', completed, batch=index, native_state=retained['state'])
        else:
            state('STAGING_OCC_BATCH', completed, batch=index, batch_frames=count)
            # The workstation is the source of already cached data; no new cloud raw-data download.
            while True:
                check = "import pathlib,json;b=pathlib.Path(%r);r=json.loads((b/'request.json').read_text());ds=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1');d=json.loads((ds/'data_link_26-10-20-1_train859_eval135_v7.json').read_text());sel=r['frames'][%d:%d];print(json.dumps({'ready':all((ds/s/'dataset'/str(f)/next(x['occ_file'] for x in d['eval'][s]['frames'] if x['frame_index']==f)).exists() for s,f in sel),'data_state':json.loads((b/'data-state.json').read_text())['state']}))" % (WORKSTATION, start, start + count)
                check_argv = ['python3', '-c', check]
                if os.environ.get('ROADMODEL_STAGE_LOCAL') != '1':
                    check_argv = ssh_command('ubuntu-workstation', check_argv)
                p = subprocess.run(check_argv, capture_output=True, timeout=60)
                if p.returncode: raise RuntimeError('Workstation GT precheck failed')
                ready = json.loads(p.stdout)
                if ready['ready']: break
                if ready['data_state'] == 'NEEDS_REVIEW': raise RuntimeError('GT preparation failed')
                time.sleep(15)
            local.mkdir(parents=True, exist_ok=True)
            archive = local / 'input-transfer.tar'
            source_argv = ['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python', str(ROOT / 'tools/formal_model_stage_remote.py'), str(start), str(count), str(TASK / 'request.json')]
            if os.environ.get('ROADMODEL_STAGE_LOCAL') != '1':
                source_argv = ssh_command('ubuntu-workstation', [source_argv[0], WORKSTATION + '/code/formal_model_stage_remote.py', str(start), str(count)])
            with archive.open('wb') as f:
                producer = subprocess.run(source_argv, stdout=f, stderr=subprocess.PIPE, timeout=600)
            if producer.returncode:
                (local / 'stage-source.stderr.txt').write_bytes(producer.stderr)
                raise RuntimeError('Model input source failed: ' + producer.stderr.decode(errors='replace')[-500:])
            expected_hash = digest(archive)
            writer = "import pathlib,sys,hashlib,os;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);assert not (p.parent/'execution_receipt.json').exists();q=p.with_suffix('.partial');h=hashlib.sha256();f=q.open('wb');\nfor chunk in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):f.write(chunk);h.update(chunk)\nf.close();actual=h.hexdigest();assert actual==sys.argv[2];os.replace(q,p);print(actual)"
            with archive.open('rb') as f:
                receiver = subprocess.run(command(['python3', '-c', writer, rd + '/input-transfer.tar', expected_hash]), stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if receiver.returncode or receiver.stdout.decode().strip() != expected_hash:
                raise RuntimeError('Verified archive transfer failed (exit '+str(receiver.returncode)+'): ' + receiver.stderr.decode(errors='replace')[-500:])
            unpack = "import pathlib,sys,tarfile,json,hashlib,time;b=pathlib.Path(sys.argv[1]);assert not (b.parent/'execution_receipt.json').exists();\nif b.exists() and not (b/'data-binding.json').exists():b.rename(b.with_name('input-incomplete-'+str(int(time.time()))))\nb.mkdir(parents=True,exist_ok=True)\nwith tarfile.open(b.parent/'input-transfer.tar') as t:\n for m in t:\n  p=b/m.name;assert m.isfile() and p.resolve().is_relative_to(b.resolve());raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw) if not p.exists() else None\nd=json.loads((b/'data-binding.json').read_text());assert all(hashlib.sha256((b/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in d['files']);print(json.dumps({'state':'READY','frames':len(d['selected_frames'])}))"
            ready = json.loads(run(['python3', '-c', unpack, rd + '/input']).decode())
            assert ready['frames'] == count
            assert archive.resolve().is_relative_to(OUT.resolve())
            archive.unlink()
            launch = deepcopy(base)
            launch.update(kind='occ', run_id=f'model-v322-full-occ-{index:04d}', version_name='v3.2.2', expected_frames=count,
                           selected_frames=request['frames'][start:start + count], source_root=SPACE.rsplit('/formal-20261029', 1)[0] + '/source',
                           dataset_version='26-10-20-1', dataset_class='SmartChassisDatasetV7', py_config='config/3-0-0-temporal-dit.py')
            launch.setdefault('source_commit', '9e16e29351d17a726ce6beca6f821de79b7ade40')
            launch['checkpoint_sha256'] = load_json(ROOT / 'evidence/model-evaluation/remaining-heads-20261027/vision-launch.json')['checkpoint_sha256']
            launch['environment']['CUDA_VISIBLE_DEVICES'] = selected_gpu(OUT.parent)
            launch['environment'].update(DATASET_PATH=rd + '/input', DATA_LINK_FILE=rd + '/input/batch-data-link.json', NUM_WORKERS='0', PERSISTENT_WORKERS='False', PIN_MEMORY='False')
            parent = SPACE.rsplit('/formal-20261029', 1)[0]
            launch.update(main_checkpoint=parent + '/weights/epoch_44.pth', classifier_checkpoint=parent + '/weights/epoch_98.pth')
            launch['argv'] = ['python3', 'eval.py', '--py-config', launch['py_config'], '--work-dir', rd + '/output',
                              '--resume-from', launch['main_checkpoint'], '--classifier-resume', launch['classifier_checkpoint'], '--zip-vis', 'none']
            save_json(local / 'launch.json', launch)
            put(local / 'launch.json', rd + '/launch.json')
            code = "import pathlib,subprocess,sys,json;b=pathlib.Path(sys.argv[1]);assert not (b/'execution_receipt.json').exists();f=(b/'worker-console.log').open('xb');p=subprocess.Popen([sys.executable,%r,str(b)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid}))" % (SPACE + '/native-worker.py')
            while True:
                free = json.loads(run(['python3','-c',PROBE,launch['environment'].get('CUDA_VISIBLE_DEVICES','1')]).decode())['free_GiB']
                if free >= 6: break
                state('WAITING_GPU_MEMORY',completed,batch=index,free_GPU_GiB=free,required_GPU_GiB=6)
                time.sleep(60)
            run(['python3', '-c', code, rd])
        while True:
            code = "import pathlib,json,re,sys;b=pathlib.Path(sys.argv[1]);p=b/'execution_receipt.json';r=json.loads(p.read_text()) if p.exists() else {'state':'starting'};l=(b/'eval-console.log').read_text(errors='replace')[-12000:] if (b/'eval-console.log').exists() else '';m=re.findall(r'(\\d+)/%d',l);print(json.dumps({'state':r['state'],'observed_progress':int(m[-1]) if m else 0}))" % count
            status = json.loads(run(['python3', '-c', code, rd]).decode())
            state('RUNNING_OCC', completed, batch=index, batch_frames=count, observed_active_batch_frames=status['observed_progress'], native_state=status['state'])
            if status['state'] in ('succeeded', 'failed'): break
            time.sleep(20)
        if status['state'] != 'succeeded': raise RuntimeError('Native OCC batch failed; preserve all artifacts')
        code = "import pathlib,sys,tarfile;b=pathlib.Path(sys.argv[1]);files=[p for p in b.rglob('*') if p.is_file() and p.suffix in ('.json','.log') and '/input/' not in str(p)];files.append(b/'input/data-binding.json');\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:\n for p in files:t.add(p,arcname=p.relative_to(b).as_posix())"
        raw = run(['python3', '-c', code, rd], timeout=120)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t:
                path = local / m.name
                assert m.isfile() and path.resolve().is_relative_to(local.resolve())
                body = t.extractfile(m).read(); path.parent.mkdir(parents=True, exist_ok=True)
                assert not path.exists() or path.read_bytes() == body
                if not path.exists(): path.write_bytes(body)
        console = (local / 'eval-console.log').read_text(errors='replace')
        assert re.search(r'100%[^\r\n]*\b'+str(count)+'/'+str(count)+r'\b', console)
        save_json(local / 'coverage_receipt.json', {'schema':'roadmodel.native-coverage/v1','state':'verified','run_id':launch['run_id'],'expected_frames':count,'observed_frames':count,'completed_batches':count,'batch_size':1,'selected_frames':launch['selected_frames'],'console_sha256':digest(local/'eval-console.log'),'loader_preflight_sha256':digest(local/'model-preflight.json')}, overwrite=(local/'coverage_receipt.json').exists())
        logs = list((local / 'output').glob('*/*.log')); assert len(logs) == 1
        result = collect_model(local, 'v3.2.2', expected_frames=count, log_file=logs[0])
        assert result['collection']['status'] == 'complete'
        save_json(local / 'normalized-result.json', result)
        # This directory contains only task-created staged copies; original caches stay untouched.
        cleanup = "import pathlib,shutil,sys;p=pathlib.Path(sys.argv[1]).resolve();root=pathlib.Path(%r).resolve();assert p.is_relative_to(root) and p.name=='input' and (p.parent/'execution_receipt.json').exists();shutil.rmtree(p);a=p.parent/'input-transfer.tar';a.unlink() if a.exists() else None" % SPACE
        run(['python3', '-c', cleanup, rd + '/input'])
        completed += count
        state('RUNNING_OCC', completed, completed_batches=index + 1)
    state('OCC_SUCCEEDED', completed, vision_state='WAITING_FULL_VISION_INPUT')


if __name__ == '__main__':
    try: main()
    except Exception as e:
        old = load_json(OUT / 'progress.json') if (OUT / 'progress.json').exists() else {}
        state('NEEDS_REVIEW', old.get('completed_OCC_frames', 0), error_type=type(e).__name__, error=str(e)[:500])
        raise
