"""Durable full validation loop: frozen frames, original Runner locks, atomic counters."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

BASE = Path('/mnt/data/developer/pcm-roadmodel-eval')
TASK = Path(os.environ.get('ROADMODEL_COMPILE_TASK_ROOT', str(BASE / 'formal-runs/pcm-formal-v322-20261029-01')))
ROOT = TASK / 'compile'
DATA = Path(os.environ.get('ROADMODEL_COMPILE_DATASET_ROOT', str(BASE / 'datasets/smartchassisdata-v3/26-10-20-1')))
PACKAGE = BASE / 'package/benchEval-skill-20261020'
sys.path[:0] = [str(TASK / 'code'), str(PACKAGE)]
import formal_subnet_chain as driver
from numpy_data_pipeline import NumpyBenchDataPipeline
from gt_adapter import Counts, adapt_sparse
from onnx_localagg_backend import localagg_from_gaussian_bins, ort_outputs_to_packup_bins
from eval_metrics_bridge import MetricsAccumulator


def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def sha(p): return driver.sha(p)
def save(p, d): driver.save(Path(p), d)


def main():
    request = load(TASK / 'request.json')
    expected = request['frames']
    assert len(expected) == request['expected_frames'] == 8337
    link = DATA / 'data_link_26-10-20-1_train859_eval135_v7.json'
    assert sha(link) == request['data_link_sha256']
    source_data = load(link)
    ROOT.mkdir(parents=True, exist_ok=True)
    claim = ROOT / 'controller.lock'
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    acc = {b: Counts() for b in ('board', 'onnx')}
    cursor = 0
    if (ROOT / 'checkpoint.npz').exists():
        with np.load(ROOT / 'checkpoint.npz', allow_pickle=False) as cp:
            cursor = int(cp['completed'])
            for b in acc:
                acc[b].confusion = cp[b + '_confusion'].copy()
                acc[b].ds = cp[b + '_ds'].copy()
                acc[b].num_frames = cursor
    def progress(state, **fields):
        save(ROOT / 'progress.json', {'state': state, 'task_id': request['task_id'], 'completed_frames': cursor,
             'expected_frames': 8337, 'updated_at': now(), 'pid': os.getpid(), **fields})
    progress('PREPARING')
    os.environ.update(DATASET_PATH=str(DATA), DATA_LINK_FILE=str(link), DATASET='SmartChassisDatasetV7')
    attempt = ROOT / ('canonical-' + str(int(time.time())))
    pipeline = NumpyBenchDataPipeline().setup(save_dir=str(attempt), prefer_shm=False, max_frames=8337 - cursor,
                    eval_subsample=100, scene_sample=0, load_ground_truth=False)
    assert [[s, f['frame_index']] for s, f in pipeline._frames] == expected
    pipeline._frames = pipeline._frames[cursor:]
    oracle_checks = 0
    try:
        for man in pipeline.iter_export():
            scene, fr = expected[cursor]
            assert man.meta['scene_name'] == scene and man.meta['frame_idx'] == fr, 'No silent skipped frames'
            if (TASK / 'STOP_AFTER_FRAME').exists():
                progress('STOPPED_AT_BOUNDARY'); return
            if shutil.disk_usage(ROOT).free < 64 * 1024 ** 3:
                raise ValueError('Disk reserve below 64 GiB; preserve completed results')
            item = next(f for f in source_data['eval'][scene]['frames'] if f['frame_index'] == fr)
            occ = DATA / scene / 'dataset' / str(fr) / item['occ_file']
            while not occ.exists():
                data_state = load(TASK / 'data-state.json')
                if data_state['state'] == 'NEEDS_REVIEW': raise ValueError('GT preparation failed')
                progress('WAITING_GT', current_frame=[scene, fr]); time.sleep(5)
            root = ROOT / 'frames' / f'frame_{cursor:06d}'
            run_id = f'pcm-v322-subnets-full-{cursor:06d}'
            if not (root / 'execution_receipt.json').exists():
                if root.exists(): raise ValueError('Incomplete inference requires reconciliation; never replay silently')
                while True:
                    q = json.loads(subprocess.run(['/home/lixiang/bench-eval/bin/bench-eval', 'queue'], capture_output=True, text=True, check=True, timeout=15).stdout)
                    if not q.get('active') and not q.get('queued') and not q.get('waiting') and not Path('/home/lixiang/bench-eval/locks/bench.lock').exists(): break
                    progress('WAITING_BENCH', current_frame=[scene, fr]); time.sleep(10)
                root.mkdir(parents=True)
                for name in ('source-audit.json',): shutil.copyfile(BASE / 'model-materials/v3.2.2-subnets' / name, root / name)
                shutil.copyfile(TASK / 'code/formal_subnet_chain.py', root / 'executed-adapter.py')
                shutil.copyfile(TASK / 'code/subnet_contract.py', root / 'executed-contract.py')
                args = SimpleNamespace(run_id=run_id, cache_frame=str(man.frame_dir), task_root=str(TASK))
                progress('RUNNING', current_frame=[scene, fr])
                graph, infos = driver.prepare(args, root)
                driver.run_board(args, root, graph, infos)
            receipt = load(root / 'execution_receipt.json')
            assert receipt['state'] == 'succeeded' and receipt['selected_frames'] == [[scene, fr]]
            assert receipt['release']['state'] == 'RELEASED'
            meta = load(root / 'all-subnet-outputs.json'); public = load(root / 'public-outputs.json')
            bins = {p['filename']: (root / ('sub' + str(meta[p['tensor']]['stage'])) / p['filename']).read_bytes()
                    for p in public if p['onnx_name'] in ('/encoder/layers.67/Concat_3_output_0', '/Reshape_34_output_0', '/Concat_28_output_0', '/MatMul_7_output_0', '/aggregator/Min_output_0', '/aggregator/Cast_1_output_0')}
            assert len(bins) == 6
            ort = root / 'onnx_dump'; ort_outputs_to_packup_bins(root / 'onnx_outputs', ort)
            raw = np.load(occ, allow_pickle=False)
            assert ((raw[:, :3] >= 0) & (raw[:, :3] < [800, 48, 200])).all()
            gt, mask, drive, crop = adapt_sparse(raw, [5, -3.6, -2], [.05, .15, .02], [5, -3.6, -1], [400, 48, 100])
            proof = {'scene': scene, 'frame_index': fr, 'frame_id': scene + '_' + str(fr), 'occ_sha256': sha(occ), 'adaptation': crop, 'predictions': {}}
            for backend, values in [('board', bins), ('onnx', {p.name: p.read_bytes() for p in ort.glob('*.bin')})]:
                body, _ = localagg_from_gaussian_bins(values, work_dir=root / (backend + '_temporary_splat'), keep_work=False)
                path = root / (backend + '_final_occ.npy')
                if not path.exists(): path.write_bytes(body)
                else: assert path.read_bytes() == body
                pred = np.load(io.BytesIO(body), allow_pickle=False)
                single = Counts(); single.update(pred, gt, mask, drive)
                if cursor % 100 == 0:
                    oracle = MetricsAccumulator(use_cuda=False); oracle.update(pred, gt, mask, compute_pillar=False)
                    assert abs(oracle.summary()['present_miou'] - single.summary()['present_miou']) < 1e-10
                    oracle_checks += 1
                acc[backend].confusion += single.confusion; acc[backend].ds += single.ds; acc[backend].num_frames += 1
                proof['predictions'][backend] = {'path': str(path), 'sha256': sha(path)}
                save(root / ('gt_metrics_' + backend + '.json'), single.summary())
            save(root / 'gt_frame_receipt.json', proof)
            cursor += 1
            tmp = ROOT / 'checkpoint.tmp'
            with tmp.open('wb') as f:
                np.savez(f, completed=cursor, **{b + '_confusion': acc[b].confusion for b in acc}, **{b + '_ds': acc[b].ds for b in acc})
            os.replace(tmp, ROOT / 'checkpoint.npz')
            for b in acc: save(ROOT / ('metrics_' + b + '.partial.json'), acc[b].summary())
            progress('RUNNING', oracle_checks=oracle_checks)
        assert cursor == 8337, 'Full scope not completed'
        progress('FINALIZING')
        comparisons = [load(ROOT / 'frames' / f'frame_{i:06d}/cosine_summary.json')['frames'][0] for i in range(8337)]
        names = [r['onnx_output_name'] for r in comparisons[0]['outputs']]
        tensors = {name: {'num_frames': 8337, 'mean_cosine': sum(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name'] == name) for f in comparisons) / 8337,
                         'min_cosine': min(next(r['cosine'] for r in f['outputs'] if r['onnx_output_name'] == name) for f in comparisons)} for name in names}
        save(ROOT / 'cosine_summary.json', {'num_frames': 8337, 'frames': comparisons, 'tensors': tensors,
             'mean_cosine_all': sum(f['mean_cosine'] for f in comparisons) / 8337, 'min_cosine_all': min(f['mean_cosine'] for f in comparisons)})
        for b in acc: save(ROOT / ('metrics_' + b + '.json'), acc[b].summary())
        progress('SUCCEEDED', oracle_checks=oracle_checks, version='v3.2.2')
    except Exception as e:
        progress('NEEDS_REVIEW', error_type=type(e).__name__, error=str(e)[:500]); raise
    finally:
        claim.unlink()


if __name__ == '__main__': main()
