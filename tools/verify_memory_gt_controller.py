"""Exercise the staged production GT function on retained predictions only."""
import hashlib
import importlib.util
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from pcm_eval.common import digest, load_json, save_json, utc_now
from pcm_eval.gt_runtime import initialize_native_gt, validated_settings

# This module's stable import name is also resolved by spawned child processes.
sys.path.insert(0, str(BASE))
import formal_compile_remote as controller


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
    full = controller.FULL; root = controller.c.ROOT
    before = {n: digest(root / n) for n in ('checkpoint.npz', 'progress.json')}
    assert load_json(root / 'progress.json')['state'] == 'STOPPED_AT_BOUNDARY'
    assert load_json(root / 'progress.json')['completed_frames'] == 2151
    marker = digest(full / 'STOP_AFTER_FRAME')
    # The real production entry must reject a paused task before claiming a lock.
    with patch.object(controller, 'Engine', side_effect=AssertionError('No inference permitted')):
        try:
            controller.main()
        except ValueError as error:
            assert 'paused by user' in str(error)
        else:
            raise AssertionError('Controller ignored the pause')
    assert not (root / 'controller.lock').exists()
    settings = validated_settings(BASE / 'gt-runtime.json')
    indices = [19, 1001, 2149, 8333]; jobs = []
    for ordinal, index in enumerate(indices):
        source = root / 'frames' / f'frame_{index:06d}'
        private = out / f'frame-{index:06d}'; private.mkdir()
        for name in ('execution_receipt.json', 'public-outputs.json', 'all-subnet-outputs.json'):
            (private / name).write_bytes((source / name).read_bytes())
        (private / 'onnx_outputs').symlink_to(source / 'onnx_outputs', target_is_directory=True)
        for folder in source.glob('sub[0-4]'):
            (private / folder.name).symlink_to(folder, target_is_directory=True)
        jobs.append((index, source, private))
    started = time.perf_counter()
    import os
    os.environ['ROADMODEL_GT_BACKEND'] = settings['backend']
    with ProcessPoolExecutor(max_workers=settings['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_native_gt, initargs=(1, settings['backend'])) as pool:
        futures = [pool.submit(controller.ensure_gt, i, private) for i, _, private in jobs]
        values = [future.result() for future in futures]
    totals = {b: [np.zeros((27, 27), np.int64), np.zeros(3, np.int64)] for b in ('board', 'onnx')}
    original_totals = {b: [x.copy() for x in arrays] for b, arrays in totals.items()}
    for (index, source, private), counts in zip(jobs, values):
        baseline = load_json(source / 'gt_frame_receipt.json')
        for backend in totals:
            assert digest(private / (backend + '_final_occ.npy')) == baseline['predictions'][backend]['sha256']
            assert load_json(private / ('gt_metrics_' + backend + '.json')) == load_json(source / ('gt_metrics_' + backend + '.json'))
            for k in (0, 1):totals[backend][k] += counts[backend][k]
        proof = load_json(private / 'single-counts-receipt.json')
        assert proof['GT_backend'] == 'memory_optimized' and not proof['counts_recovered_from_retained_prediction']
        # Re-entering the completed function must read counts, not recompute GT.
        with patch.object(controller, 'GT_BODY', "raise AssertionError('GT repeated')"):
            recovered = controller.ensure_gt(index, private)
        assert all(np.array_equal(counts[b][k], recovered[b][k]) for b in totals for k in (0, 1))
        # Independent original counts, using the exact retained labels and GT.
        gt, mask, drive, _ = controller.c.adapt_sparse(np.load(controller.occ_for(index), allow_pickle=False),
                         [5, -3.6, -2], [.05, .15, .02], [5, -3.6, -1], [400, 48, 100])
        for backend in totals:
            old = controller.c.Counts();old.update(np.load(source / (backend + '_final_occ.npy'), allow_pickle=False), gt, mask, drive)
            original_totals[backend][0] += old.confusion;original_totals[backend][1] += old.ds
    assert all(np.array_equal(totals[b][k], original_totals[b][k]) for b in totals for k in (0, 1))
    assert all(digest(root / n) == h for n, h in before.items())
    assert digest(full / 'STOP_AFTER_FRAME') == marker
    save_json(out / 'verification.json', dict(state='VERIFIED', indices=indices,
        production_GT_function=True, process_pool_spawn=True, resumed_counts_reused=True,
        all_voxel_labels_metrics_and_counts_exact=True, production_checkpoint_unchanged=True,
        pause_blocks_controller_before_lock_or_inference=True, native_inference_frames=0,
        full_evaluation_resumed=False, wall_seconds=time.perf_counter() - started, finished_at=utc_now()))
    print(json.dumps(load_json(out / 'verification.json')), flush=True)


if __name__ == '__main__':main()
