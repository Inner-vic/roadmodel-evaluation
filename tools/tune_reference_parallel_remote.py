"""Four fresh inputs; compare original CPU sessions at 1/2/4 workers."""
import json
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

RELEASE = Path('/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r1')
sys.path.insert(0, str(RELEASE))
import formal_compile_remote as c
from pcm_eval.common import digest, load_json, save_json, utc_now
# This experiment's new worker module lives alongside this script, not in a changed release.
import reference_workers as refs


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False); indices = [2159, 2160, 2170, 2171]
    assert load_json(c.c.ROOT / 'progress.json')['state'] == 'STOPPED_AT_BOUNDARY'
    initial = {p: digest(c.c.ROOT / p) for p in ('checkpoint.npz', 'progress.json')}
    marker = digest(c.FULL / 'STOP_AFTER_FRAME')
    for index in indices:assert not (c.c.ROOT / 'frames' / f'frame_{index:06d}').exists()
    save_json(out / 'reservation.json', dict(indices=indices, checkpoint_sha256=initial['checkpoint.npz'], native_inference_frames=0))
    os.environ['ROADMODEL_OPT_VALIDATION'] = str(out)
    engine = c.Engine(); jobs = []; times = []; baseline_start = time.monotonic()
    try:
        exporter = c.make_pipeline(indices, out / 'canonical')
        for index, manifest in zip(indices, exporter):
            assert [manifest.meta['scene_name'], manifest.meta['frame_idx']] == c.FRAMES[index]
            root = c.c.ROOT / 'frames' / f'frame_{index:06d}'; root.mkdir()
            start = time.monotonic(); engine.prepare(SimpleNamespace(cache_frame=str(manifest.frame_dir), task_root=str(out)), root)
            times.append(time.monotonic() - start)
            jobs.append(dict(root='', baseline_root=str(root), cache_frame=str(manifest.frame_dir), task_root=str(out),
                             onnx_fp16_sha256=engine.binding['onnx_fp16_sha256']))
        cases = [dict(workers=1, reference_prepare_seconds=sum(times), mean_seconds_per_frame=statistics.mean(times), all18_bytes_exact=True)]
        context = multiprocessing.get_context('spawn')
        for workers in (2, 4):
            case = out / ('workers-%d' % workers); case.mkdir(); barrier = context.Barrier(workers)
            start = time.monotonic()
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=refs.initialize_reference, initargs=(barrier,)) as pool:
                warm = [pool.submit(refs.ready) for _ in range(workers)]
                for future in warm:future.result()
                initialization = time.monotonic() - start; start = time.monotonic()
                futures = [pool.submit(refs.prepare_reference, dict(job, root=str(case / ('frame_%06d' % index)))) for index, job in zip(indices, jobs)]
                results = [future.result() for future in futures]; wall = time.monotonic() - start
            row = dict(workers=workers, reference_prepare_seconds=wall, mean_seconds_per_frame=wall / len(indices),
                       initialization_seconds=initialization, all18_bytes_exact=all(r['all18_bytes_exact'] for r in results), frames=results)
            cases.append(row); save_json(out / 'progress.json', dict(state='COMPARING_REFERENCE_WORKERS', cases=cases), overwrite=(out / 'progress.json').exists())
        best = min((r for r in cases if r['all18_bytes_exact']), key=lambda r:r['mean_seconds_per_frame'])
        if best['mean_seconds_per_frame'] > cases[0]['mean_seconds_per_frame'] * .9:best = cases[0]
        assert all(digest(c.c.ROOT / p) == h for p, h in initial.items()) and digest(c.FULL / 'STOP_AFTER_FRAME') == marker
        proof = dict(state='VERIFIED', indices=indices, cases=cases, selected_workers=best['workers'],
                     speedup_vs_one_worker=cases[0]['mean_seconds_per_frame'] / best['mean_seconds_per_frame'],
                     all_selected18_bytes_exact=True, reference_onnx_sha256=digest(engine.onnx), frozen_request_sha256=digest(c.FULL / 'request.json'),
                     baseline_new_ONNX_frames=4, additional_private_candidate_forwards=8, native_inference_frames=0,
                     same_original_session_options=True, previously_completed_frames_repeated=0, full_evaluation_resumed=False,
                     production_checkpoint_unchanged=True, finished_at=utc_now())
        save_json(out / 'verification.json', proof); print(json.dumps(proof), flush=True)
    finally:engine.close()


if __name__ == '__main__':main()
