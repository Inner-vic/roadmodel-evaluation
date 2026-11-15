"""Two fresh frames, same graph/provider; exact-output gate for CPU threads."""
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RELEASE = Path('/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r1')
sys.path.insert(0, str(RELEASE))
import formal_compile_remote as c
import fast_subnet_driver as driver
from pcm_eval.common import digest, load_json, save_json, utc_now


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
    indices = [2158, 2169]
    before = {name: digest(c.c.ROOT / name) for name in ('checkpoint.npz', 'progress.json')}
    marker = digest(c.FULL / 'STOP_AFTER_FRAME')
    assert load_json(c.c.ROOT / 'progress.json')['state'] == 'STOPPED_AT_BOUNDARY'
    for index in indices:assert not (c.c.ROOT / 'frames' / f'frame_{index:06d}').exists()
    save_json(out / 'reservation.json', dict(indices=indices, frozen_request_sha256=digest(c.FULL / 'request.json'),
              source_release=str(RELEASE), native_inference_frames=0, production_checkpoint_sha256=before['checkpoint.npz']))
    os.environ['ROADMODEL_OPT_VALIDATION'] = str(out)
    engine = c.Engine(); original_session = engine.session; original_run = original_session.run
    times = []
    def timed(*args, **kwargs):
        start = time.monotonic(); value = original_run(*args, **kwargs); times.append(time.monotonic() - start); return value
    original_session.run = timed
    roots = []; feeds = []
    try:
        pipeline = c.make_pipeline(indices, out / 'canonical')
        for index, manifest in zip(indices, pipeline):
            assert [manifest.meta['scene_name'], manifest.meta['frame_idx']] == c.FRAMES[index]
            root = c.c.ROOT / 'frames' / f'frame_{index:06d}'; root.mkdir()
            engine.prepare(SimpleNamespace(cache_frame=str(manifest.frame_dir), task_root=str(out)), root)
            roots.append(root); feeds.append(driver.ort_io._load_inputs(root / 'inputs', engine.input_spec))
        cases = [dict(threads=0, session_run_seconds=times, mean_seconds=statistics.mean(times), all18_bytes_exact=True)]
        for threads in (8, 4):
            options = driver.ort.SessionOptions(); options.graph_optimization_level = driver.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.log_severity_level = 3; options.intra_op_num_threads = threads
            session = driver.ort.InferenceSession(str(engine.onnx), sess_options=options, providers=['CPUExecutionProvider'])
            row = dict(threads=threads, session_run_seconds=[], frames=[], all18_bytes_exact=True)
            assert session.get_providers() == ['CPUExecutionProvider']
            for index, root, feed in zip(indices, roots, feeds):
                start = time.monotonic(); values = session.run(engine.output_names, feed); elapsed = time.monotonic() - start
                row['session_run_seconds'].append(elapsed); hashes = []
                for name, value in zip(engine.output_names, values):
                    prefix = name.replace('/', '@') + '.'
                    matches = [p for p in (root / 'onnx_outputs').iterdir() if p.name.startswith(prefix) and p.suffix == '.bin']
                    assert len(matches) == 1, (name, matches)
                    baseline = matches[0].read_bytes(); candidate = value.tobytes(order='C')
                    same = baseline == candidate; row['all18_bytes_exact'] &= same
                    import hashlib
                    hashes.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype), same_bytes=same,
                                       baseline_sha256=digest(matches[0]), candidate_sha256=hashlib.sha256(candidate).hexdigest()))
                assert len(hashes) == 18
                row['frames'].append(dict(index=index, outputs=hashes))
            row['mean_seconds'] = statistics.mean(row['session_run_seconds']); cases.append(row)
            save_json(out / 'progress.json', dict(state='COMPARING_THREADS', cases=cases), overwrite=(out / 'progress.json').exists())
            del session
        best = min((row for row in cases if row['all18_bytes_exact']), key=lambda row: row['mean_seconds'])
        if best['mean_seconds'] > cases[0]['mean_seconds'] * .9:best = cases[0]
        assert all(digest(c.c.ROOT / name) == sha for name, sha in before.items())
        assert digest(c.FULL / 'STOP_AFTER_FRAME') == marker
        proof = dict(state='VERIFIED', indices=indices, cases=cases, selected_threads=best['threads'],
                     speedup_vs_default=cases[0]['mean_seconds'] / best['mean_seconds'],
                     reference_onnx_sha256=digest(engine.onnx), frozen_request_sha256=digest(c.FULL / 'request.json'),
                     same_provider='CPUExecutionProvider', graph_optimization='ORT_DISABLE_ALL', all_selected18_bytes_exact=True,
                     baseline_new_ONNX_frames=2, additional_private_candidate_forwards=4, native_inference_frames=0,
                     previously_completed_frames_repeated=0, full_evaluation_resumed=False, production_checkpoint_unchanged=True,
                     finished_at=utc_now())
        save_json(out / 'verification.json', proof)
        print(json.dumps({k:v for k,v in proof.items() if k != 'cases'}), flush=True)
    finally:engine.close()


if __name__ == '__main__':main()
