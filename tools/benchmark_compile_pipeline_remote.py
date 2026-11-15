"""Six frozen, previously unexecuted frames; never starts the full controller."""
import json
import multiprocessing
import os
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

RELEASE = Path(os.environ.get('ROADMODEL_PIPELINE_RELEASE','/mnt/data/bench-eval/service/roadmodel/compile-memory-gt-20261030-r1'))
BATCH_TRANSPORT = os.environ.get('ROADMODEL_PIPELINE_BATCH')=='1'
sys.path.insert(0, str(RELEASE))
import formal_compile_remote as controller
import fast_subnet_driver as driver
from pcm_eval.common import digest, load_json, save_json, utc_now
from pcm_eval.gt_runtime import initialize_native_gt, validated_settings
from pcm_eval.reference_prefetch import settings as reference_settings,ReferencePrefetch


def timed_gt(index, root):
    started = time.monotonic()
    counts = controller.ensure_gt(index, root)
    ended = time.monotonic()
    return counts, {'start': started, 'end': ended, 'seconds': ended - started}


class Trace:
    def __init__(self):
        self.events = []; self.lock = threading.Lock(); self.local = threading.local()

    def call(self, category, function, *args, metadata=None, **kwargs):
        started = time.monotonic(); ok = False
        try:
            value = function(*args, **kwargs); ok = True; return value
        finally:
            event = dict(category=category, index=getattr(self.local, 'index', None),
                         start=started, end=time.monotonic(), ok=ok, **(metadata or {}))
            event['seconds'] = event['end'] - event['start']
            with self.lock:self.events.append(event)

    def snapshot(self):
        with self.lock:return list(self.events)

    def client(self, client):
        original_exec = client._exec
        def execute(command, *args, **kwargs):
            category = ('InferenceExec_launch_and_execution' if 'InferenceExec model_info.json' in command else
                        'board_batch_execution_and_validation' if command.startswith('python3 ') and '/board_batch_' in command else
                        'board_output_validation' if ' validate ' in command else
                        'board_frame_initialize' if ' initialize ' in command else
                        'board_hash_check' if 'sha256sum ' in command else
                        'board_safety_check' if 'pgrep ' in command else
                        'board_cleanup' if 'rm -rf -- ' in command else 'board_other_command')
            stage = re.search(r'/sub([0-4])(?:/|\s)', command)
            return self.call(category, original_exec, command, *args,
                             metadata={'stage': int(stage.group(1)) if stage else None}, **kwargs)
        client._exec = execute
        original_put = client._sftp.put
        def put(localpath, remotepath, *args, **kwargs):
            return self.call('input_SFTP_upload', original_put, localpath, remotepath, *args,
                             metadata={'nbytes': Path(localpath).stat().st_size}, **kwargs)
        client._sftp.put = put
        original_putfo=client._sftp.putfo
        def putfo(body,path,*args,**kwargs):
            return self.call('input_bundle_upload',original_putfo,body,path,*args,metadata={'nbytes':kwargs.get('file_size')},**kwargs)
        client._sftp.putfo=putfo
        original_write = client._sftp_write_bytes
        def write(path, body, *args, **kwargs):
            return self.call('board_metadata_write', original_write, path, body, *args,
                             metadata={'nbytes': len(body)}, **kwargs)
        client._sftp_write_bytes = write


class SessionProxy:
    def __init__(self, session, trace):self.session = session; self.trace = trace
    def __getattr__(self, key):return getattr(self.session, key)
    def run(self, *args, **kwargs):return self.trace.call('ONNX_session_run', self.session.run, *args, **kwargs)


def check_pause(reservation):
    full = controller.FULL; root = controller.c.ROOT
    assert digest(root / 'checkpoint.npz') == reservation['checkpoint_sha256']
    assert digest(root / 'progress.json') == reservation['progress_sha256']
    assert digest(full / 'STOP_AFTER_FRAME') == reservation['stop_marker_sha256']
    assert not (root / 'controller.lock').exists()


def check_queue():
    old = controller.old
    queue = json.loads(old.subprocess.run([str(old.RUNNER / 'bin/bench-eval'), 'queue'],
                      capture_output=True, text=True, timeout=20, check=True).stdout)
    assert not queue.get('active') and not queue.get('queued') and not queue.get('waiting'), 'Shared bench is occupied; no dispatch'
    assert not (old.RUNNER / 'locks/bench.lock').exists(), 'Shared bench is locked'


def main():
    out = Path(sys.argv[1]); reservation = load_json(out / 'reservation.json')
    assert not (out / 'started.json').exists(), 'Inspect the existing attempt; never replay'
    indices = reservation['indices']; assert len(indices) == 6 and len(set(indices)) == 6
    settings = validated_settings(RELEASE / 'gt-runtime.json')
    assert digest(controller.FULL / 'request.json') == settings['frozen_request_sha256'] == reservation['frozen_request_sha256']
    check_pause(reservation); check_queue()
    for index in indices:
        assert not (controller.c.ROOT / 'frames' / f'frame_{index:06d}').exists()
    os.environ['ROADMODEL_OPT_VALIDATION'] = str(out)
    os.environ['ROADMODEL_GT_BACKEND'] = settings['backend']
    save_json(out / 'started.json', dict(pid=os.getpid(), started_at=utc_now(), full_evaluation_resumed=False))
    trace = Trace(); engine = None; pool = None; reference = None; futures = []; prepares = {}; boards = {}; results = []
    startup = time.monotonic(); first_work = None; last_gt_end = None
    try:
        engine = controller.Engine(); initialization = time.monotonic() - startup
        engine.session = SessionProxy(engine.session, trace)
        reference_config=reference_settings(RELEASE/'reference-runtime.json',digest(controller.FULL/'request.json'))
        setup = time.monotonic(); exporter = controller.make_pipeline(indices, out / 'canonical')
        pipeline_setup = time.monotonic() - setup
        trace.local.index = None
        connect_start = time.monotonic(); client = engine.connect(); connect_seconds = time.monotonic() - connect_start
        # Do not run hardware here: only ensure that this original client sees no residual process.
        assert not client._exec('pgrep -x InferenceExec', timeout=10, check=False)[1].strip()
        trace.client(client)
        original_stream = driver.stream_read
        def stream(*args, **kwargs):return trace.call('public_output_download', original_stream, *args, **kwargs)
        driver.stream_read = stream
        original_compare = engine.compare
        engine.compare = lambda *args, **kwargs: trace.call('tensor_comparison', original_compare, *args, **kwargs)
        pool = ProcessPoolExecutor(max_workers=settings['workers'], mp_context=multiprocessing.get_context('spawn'),
                                   initializer=initialize_native_gt, initargs=(1, settings['backend']))

        def prepare(index):
            trace.local.index = index; started = time.monotonic()
            manifest = trace.call('data_prepare_and_export', next, exporter)
            assert [manifest.meta['scene_name'], manifest.meta['frame_idx']] == controller.FRAMES[index]
            root = controller.c.ROOT / 'frames' / f'frame_{index:06d}'
            root.mkdir(exist_ok=False)
            trace.call('prepare_including_ONNX', engine.prepare,
                       SimpleNamespace(cache_frame=str(manifest.frame_dir), task_root=str(out)), root)
            prepares[index] = {'start': started, 'end': time.monotonic()}
            return root

        first_work = time.monotonic()
        if reference_config['workers']>1:
            reference=ReferencePrefetch(engine,indices,controller.FRAMES,exporter,controller.c.ROOT/'frames',out,reference_config)
        else:prepared = engine.producers.submit(prepare, indices[0])
        for ordinal, index in enumerate(indices):
            trace.local.index = index
            root = trace.call('wait_prepared', reference.get,index) if reference else trace.call('wait_prepared',prepared.result)
            trace.call('validation_pause_check', check_pause, reservation)
            trace.call('validation_queue_precheck', check_queue)
            assert not (root / 'execution.json').exists() and not (root / 'execution_receipt.json').exists()
            if not reference and ordinal + 1 < len(indices):prepared = engine.producers.submit(prepare, indices[ordinal + 1])
            trace.local.index = index; started = time.monotonic()
            engine.run_board(SimpleNamespace(run_id=f'pcm-v322-subnets-full-{index:06d}-optv1'), root,reference_check=BATCH_TRANSPORT and ordinal==0)
            boards[index] = {'start': started, 'end': time.monotonic()}
            submitted = time.monotonic(); future = pool.submit(timed_gt, index, root)
            futures.append((index, root, submitted, future))
            save_json(out / 'progress.json', dict(state='BOUNDED_NATIVE_EXECUTION', native_completed=ordinal + 1,
                      maximum_native_frames=6, indices=indices, full_evaluation_resumed=False), overwrite=(out / 'progress.json').exists())
            save_json(out / 'trace-events.json', trace.snapshot(), overwrite=(out / 'trace-events.json').exists())
        raw_counts = {b + '_' + k: __import__('numpy').zeros(shape, __import__('numpy').int64)
                      for b in ('board', 'onnx') for k, shape in (('confusion', (27, 27)), ('ds', (3,)))}
        for index, root, submitted, future in futures:
            counts, gt = future.result(); last_gt_end = max(last_gt_end or 0, gt['end'])
            execution = load_json(root / 'execution_receipt.json'); integrity = load_json(root / 'all-subnet-outputs.json')
            assert execution['state'] == 'succeeded' and execution['selected_frames'] == [controller.FRAMES[index]] and execution['release']['state'] == 'RELEASED'
            assert len(integrity) == 49 and all(x['finite'] for x in integrity.values())
            public = load_json(root / 'public-outputs.json'); assert len(public) == 18
            import numpy as np
            for port in public:
                meta = integrity[port['tensor']]; p = root / ('sub' + str(meta['stage'])) / port['filename']
                assert p.stat().st_size == port['nbytes'] and digest(p) == meta['sha256']
                assert np.isfinite(np.fromfile(p, dtype=port['dtype'])).all()
            for backend in counts:
                raw_counts[backend + '_confusion'] += counts[backend][0]; raw_counts[backend + '_ds'] += counts[backend][1]
            events = [x for x in trace.snapshot() if x['index'] == index]
            if reference:
                prep=load_json(root/'prepare.json');prepares[index]={'start':0,'end':prep['preparation_seconds']}
                assert prep['state']=='READY' and (root/'onnx_outputs').is_dir()
            else:assert sum(x['category']=='ONNX_session_run' for x in events)==1
            if BATCH_TRANSPORT:
                assert sum(x['category']=='board_batch_execution_and_validation' for x in events)==1
                assert sum(x['category']=='public_output_download' for x in events)==1
                assert len(load_json(root/'execution.json')['stages'])==5
            else:
                assert sum(x['category']=='InferenceExec_launch_and_execution' for x in events)==5
                assert sum(x['category']=='public_output_download' for x in events)==18
            categories = defaultdict(float)
            for event in events:categories[event['category']] += event['seconds']
            results.append(dict(index=index, frame=controller.FRAMES[index], preparation=prepares[index], board=boards[index],
                                GT=gt, GT_dispatch_wait_seconds=gt['start'] - submitted, timings=dict(categories),
                                new_ONNX_inferences=1, new_NPU_chains=1, checked_outputs=49, compared_public_outputs=18,
                                downloaded_bytes=load_json(root / 'execution.json')['public_tensor_bytes_downloaded'],
                                optimized_board_seconds=load_json(root/'execution.json')['optimized_board_seconds_before_diagnostic'],
                                batch_details={k:v for k,v in load_json(root/'execution.json').items() if k.startswith('batch_') or k=='board_batch_timing' or k=='reference_validation'}))
        complete = time.monotonic(); check_pause(reservation)
        save_json(out/'measured-execution.json', dict(frames=results, first_work=first_work, last_GT_end=last_gt_end,
                  initialization_seconds=initialization, pipeline_setup_seconds=pipeline_setup, board_connection_seconds=connect_seconds))
        with (out / 'counts-checkpoint.npz').open('xb') as stream:np.savez(stream, completed=6, **raw_counts)
        # Collect with the existing report implementation. No publish function is imported/called.
        # The candidate's numerical package intentionally has no report configs.
        # Reuse the immutable production configs, checking report source identity.
        production=Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
        for name in ('compile_aggregate.py','compile_report.py','business_layout.py'):
            assert digest(RELEASE/'pcm_eval'/name)==digest(production/'pcm_eval'/name)
        import pcm_eval.compile_report as report_module
        import pcm_eval.business_layout as layout_module
        import pcm_eval.validation as validation_module
        validation_module.SCHEMA=production/'schemas/single_eval.schema.json'
        report_module.SEMANTICS=production/'configs/compile_output_semantics.json'
        layout_module.TEMPLATE=production/'configs/report_template.json'
        from pcm_eval.compile_aggregate import aggregate_compile
        from pcm_eval.compile_report import render_sections
        from pcm_eval.business_layout import compile_business_sections
        config = load_json(Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01/full-report-coordinator.json'))
        report = aggregate_compile([str(x[1]) for x in futures], [controller.FRAMES[i] for i in indices], out / 'report',
                    request=config['report']['compile_execution_request'], run_id='bounded-pipeline-timing-6', checkpoint=out / 'counts-checkpoint.npz')
        import xml.etree.ElementTree as ET
        xml = '<document>' + ''.join(s['xml'] for s in compile_business_sections(report, render_sections(report))) + '</document>'
        ET.fromstring(xml); (out / 'report/report.xml').write_text(xml)
        assert report['result_validity'] == 'validated'
        timings = {}
        for category in sorted({k for x in results for k in x['timings']}):
            timings[category] = statistics.mean(x['timings'].get(category, 0) for x in results)
        intervals = [results[i]['board']['end'] - results[i - 1]['board']['end'] for i in range(1, len(results))]
        proof = dict(state='VERIFIED', indices=indices, frames=results, average_stage_seconds=timings,
                     board_completion_intervals=intervals, mean_board_completion_interval_seconds=statistics.mean(intervals),
                     mean_later_three_completion_intervals_seconds=statistics.mean(intervals[-3:]),
                     measured_work_wall_seconds=last_gt_end - first_work, observed_wall_seconds_per_frame=(last_gt_end - first_work) / 6,
                     mean_GT_compute_seconds=statistics.mean(x['GT']['seconds'] for x in results),
                     mean_prepare_seconds=statistics.mean(x['preparation']['end'] - x['preparation']['start'] for x in results),
                     mean_board_seconds=statistics.mean(x['board']['end'] - x['board']['start'] for x in results),
                     startup_seconds=dict(engine=initialization, pipeline_setup=pipeline_setup, board_connection=connect_seconds),
                     end_to_end_seconds_including_startup_and_report=time.monotonic()-startup,
                     postrun_verification_and_report_seconds=time.monotonic()-complete,
                     reference_workers=reference_config['workers'],
                     new_ONNX_inferences=6, new_NPU_chains=6, previously_completed_frames_repeated=0,
                     all49_checks_and18_comparisons_preserved=True, metric_definitions_unchanged=True,
                     report_validity=report['result_validity'], report_tensor_records=108, Feishu_writes=0,
                     full_evaluation_resumed=False, production_prefix=2151, production_checkpoint_unchanged=True,
                     new_results_retained_for_future_resume=True, finished_at=utc_now())
        save_json(out / 'verification.json', proof)
        save_json(out / 'progress.json', dict(state='VERIFIED', native_completed=6, maximum_native_frames=6,
                  full_evaluation_resumed=False), overwrite=True)
        print(json.dumps(proof), flush=True)
    except Exception as error:
        save_json(out / 'failure.json', dict(state='NEEDS_REVIEW', error_type=type(error).__name__, error=str(error)[:800],
                  native_attempts=[i for i in indices if (controller.c.ROOT / 'frames' / f'frame_{i:06d}' / 'execution.json').exists()],
                  full_evaluation_resumed=False, automatic_retry=False))
        raise
    finally:
        if reference:reference.close()
        if pool:pool.shutdown(wait=True, cancel_futures=True)
        if engine:engine.close()
        save_json(out / 'trace-events.json', trace.snapshot(), overwrite=(out / 'trace-events.json').exists())
        check_pause(reservation)


if __name__ == '__main__':main()
