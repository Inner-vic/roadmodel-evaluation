"""Run the original OCC/vision controllers with one fresh, continuous clock."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rerun_timing import event, span, summarize, utc_now


def main(config_file):
    config = json.loads(Path(config_file).read_text())
    task = Path(config['task_root']).resolve()
    assert task.name == config['task_id'] and task.parent == Path('/mnt/data/bench-eval/roadmodel/formal')
    assert not list((task/'model').glob('*/normalized-result.json'))
    assert not (task/'vision/normalized-result.json').exists()
    sink = task/'timing-events.jsonl'
    if (task/'timing-start.json').exists():
        assert config.get('resume_setup_only') is True, 'Reconcile an existing timed run; never silently restart'
        assert not list((task/'model').glob('batch-*')), 'Cannot replay a dispatched native batch'
        prior=json.loads((task/'timing-summary.json').read_text())
        assert prior['state']=='NEEDS_REVIEW'
        start=json.loads((task/'timing-start.json').read_text())
        assert start['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        event(sink,'setup_recovery','RUNNING',original_started_at=start['started_at'])
    else:
        start = dict(state='RUNNING', task_id=config['task_id'], started_at=utc_now(),
                 started_monotonic_ns=time.monotonic_ns(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 old_result_reuse=False, data_cache_reuse=True,
                 expected_OCC_frames=8337, expected_vision_frames=11543)
        (task/'timing-start.json').write_text(json.dumps(start, indent=2))
    (task/'progress.json').write_text(json.dumps({'state':'RUNNING','started_at':start['started_at'],'setup_recovery':config.get('resume_setup_only',False)}))
    original_run = subprocess.run
    current = {'head': 'setup'}

    def measured_run(argv, *args, **kwargs):
        with span(sink, classify_command(argv), head=current['head']):
            result = original_run(argv, *args, **kwargs)
        if result.returncode:
            event(sink, classify_command(argv), 'NONZERO_EXIT', head=current['head'], exit_code=result.returncode)
        return result

    subprocess.run = measured_run
    os.environ['ROADMODEL_TASK_ROOT'] = str(task)
    os.environ['ROADMODEL_STAGE_LOCAL'] = '1'
    os.environ['ROADMODEL_IDENTITY'] = config['identity_file']
    try:
        # Verify report dependencies before any native evaluation is dispatched.
        from pcm_eval.model_aggregate import aggregate_occ
        from pcm_eval.model_report import build_model_report, render_model_xml
        from pcm_eval.business_layout import render_model_business, TEMPLATE
        assert TEMPLATE.is_file(), 'Missing report template; stop before inference'
        occ = importlib.import_module('formal_model_controller')
        vision = importlib.import_module('formal_vision_controller')
        for module in (occ, vision):
            assert module.TASK.resolve() == task
            assert module.SPACE == config['space_root']
            original_state = module.state

            def measured_state(*args, _fn=original_state, **kwargs):
                event(sink, 'controller_state', str(args[0]), head=current['head'],
                      completed_frames=args[1] if len(args)>1 else kwargs.get('completed_frames'))
                return _fn(*args, **kwargs)

            module.state = measured_state
        # Upload timing support before either native worker is launched.
        with span(sink, 'timing_support_upload'):
            occ.put(Path(__file__).with_name('model_rerun_timing.py'), config['space_root']+'/model_rerun_timing.py')
        with span(sink, 'vision_cache_validation'):
            manifest=task/'vision-cache-manifest.json'
            occ.put(manifest,config['space_root']+'/vision-cache-manifest.json')
            # Supplier checker handles multipart ETags through trusted SHA256 receipts.
            # An ETag with a '-N' suffix is not a whole-file MD5.
            inventory=json.loads(manifest.read_text())
            assert len(inventory['files'])==23086
            checker=str(Path(config['vision_cache_root']).parent/'code/vision-checker.py')
            check_request={'root':config['vision_cache_root'],'files':inventory['files'],'rehash':True}
            verified=json.loads(occ.run(['python3',checker],json.dumps(check_request).encode(),timeout=600).decode())
            assert verified['state']=='complete' and verified['verified_files']==23086 and not verified['conflicts']
        for name, module in [('OCC', occ), ('vision', vision)]:
            current['head'] = name
            with span(sink, 'head_end_to_end', head=name):
                module.main()
        current['head'] = 'aggregation'
        with span(sink, 'model_report_generation'):
            from pcm_eval.model_aggregate import aggregate_occ
            from pcm_eval.model_report import build_model_report, render_model_xml
            from pcm_eval.business_layout import render_model_business
            req = json.loads((task/'request.json').read_text())
            files = [task/'model'/f'batch-{i:04d}-a02/normalized-result.json' for i in range(84)]
            result = aggregate_occ(files, req['frames'], task/'report/occ', run_id=config['task_id']+'-occ')
            batches=[json.loads(p.read_text())['provenance']['native_run'] for p in files]
            result['provenance']['native_run'].update(started_at=min(b['started_at'] for b in batches),finished_at=max(b['finished_at'] for b in batches))
            (task/'report/occ/normalized-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            baseline = json.loads(Path(config['baseline_model_report']).read_text())
            binding = baseline['binding']
            binding['dataset'].update(selected_frames=req['frames'], version='26-10-20-1')
            vresult = json.loads((task/'vision/normalized-result.json').read_text())
            report = build_model_report(result, binding=binding, head_results=[vresult])
            report['timing_run_id'] = config['task_id']
            (task/'report/model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            (task/'report/model-report.xml').write_text(render_model_business(report, render_model_xml(report, json_attachment='model-report.json')), encoding='utf8')
        state = 'SUCCEEDED'
    except BaseException as exc:
        state = 'NEEDS_REVIEW'
        event(sink, 'task', state, head=current['head'], error_type=type(exc).__name__, error=str(exc)[:400])
        raise
    finally:
        subprocess.run = original_run
        end = dict(state=locals().get('state', 'NEEDS_REVIEW'), task_id=config['task_id'],
                   started_at=start['started_at'], observed_at=utc_now(),
                   elapsed_seconds=(time.monotonic_ns()-start['started_monotonic_ns'])/1e9,
                   elapsed_clock='single_workstation_boot_monotonic', stage_totals=summarize(sink),
                   note='stage totals can nest; elapsed_seconds is authoritative; data/weights cached, predictions fresh',
                   Feishu_publication_included=False)
        (task/'timing-summary.json').write_text(json.dumps(end, ensure_ascii=False, indent=2))
        (task/'progress.json').write_text(json.dumps(end, ensure_ascii=False, indent=2))


def classify_command(argv):
    from model_rerun_timing import classify
    return classify(argv)


if __name__ == '__main__':
    main(sys.argv[1])
