"""Batch independent reports, validate delivery, and retain stage receipts."""
from __future__ import annotations

from pathlib import Path

from .common import load_json, save_json, utc_now
from .intake import resolve_submission
from .delivery import export_delivery
from .model_report import write_model_report
from .compile_report import write_compile_report


def run_report_batch(submission, output, *, profile_file=None):
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('Batch output already exists; use a new task ID')
    resolution = resolve_submission(submission, profile_file=profile_file)
    if resolution['action'] != 'report_existing':
        raise ValueError('Batch reports consume existing results; use execute for a prepared execution plan')
    root.mkdir(parents=True)
    save_json(root / 'resolution.json', resolution)
    receipt = {'schema': 'roadmodel.task-receipt/v1', 'state': 'preflight_failed' if not resolution['ready'] else 'running',
               'started_at': utc_now(), 'jobs': [], 'new_inference_frames': 0, 'publication': 'not_requested'}
    if resolution['ready']:
        for job in resolution['jobs']:
            item = {'id': job['id'], 'kind': job['kind'], 'states': {'preflight': 'passed', 'report': 'running',
                     'delivery': 'not_started', 'publication': 'not_requested', 'quality': 'not_assessed'}}
            try:
                inputs = job['inputs']
                target = root / job['id']
                if job['kind'] == 'model_accuracy':
                    report = write_model_report(target / 'generated', result_file=inputs['result'],
                        binding_file=inputs.get('binding_file'), gt_preparation_file=inputs.get('gt_preparation_file'),
                        head_result_files=inputs['head_results'], trajectory_figure_file=inputs.get('trajectory_figure_file'),
                        layout=job['layout'])
                    delivery = export_delivery(target / 'delivery', kind=job['kind'], report_json=report['report_json'],
                        report_xml=report['report_xml'], evidence_bundle=report['evidence_bundle'],
                        results=[inputs['result'], *inputs['head_results']])
                else:
                    report = write_compile_report(target / 'generated/report.xml', result_file=inputs['result'],
                        request_file=inputs.get('request_file'), source_dir=inputs.get('source_dir'),
                        supplemental_dir=inputs.get('supplemental_dir'), layout=job['layout'])
                    delivery = export_delivery(target / 'delivery', kind=job['kind'], report_json=report['json'],
                        report_xml=report['xml'], evidence_bundle=report['details'], results=[inputs['result']], sections=report['sections'])
                item.update(report=report, delivery=delivery)
                item['states'].update(report='generated', delivery='verified')
            except (ValueError, OSError, KeyError, TypeError) as error:
                item.update(error=str(error))
                item['states']['report'] = 'failed'
            receipt['jobs'].append(item)
            save_json(root / 'task.json', receipt, overwrite=(root / 'task.json').exists())
        successes = sum(job['states']['delivery'] == 'verified' for job in receipt['jobs'])
        receipt['state'] = 'succeeded' if successes == len(receipt['jobs']) else 'partial' if successes else 'failed'
    receipt.update(finished_at=utc_now(), issues=resolution['issues'])
    save_json(root / 'task.json', receipt, overwrite=(root / 'task.json').exists())
    return receipt
