"""Collect JSON evidence only, normalize both GT backends, create an atomic report."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json, digest
from pcm_eval.compile_report import write_compile_report
from pcm_eval.parsers import collect_compile
from pcm_eval.planning import ssh_command
from pcm_eval.validation import require_result
from prepare_subnet_remote import remote
import subprocess

OUT = Path('evidence/coverage-supplement-20261029')
REMOTE = '/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/pcm-v322-subnets-coverage-03'


def main():
    code = """import pathlib,sys,tarfile,json
b=pathlib.Path(%r);assert json.loads((b/'progress.json').read_text())['state']=='succeeded'
files=list(b.glob('*.json'))+list((b/'gt-adapted').glob('*.json'))+list(b.glob('frame_*/frame_manifest.json'))
assert sum(p.stat().st_size for p in files)<20*1024*1024
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p in files:t.add(p,arcname=p.relative_to(b).as_posix())
""" % REMOTE
    proc = subprocess.run(ssh_command('ubuntu-workstation', ['python3', '-c', code]), capture_output=True, timeout=90)
    if proc.returncode: raise RuntimeError(proc.stderr.decode(errors='replace')[-1000:])
    dest = OUT / 'compile-collected'
    dest.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        for member in tar:
            path = dest / member.name
            assert member.isfile() and path.resolve().is_relative_to(dest.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    result = collect_compile(dest, 'v3.2.2', expected_frames=3, gt_dir=dest / 'gt-adapted')
    require_result(result)
    save_json(dest / 'gt-adapted/result.json', result)
    request = deepcopy(load_json('examples/compile_v322_subnets_small.json'))
    receipt = load_json(dest / 'execution_receipt.json')
    request['source'].update(note='Reported composition of three frozen same-version child runs; GT-only postprocessing. Not a Runtime submission.',
                             child_runs=receipt['composition']['children'])
    request['execution'].update(driver_script='/mnt/data/developer/pcm-roadmodel-eval/coverage-gt-20261029/compile_coverage_gt_remote.py',
                                 driver_sha256=digest(Path('tools/compile_coverage_gt_remote.py')),
                                 phase='retained_prediction_aggregation_and_gt_only')
    request['dataset']['version'] = '26-10-20-1'
    request['dataset'].pop('input_dir', None)
    request['dataset'].pop('cached_inputs', None)
    request['dataset']['selected_input_receipts'] = [{'run_id': r['run_id'], 'receipt_sha256': r['source_execution_receipt_sha256']}
                                                    for r in load_json(dest / 'output_integrity.json')['children']]
    request['evaluation'].update(max_frames=3, expected_frames=3, compute_gt_metrics=True, selection={'frames': receipt['selected_frames']})
    save_json(OUT / 'compile-reported-request.json', request)
    report = write_compile_report(OUT / 'compile-report/report.xml', result_file=dest / 'gt-adapted/result.json', request_file=OUT / 'compile-reported-request.json')
    value = load_json(report['json'])
    assert value['ground_truth']['state'] == 'validated' and value['coverage']['classes_with_gt'] == 8
    assert value['coverage']['actual_frames'] == 3 and value['coverage']['valid_tensor_pairs'] == 54
    assert value['result_validity'] == 'validated'
    cross = load_json(dest / 'gt-adapted/crosscheck.json')
    assert cross['state'] == 'passed' and cross['all_class_counts_equal'] and cross['ds_counts_equal']
    audit = {'state': 'passed', 'frames': 3, 'new_inference_frames': 2, 'retained_inference_frames': 1,
             'valid_tensor_pairs': 54, 'GT_classes': 24, 'GT_classes_present': 8,
             'GT_oracle_crosscheck': cross, 'normalized_result_sha256': digest(dest / 'gt-adapted/result.json'),
             'atomic_report_sha256': digest(Path(report['json']))}
    save_json(OUT / 'compile-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
