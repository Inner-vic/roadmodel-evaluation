"""Collect the bounded native run and independently check GT/support/log/JSON."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json, digest
from pcm_eval.parsers import collect_model
from pcm_eval.validation import require_result
from model_space_transport import run

OUT = Path('evidence/coverage-supplement-20261029')
BASE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/coverage-20261029'


def main():
    code = """import pathlib,tarfile,sys,json
b=pathlib.Path(%r);r=json.loads((b/'vision-01/execution_receipt.json').read_text());assert r['state']=='succeeded'
files=[p for p in (b/'vision-01').rglob('*') if p.is_file() and p.suffix in ('.json','.log')]
assert sum(p.stat().st_size for p in files)<32*1024*1024
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p in files:t.add(p,arcname=p.relative_to(b/'vision-01').as_posix())
""" % BASE
    raw = run(['python3', '-c', code], timeout=90)
    target = OUT / 'vision-collected'
    target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for m in tar:
            p = target / m.name
            assert m.isfile() and p.resolve().is_relative_to(target.resolve())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tar.extractfile(m).read())
    receipt = load_json(target / 'execution_receipt.json')
    pre = load_json(target / 'model-preflight.json')
    console = (target / 'eval-console.log').read_text(encoding='utf-8')
    assert pre['state'] == 'passed' and pre['dataset_size'] == 4
    assert re.search(r'100%[^\r\n]*\b4/4\b', console) and 'Total samples used: 4' in console
    save_json(target / 'coverage_receipt.json', {'schema': 'roadmodel.native-coverage/v1', 'state': 'verified',
               'run_id': receipt['run_id'], 'expected_frames': 4, 'observed_frames': 4, 'completed_batches': 4, 'batch_size': 1,
               'selected_frames': receipt['selected_frames'], 'console_sha256': digest(target / 'eval-console.log'),
               'loader_preflight_sha256': digest(target / 'model-preflight.json'),
               'evidence': 'native eval 100% 4/4; Total samples used 4; strict frozen loader and independent label counts'})
    logs = list((target / 'output').glob('*/*.log'))
    assert len(logs) == 1
    result = collect_model(target, 'v3.2.2', expected_frames=4, log_file=logs[0])
    require_result(result)
    save_json(target / 'normalized-result.json', result)
    support = Counter((task, str(value)) for r in pre['samples'] for task, value in r['gt'].items())
    desired = load_json('configs/report_template.json')['model_accuracy']['vision_major_classes']
    checks = []
    for r in desired:
        key = (r['task'], r['class_id'])
        records = [m for m in result['metrics'] if m['name'] == 'vision.class.support' and m['dimensions']['task'] == key[0] and m['dimensions']['class_id'] == key[1]]
        assert len(records) == 1 and records[0]['value'] == support[key] > 0, key
        for metric in ('support', 'support_percent', 'recall', 'ap'):
            m = next(m for m in result['metrics'] if m['name'] == 'vision.class.' + metric and m['dimensions']['task'] == key[0] and m['dimensions']['class_id'] == key[1])
            line = logs[0].read_text(encoding='utf-8').splitlines()[int(m['source']['locator'].split(':')[1]) - 1]
            pattern = {'support': r'n=(\d+)', 'support_percent': r'\(([\d.]+)%\)', 'recall': r'recall=([\d.]+)%', 'ap': r'AP=([\d.]+)'}[metric]
            assert float(re.search(pattern, line)[1]) == m['value']
        checks.append({'task': key[0], 'class_id': key[1], 'label': r['label'], 'gt': support[key], 'native_log_and_json_match': True})
    save_json(OUT / 'vision-audit.json', {'state': 'passed', 'frames': 4, 'reference_subclasses': len(checks), 'checks': checks,
               'source_gt_preflight_sha256': digest(target / 'model-preflight.json'), 'normalized_result_sha256': digest(target / 'normalized-result.json'),
               'new_inference_frames': 4})
    print(json.dumps({'state': 'passed', 'frames': 4, 'reference_subclasses': len(checks), 'metric_records': len(result['metrics'])}), flush=True)


if __name__ == '__main__':
    main()
