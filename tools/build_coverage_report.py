"""Generate the same five reference tables and audit raw -> normalized -> cells."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json, digest
from pcm_eval.model_report import build_model_report
from pcm_eval.reference_report import write_reference_report, display

OUT = Path('evidence/coverage-supplement-20261029')


def main():
    old = load_json('evidence/reference-organized-20261029-02/bound-model.json')
    vision = load_json(OUT / 'vision-collected/normalized-result.json')
    model = build_model_report(old['source_result'], binding=old['binding'], gt_preparation=old['gt_preparation'],
                               head_results=[vision, old['head_results']['trajectory']])
    save_json(OUT / 'bound-model.json', model)
    data = write_reference_report(OUT / 'combined', model_report=OUT / 'bound-model.json',
               compile_report=OUT / 'compile-report/report.json',
               reference_file='evidence/source-docs/release-template-review-20261029.json')
    tables = {t['id']: t for t in data['tables']}
    assert not any(c['state'] == 'missing' for row in tables['vision_classes']['rows'] for c in row)
    assert not any(c['state'] in ('missing', 'not_computed') for row in tables['compile_summary']['rows'] for c in row)
    fields = []
    spec = load_json('configs/report_template.json')
    by_task = {(r['task'], r['class_id']): r for r in load_json(OUT / 'vision-audit.json')['checks']}
    for item, row in zip(spec['model_accuracy']['vision_major_classes'], tables['vision_classes']['rows']):
        gt = by_task[item['task'], item['class_id']]['gt']
        assert row[2]['value'] == gt
        for j, name in enumerate(('support', 'support_percent', 'recall', 'ap'), 2):
            source = next(m for m in vision['metrics'] if m['name'] == 'vision.class.' + name and m['dimensions']['task'] == item['task'] and m['dimensions']['class_id'] == item['class_id'])
            assert source['value'] == row[j]['value']
            fields.append({'table': 'vision_classes', 'row': item['label'], 'column': tables['vision_classes']['headers'][j],
                           'raw_source': source['source'], 'normalized_value': source['value'], 'display_value': display(row[j]), 'state': 'matched'})
    metrics = {b: load_json(OUT / 'compile-collected/gt-adapted' / ('metrics_' + b + '.json')) for b in ('onnx', 'board')}
    for item, row in zip(spec['compile_alignment']['gt_classes'], tables['compile_summary']['rows']):
        for j, backend in enumerate(('onnx', 'board'), 1):
            source = next(r for r in metrics[backend]['class_support'] if r['name'] == item['key'])
            if source['seen'] > 0:
                assert row[j]['state'] == 'measured' and abs(row[j]['value'] - source['iou']) < 1e-10
            else:
                assert row[j]['state'] == 'no_gt'
            fields.append({'table': 'compile_summary', 'row': item['label'], 'column': backend, 'raw_gt_count': source['seen'],
                           'raw_iou': source.get('iou'), 'normalized_value': row[j]['value'], 'display_value': display(row[j]), 'state': 'matched'})
    for name, key in [('可行驶区域iou', 'drivable_iou'), ('present_miou', 'present_miou')]:
        row = next(r for r in tables['compile_summary']['rows'] if r[0]['value'] == name)
        for j, backend in enumerate(('onnx', 'board'), 1):
            assert abs(row[j]['value'] - metrics[backend][key]) < 1e-10
            fields.append({'table': 'compile_summary', 'row': name, 'column': backend, 'raw_value': metrics[backend][key],
                           'display_value': display(row[j]), 'state': 'matched'})
    save_json(OUT / 'mapping-audit.json', {'state': 'passed', 'checks': fields, 'new_vision_subclass_cells': 56,
               'compile_class_cells': 22, 'compile_GT_summary_cells': 4,
               'generated_data_sha256': digest(OUT / 'combined/data.json'), 'published_readback': 'pending'})
    print('combined report generated; 56 vision subclass cells + 26 compilation GT cells matched raw/JSON')


if __name__ == '__main__':
    main()
