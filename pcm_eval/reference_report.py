"""Compose existing atomic results using only the supplied PCM reference tables.

This presentation step never runs inference, computes a release verdict, or changes
the model/compile atomic capabilities. Cell states and source counts stay in JSON.
"""
from copy import deepcopy
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

from .common import digest, load_json, save_json, utc_now
from .business_layout import TEMPLATE, p, table


STATES = {'missing': '未提供', 'no_gt': '无GT样本',
          'not_computed': '未计算', 'not_applicable': '—'}


def cell(value=None, *, state='measured', unit=None, source=None):
    if value is None and state == 'measured':
        state = 'missing'
    return {'value': value, 'state': state, 'unit': unit, 'source': source}


def display(c):
    if c['state'] != 'measured':
        return STATES[c['state']]
    v = c['value']
    if c['unit'] == 'mean_min_percent':
        return f'{v[0]:.2f} / {v[1]:.2f}'
    if c['unit'] == 'ratio':
        return f'{v:.4f}'
    if c['unit'] == 'percent':
        return f'{v:.2f}%'
    if c['unit'] == 'percent_number':
        return f'{v:.2f}'
    return str(v)


def metric(metrics, name, **dims):
    matches = [m for m in metrics if m['name'] == name
               and all(str(m.get('dimensions', {}).get(k)) == str(v) for k, v in dims.items())]
    if len(matches) > 1:
        raise ValueError('Ambiguous metric: ' + name)
    m = matches[0] if matches else {}
    return cell(m.get('value'), unit=m.get('unit'), source=m.get('source'))


def organize(model, compiled, reference):
    if model.get('schema') != 'roadmodel.model-single-report/v1':
        raise ValueError('Validated model single report required')
    if compiled.get('schema') != 'roadmodel.compile-single-report/v3':
        raise ValueError('Validated compile single report required')
    version = model['version_name']
    if version != compiled['identity'].get('model_version'):
        raise ValueError('Model and compilation versions differ; cannot label as one version')
    if any(r['execution'].get('state') != 'succeeded' for r in (model, compiled)):
        raise ValueError('Successful existing executions required')
    spec = load_json(TEMPLATE)
    tree = ET.fromstring('<root>' + reference['data']['document']['content'] + '</root>')
    originals = list(tree.iter('table'))
    if len(originals) != 5:
        raise ValueError('Reference table structure changed; engineering review required')
    labels = [[[''.join(c.itertext()) for c in row] for row in t.findall('.//tr')]
              for t in originals]
    # Detect drift rather than silently rendering a newly invented business table.
    if labels[0][2:-2] and [r[0] for r in labels[0][2:-2]] != [x['label'] for x in spec['model_accuracy']['occ_classes']]:
        raise ValueError('OCC reference row mapping changed')
    if [r[0] for r in labels[3][1:-3]] != [x['label'] for x in spec['compile_alignment']['gt_classes']]:
        raise ValueError('Compile reference row mapping changed')
    sections = {s['scope']: s for s in model['occ_sections']}
    occ = {s: {r['class_name']: r for r in x['classes']} for s, x in sections.items()}
    audit = []
    def occ_value(row, key):
        if not row:
            return cell()
        return cell(row.get(key), state='no_gt' if row['seen'] == 0 else 'measured',
                    unit='percent_number', source=row.get('source'))
    rows = []
    for item in spec['model_accuracy']['occ_classes']:
        a, b = occ.get('EVAL', {}).get(item['key']), occ.get('EVAL_PILLAR', {}).get(item['key'])
        rows.append([cell(item['label']), *[occ_value(a, k) for k in ('iou', 'precision', 'recall')], occ_value(b, 'iou')])
        audit.append({'label': item['label'], 'class_name': item['key'],
                      'occ_gt': a['seen'] if a else None, 'pillar_gt': b['seen'] if b else None})
    drivable = metric(model['all_metrics'], 'drivable.iou')
    drivable['unit'] = 'percent_number'
    rows.append([cell(labels[0][-2][0]), drivable, *[cell(state='not_applicable') for _ in range(3)]])
    a, b = [sections.get(s, {}).get('non_empty') for s in ('EVAL', 'EVAL_PILLAR')]
    rows.append([cell(labels[0][-1][0]), *[occ_value(a, k) for k in ('iou', 'precision', 'recall')], occ_value(b, 'iou')])
    # Merged paired version columns become one column per original metric.
    tables = [{'id': 'occ', 'headers': labels[0][0], 'rows': rows}]
    vision_rows = []
    for i, item in enumerate(spec['model_accuracy']['vision_tasks']):
        vision_rows.append([cell(labels[1][i + 1][0]), *[metric(model['vision_metrics'], name, task=item['key'])
                           for name in ('vision.map', 'vision.accuracy', 'vision.macro_recall', 'vision.major_classes')]])
    vision_rows.append([cell(labels[1][-1][0]), metric(model['vision_metrics'], 'vision.avg_map'),
                        *[cell(state='not_applicable') for _ in range(3)]])
    tables.append({'id': 'vision_summary', 'headers': labels[1][0], 'rows': vision_rows})
    details = []
    for i, item in enumerate(spec['model_accuracy']['vision_major_classes']):
        dims = {'task': item['task'], 'class_id': item['class_id']}
        values = [metric(model['vision_metrics'], n, **dims) for n in
                  ('vision.class.support', 'vision.class.support_percent', 'vision.class.recall', 'vision.class.ap')]
        # A missing class record is not proof that its GT count is zero.
        details.append([cell(labels[2][i + 1][0]), cell(labels[2][i + 1][1]), *values])
    tables.append({'id': 'vision_classes', 'headers': labels[2][0], 'rows': details})
    gt = compiled['ground_truth']
    lookup = {r['name']: r for r in gt['classes']}
    cr = []
    for item in spec['compile_alignment']['gt_classes']:
        r = lookup.get(item['key'])
        state = 'not_computed' if gt['state'] != 'validated' else 'missing' if r is None else 'no_gt' if r['seen'] == 0 else 'measured'
        cr.append([cell(item['label']), *[cell((r or {}).get(k), state=state, unit='percent_number') for k in ('onnx_iou', 'board_iou')]])
    for name, metric_id in [('可行驶区域iou', 'drivable_iou'), ('present_miou', 'present_miou')]:
        matches = [r for r in gt['summary'] if r['metric_id'] == metric_id]
        r = matches[0] if len(matches) == 1 else {}
        cr.append([cell(name), *[cell(r.get(k), state='measured' if gt['state'] == 'validated' else 'not_computed', unit='percent_number') for k in ('onnx', 'board')]])
    cr.append([cell('余弦 mean(all tensors)'), cell(state='not_applicable'),
               cell(compiled['tensor_alignment']['mean_frame_mean_cosine'] * 100, unit='percent_number')])
    tables.append({'id': 'compile_summary', 'headers': ['类别/指标', 'ONNX fp16', '本次板端'], 'rows': cr})
    outputs = {r['name']: r for r in compiled['tensor_alignment']['outputs']}
    tr = []
    for item in spec['compile_alignment']['outputs']:
        r = outputs.get(item['key'])
        tr.append([cell(item['key']), cell(item['meaning']), cell(item['group']),
                   cell([r['mean_cosine'] * 100, r['min_cosine'] * 100] if r else None, unit='mean_min_percent')])
    tables.append({'id': 'compile_tensors', 'headers': ['输出 tensor', '含义', '分组', '本次板端 mean/min'], 'rows': tr})
    scopes = {'occ': {'run_id': model['run_id'], 'dataset_version': model['binding'].get('dataset', {}).get('version'),
                       'frames': model['coverage']['observed'].get('model_samples')},
              'compile': {'run_id': compiled['identity']['run_id'], 'frames': compiled['coverage']['actual_frames'],
                          'dataset_version': compiled['configuration']['request']['dataset'].get('version'), 'gt_state': gt['state'],
                          'classes_with_gt': compiled['coverage']['classes_with_gt'],
                          'inference_composition': compiled['configuration']['executed'].get('composition')}}
    for kind, head in model.get('head_results', {}).items():
        native = head['provenance']['native_run']
        scopes[kind] = {'run_id': native['run_id'], 'dataset_version': native.get('dataset_version'),
                        'frames': head['coverage']['observed'].get('model_samples')}
    return {'schema': 'roadmodel.reference-combined-data/v1', 'version_name': version,
            'generated_at': utc_now(), 'template_source': spec['source'], 'tables': tables,
            'model_delivery_url': model['binding'].get('model_reference_url'),
            'sample_scopes': scopes, 'occ_gt_support': audit,
            'trajectory': {'state': 'no_metric_table_in_reference', 'existing_metrics_retained': len(model['trajectory_metrics'])},
            'generation': {'new_inference_frames': 0, 'cross_version_comparison': False, 'combined_score': None}}


def render(data, attachment):
    tables = {t['id']: table(t['headers'], [[display(c) for c in row] for row in t['rows']]) for t in data['tables']}
    def h(level, text):
        return f'<h{level} seq="auto">' + escape(text) + f'</h{level}>'
    s = data['sample_scopes']
    out = ['<title>RoadModel '+escape(data['version_name'])+' 单次评测结果（按PCM V3.1.0原表整理）</title>',
           p('模型版本：'+data['version_name']),
           '<p>模型交付：<a href="'+escape(data.get('model_delivery_url') or '')+'">原报告所附模型记录</a></p>',
           p('全量评测结果；不自动作发版通过判定。' if data.get('evaluation_scope')=='full' else '所列范围的单次评测结果；正式全量任务仍独立执行，不作为全量发版结论。' if data.get('evaluation_scope')=='selected_validation' else '小场景结果汇总；不作为全量发版结论。'),
           '<p>表格依据：<a href="'+escape(data['template_source']['url'])+'">PCM 发版报告V3.1.0</a>。保留原表指标和顺序，仅去除跨版本列及未执行的板端配置列。</p>',
           h(1, '模型精度评测'), h(2, 'occ头'),
           p(f"数据集 {s['occ']['dataset_version']}；{s['occ']['frames']} 帧；运行 {s['occ']['run_id']}。"),
           tables['occ'], p('无GT样本表示本次选定样本中该类GT计数为0；未提供表示原结果未输出该项。'),
           h(2, '轨迹头'), p('参考报告未定义该头的指标表，本次不扩展表格；已有轨迹结果保留在结构化附件。'),
           h(2, '视觉头')]
    if 'vision' in s:
        v = s['vision']
        out.append(p(f"数据集 {v['dataset_version']} 的 {v['frames']} 帧视觉结果；运行 {v['run_id']}。与上述OCC样本范围不同；"+('采用原生完整验证集范围。' if data.get('evaluation_scope')=='full' else '类别覆盖选样不代表数据集分布。')))
    else:
        out.append(p('本次未取得视觉头结果。'))
    out.extend([p('核心类 mAP 与准确率'), tables['vision_summary'],
                p('参与 mAP 的子类明细（major 子类；沿用原表子类行，未输出项标为未提供）'), tables['vision_classes'],
                p('准确率=全局argmax命中率；mAP为原生完整验证集一次计算的输出。' if data.get('evaluation_scope')=='full' else '准确率=全局argmax命中率；mAP为原生脚本输出。小样本AP退化风险保留，不能据此判定全量表现。'),
                h(1, '编译精度评测'), h(2, '板端产物与fp16'),
                p(f"{s['compile']['frames']} 帧；运行 {s['compile']['run_id']}；本次五份分网包串接。" +
                  (f"GT已按同一批留存预测计算，覆盖{s['compile']['classes_with_gt']}类；无GT样本仅表示本批未覆盖。" if s['compile']['gt_state'] == 'validated' else
                   '本次GT未计算，不能用张量一致性替代GT表现。') + '不混用旧v3.2.3结果。'),
                tables['compile_summary'], p('GT表现与余弦张量一致性分别读取，不计算混合总分；余弦以百分数显示。'),
                h(2, '输出 tensor 余弦对比（板端 vs ONNX）'), tables['compile_tensors'],
                '<source path="@./'+escape(str(attachment).replace('\\', '/'))+'" name="原表数据及来源.json"/>'])
    return ''.join(out)


def write_reference_report(output, *, model_report, compile_report, reference_file):
    root = Path(output)
    if root.exists():
        raise ValueError('Choose a new output directory to preserve prior reports')
    m, c, ref = [load_json(x) for x in (model_report, compile_report, reference_file)]
    from .validation import require_result
    for result in [m['source_result'], *m.get('head_results', {}).values(), c['source_normalized_result']]:
        require_result(result)
    data = organize(m, c, ref)
    data['inputs'] = [{'role': role, 'path': str(Path(path).resolve()), 'sha256': digest(Path(path))}
                     for role, path in [('model', model_report), ('compile', compile_report), ('reference', reference_file)]]
    data['atomic_results'] = {'model': deepcopy(m), 'compile': deepcopy(c)}
    save_json(root / 'data.json', data)
    (root / 'report.xml').write_text(render(data, 'data.json'), encoding='utf-8')
    return data
