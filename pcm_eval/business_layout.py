"""Static single-version business tables derived from the supplied PCM report.

The source report is read at design time, never fetched during generation.
Cross-version columns are removed; within-run ONNX/board comparisons remain.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

from .common import load_json

TEMPLATE = Path(__file__).resolve().parents[1] / 'configs/report_template.json'


def p(value):
    return '<p>' + escape(str(value)) + '</p>'


def fmt(value, digits=2):
    return '未取得' if value is None else f'{value:.{digits}f}' if type(value) in (int, float) else str(value)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + p(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + p(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def replace_chapter(xml, title, body):
    root = ET.fromstring('<root>' + xml + '</root>')
    nodes = list(root)
    starts = [i for i, x in enumerate(nodes) if x.tag == 'h1' and ''.join(x.itertext()) == title]
    if len(starts) != 1:
        raise ValueError('Expected one model report chapter: ' + title)
    start = starts[0]
    end = next((i for i in range(start + 1, len(nodes)) if nodes[i].tag == 'h1'), len(nodes))
    content = ET.fromstring('<root>' + body + '</root>')
    nodes[start + 1:end] = list(content)
    return ''.join(ET.tostring(x, encoding='unicode') for x in nodes)


def render_model_business(report, diagnostic_xml):
    spec = load_json(TEMPLATE)['model_accuracy']
    metrics = report['all_metrics']
    def value(name, **dims):
        matches = [m for m in metrics if m['name'] == name and all(m.get('dimensions', {}).get(k) == v for k, v in dims.items())]
        if len(matches) != 1:
            return None
        return matches[0]['value']
    sections = {s['scope']: {r['class_name']: r for r in s['classes']} for s in report['occ_sections']}
    occ, pillar = sections.get('EVAL', {}), sections.get('EVAL_PILLAR', {})
    rows = []
    for item in spec['occ_classes']:
        a, b = occ.get(item['key']), pillar.get(item['key'])
        def cell(row, key):
            return '未取得' if not row else '无GT样本' if not row['seen'] else fmt(row.get(key))
        rows.append([item['label'], cell(a, 'exact_iou'), cell(a, 'precision'), cell(a, 'recall'), cell(b, 'exact_iou')])
    rows.append(['可行驶区域 IoU', fmt(value('drivable.iou')), '不适用', '不适用', '未取得'])
    nn = {s['scope']: s['non_empty'] for s in report['occ_sections']}
    a, b = nn.get('EVAL') or {}, nn.get('EVAL_PILLAR') or {}
    rows.append(['非空 IoU', fmt(a.get('exact_iou')), fmt(a.get('precision')), fmt(a.get('recall')), fmt(b.get('exact_iou'))])
    body = '<h2>occ头</h2>' + table(['类别', 'IoU（%）', 'precision（%）', 'recall（%）', 'Pillar IoU（%）'], rows)
    body += p('每个指标仅列本次版本的一个数值。无GT样本与实测0%分别表示；原始全部类别、支持数及计数保留在结构化结果。')
    body += table(['统计项', '本次结果'], [[('OCC' if s['scope'] == 'EVAL' else 'Pillar') + ' present mIoU（%）', fmt(s['present_miou'])] for s in report['occ_sections']])
    xml = replace_chapter(diagnostic_xml, 'OCC 与柱状结果', body)
    task_names = {x['key']: x['label'] for x in spec['vision_tasks']}
    totals = {m.get('dimensions', {}).get('task'): m['value'] for m in metrics if m['name'] == 'vision.total'}
    task_rows = []
    for item in spec['vision_tasks']:
        task = item['key']
        task_rows.append([item['label'] + ' (' + task + ')', fmt(value('vision.map', task=task), 4),
                          fmt(value('vision.accuracy', task=task)), fmt(value('vision.macro_recall', task=task)),
                          fmt(value('vision.major_classes', task=task), 0)])
    task_rows.append(['avg mAP（总体）', fmt(value('vision.avg_map'), 4), '不适用', '不适用', '不适用'])
    vision = '<h2>视觉头</h2><h3>核心类 mAP 与准确率</h3>' + table(['类别', 'mAP', '准确率（%）', 'macro_recall（%）', '参与子类数'], task_rows)
    classes = {}
    for m in metrics:
        d = m.get('dimensions', {})
        if m['name'].startswith('vision.class.') and 'class_id' in d:
            key = (d['task'], str(d['class_id']))
            classes.setdefault(key, {'included': d.get('included_in_map')})[m['name'].removeprefix('vision.class.')] = m['value']
    labels = {(x['task'], x['class_id']): x['label'] for x in spec['vision_major_classes']}
    details = []
    for (task, cid), m in sorted(classes.items()):
        support = m.get('support')
        if support is None or support == 0:
            continue
        share = m.get('support_percent')
        if share is None and totals.get(task):
            share = support / totals[task] * 100
        details.append([task_names.get(task, task), 'class_' + cid + ' ' + labels.get((task, cid), ''),
                        str(support), fmt(share), fmt(m.get('recall')), fmt(m.get('ap'), 4),
                        '是' if m['included'] is True else '否' if m['included'] is False else '见原生口径'])
    vision += '<h3>子类明细</h3>' + table(['父类', '子类', '样本数', '占比（%）', 'recall（%）', 'AP', '计入 mAP'], details)
    vision += p('主要类筛选沿用本次评测脚本和配置；未提供参与子类数或全局mAP时明确留缺，不从少量样本自行重定义业务指标。单一GT类别的AP可能退化，需结合准确率与支持数阅读。')
    from .model_report_heads import scope_xml
    if 'vision' in report.get('head_results', {}):
        vision = scope_xml(report['head_results']['vision']) + vision
    if not report['vision_metrics']:
        vision = p('本次未评测：未取得视觉分类结果。') + vision
    return replace_chapter(xml, '视觉分类结果', vision)


def compile_business_sections(report, diagnostic_sections):
    spec = load_json(TEMPLATE)['compile_alignment']
    gt = report['ground_truth']
    lookup = {row['name']: row for row in gt['classes']}
    rows = []
    for item in spec['gt_classes']:
        row = lookup.get(item['key'])
        state = '未取得' if row is None else '无GT样本' if not row['seen'] else None
        rows.append([item['label'], state or fmt(row['onnx_iou']), state or fmt(row['board_iou'])])
    rows.extend([r['label'], fmt(r['onnx']), fmt(r['board'])] for r in gt['summary'])
    body = '<h2>板端产物与fp16：本次GT指标</h2>' + table(['类别/指标', 'ONNX fp16（%）', '本次板端（%）'], rows)
    body += p('两列属于同一次评测的两个后端，不是两个版本。只列实际执行的板端组合；其余tile配置未执行时不生成对应列。')
    body += p('GT指标与张量一致性分别展示，不混合计算总分。全部类别及两端支持数仍保留在结构化报告和证据包。')
    tensor_rows = [[r['name'], r['label'], r['group'], fmt(r['mean_cosine'] * 100, 4), fmt(r['min_cosine'] * 100, 4)]
                   for r in report['tensor_alignment']['outputs']]
    tensor = '<h2>输出 tensor 余弦对比（板端 vs ONNX）</h2>' + table(
        ['输出 tensor', '含义', '分组', '本次板端 mean（%）', '本次板端 min（%）'], tensor_rows)
    tensor += p('余弦按原业务报告以百分数展示，JSON保留无量纲原值；原始逐帧比较、MAE与最大误差不丢弃。')
    integers=[r for r in report['tensor_alignment']['records'] if r.get('compare_dtype')=='int32' and r.get('integer_exact_ratio') is not None]
    if integers:
        tensor+='<h2>整数输出逐元素一致率</h2>'+table(['输出','一致率（%）','最大整数差'],[
            [r['onnx_output_name'],fmt(r['integer_exact_ratio']*100,3),fmt(r['max_abs_diff'])] for r in integers])
        tensor+=p('余弦接近1不代表整数输出逐元素完全相同。')
    result = []
    for section in diagnostic_sections:
        copy = dict(section)
        if copy['id'] in ('tensor', 'gt'):
            heading = '<h1 seq="auto">' + escape(copy['title']) + '</h1>'
            copy['xml'] = heading + (tensor if copy['id'] == 'tensor' else body if gt['state'] == 'validated' else p('本次GT未评测，不能从张量一致性推断GT精度。'))
        result.append(copy)
    return result
