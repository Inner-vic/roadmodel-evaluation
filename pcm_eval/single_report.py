"""Legacy v2 layouts kept for historical reproduction and the pending model report.

Compile report generation delegates to the independent compile_report module.
"""
from html import escape
import math
from pathlib import Path

from .common import digest, json_digest, load_json, save_json, utc_now

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / 'configs/report_template.json'


def cell(result, name, unit, **dimensions):
    rows = [m for m in (result or {}).get('metrics', []) if m['name'] == name
            and all(m.get('dimensions', {}).get(k) == v for k, v in dimensions.items())]
    valid = [m for m in rows if not m.get('validity', '').startswith('invalid')]
    if len(valid) != 1:
        return {'value': None, 'unit': unit, 'status': 'invalid' if rows and not valid else 'missing',
                'metric': name, 'dimensions': dimensions}
    metric = valid[0]
    value = metric['value']
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Report metrics must be finite numeric values')
    if metric['unit'] == 'ratio' and unit == 'percent':
        value *= 100
    elif metric['unit'] != unit:
        raise ValueError('Metric unit mismatch: ' + name)
    return {'value': value, 'unit': unit, 'status': 'available', 'metric': name,
            'dimensions': dimensions, 'source': metric.get('source'), 'source_unit': metric['unit']}


def gt_cell(result, backend, class_name):
    value = cell(result, 'occ.class.iou', 'percent', backend=backend, class_name=class_name)
    support = (result or {}).get('provenance', {}).get('gt_class_support', {}).get(backend, [])
    rows = [r for r in support if r.get('name') == class_name]
    if len(rows) == 1:
        value['support'] = {k:rows[0][k] for k in ('seen','correct','positive')}
        if value['status'] == 'missing' and rows[0].get('seen') == 0 and rows[0].get('status') == 'no_gt_samples':
            value['status'] = 'no_gt_samples'
    return value


def table(key, title, columns, rows):
    return {'key': key, 'title': title, 'columns': columns, 'rows': rows}


def build_single_report(kind, result=None, *, readiness=None, contract=None, request=None):
    if kind not in ('model_accuracy', 'compile_alignment'):
        raise ValueError('Unknown evaluation kind')
    if result is not None and (result.get('schema') != 'pcm.single-eval/v1' or result.get('evaluation_type') != kind):
        raise ValueError('A single report must contain exactly one matching evaluation type')
    if request and request.get('kind') != kind:
        raise ValueError('Request and report evaluation types differ')
    contract = contract or load_json(DEFAULT_TEMPLATE)
    if contract.get('schema') != 'pcm.single-report-template/v1':
        raise ValueError('Wrong report template schema')
    provenance = (result or {}).get('provenance', {})
    live, plan = provenance.get('live_run', {}), provenance.get('execution_plan', {})
    request = request or {}
    if result and request.get('version_name') and request['version_name'] != result.get('version_name'):
        raise ValueError('Report request and executed version differ')
    for key in ('onnx_sha256', 'packup_sha256'):
        if live.get(key) and request.get('model', {}).get(key) and live[key] != request['model'][key]:
            raise ValueError('Report request differs from executed model identity')
    raw_version = (result or {}).get('version_name') or request.get('version_name')
    selected = live.get('selected_frames')
    identity = {'version_name': raw_version,
                'model_version': raw_version.removeprefix('supplied-reference-') if raw_version else None,
                'run_id': (result or {}).get('execution', {}).get('run_id') or live.get('run_id'),
                'dataset': plan.get('dataset') or request.get('dataset', {}),
                'data_link_sha256': live.get('data_link_sha256') or request.get('dataset', {}).get('data_link_sha256'),
                'sample_selection_sha256': json_digest(selected) if selected else None,
                'selected_frames': selected, 'selected_scenes': live.get('selected_scenes'),
                'onnx_sha256': live.get('onnx_sha256'), 'packup_sha256': live.get('packup_sha256'),
                'model_inputs': request.get('model', {}),
                'code_commit': request.get('source', {}).get('commit_id'),
                'started_at': live.get('started_at'), 'finished_at': live.get('finished_at'),
                'historical_result': provenance.get('lpai_task', {}).get('historical_result', False),
                'reference_only': plan.get('source', {}).get('reference_only', False)}
    coverage = (result or {}).get('coverage', {'expected_frames': None, 'observed': {}})
    spec, tables = contract[kind], []
    if kind == 'compile_alignment':
        gt = [{'key': r['key'], 'label': r['label'],
               **{b: gt_cell(result, b, r['key']) for b in ('onnx', 'board')}}
              for r in spec['gt_classes']]
        # Supplied bridge occ_iou means nonempty IoU, not the template's drivable IoU.
        for key, metric in [('drivable', 'drivable_iou'), ('present_miou', 'present_miou')]:
            gt.append({'key': key, 'label': spec['gt_summary_labels'][key],
                       **{b: cell(result, metric, 'percent', backend=b) for b in ('onnx', 'board')}})
        gt.append({'key': 'cosine_all', 'label': spec['gt_summary_labels']['cosine'],
                   'onnx': {'value': None, 'unit': 'percent', 'status': 'not_applicable'},
                   'board': cell(result, 'tensor.frame_mean_cosine.mean', 'percent')})
        tables.append(table('compile_summary', '对齐部分（fp16）',
                            [('label', '类别/指标'), ('onnx', 'ONNX fp16（%）'), ('board', '本次板端（%）')], gt))
        outputs = [{**r, 'mean': cell(result, 'tensor.mean_cosine', 'percent', output=r['key']),
                    'min': cell(result, 'tensor.min_cosine', 'percent', output=r['key']),
                    'frames': coverage.get('tensor_frames', {}).get(r['key'])} for r in spec['outputs']]
        for row in outputs:
            row['mean_min'] = display(row['mean']) + ' / ' + display(row['min'])
        tables.append(table('output_cosine', '输出 tensor 余弦对比（板端 vs ONNX）',
                            [('key', '输出 tensor'), ('meaning', '含义'), ('group', '分组'),
                             ('mean_min', '本次板端 mean/min（%）')], outputs))
        notes = ['沿用原模板的GT指标表、tensor含义和分组；只列本次实际执行的编译产物，不添加未执行的4tile/20tile列。',
                 '余弦转换为百分数展示；mean/min是每个输出在本次评测帧上的均值/最小值，不是业务准确率。',
                 '“—”表示未取得有效结果，不填0、不复制模板历史分数。',
                 '可行驶区域IoU与非空IoU分别存储；不将bridge的occ_iou当作可行驶区域IoU。']
        gt_state = provenance.get('input_validation', {}).get('ground_truth', {}).get('state', 'not_available')
        if gt_state == 'not_requested':
            notes.append('本次GT指标未评测；冷启动非时序输入，时序输出参与余弦不代表完成时序场景评测。')
        if gt_state == 'validated' and provenance.get('gt_adaptation'):
            notes += ['GT与余弦覆盖同一组10场景20帧；复用留存预测补算，未新增推理、未跑全量。',
                      'V161稀疏GT源网格800×48×200，原点(5,-3.6,-2)m；按同分辨率(0.05,0.15,0.02)m裁到400×48×100、原点(5,-3.6,-1)m，Z索引减50；第五列保留为可行驶标注。',
                      'GT忽略类别0(ignore)和15(road_noise)，保留空类26；逐类IoU先累计20帧TP/FP/FN再求比值。present mIoU仅平均GT出现过的24类中的类别，不含空类和缺席类。',
                      '可行驶区域IoU映射到固定源码MeanIoU.single_layer_ds_count(probs=None)：使用GT第五列及22个地面/路面设施类别，按首个预测可行驶Z层计算交集；该分支适用于本次仅留存final_occ、未留存概率体的结果。',
                      '“无GT样本”表示这20帧裁剪后的GT中该类体素数为0，未填造0%或100%，不计入present mIoU；结构化附件保留GT与预测支持数。',
                      'GT核验独立使用数据集height_field物理坐标；两端final_occ均为交付脚本的CPU聚合结果。本次用于适配和链路验证，不构成发版通过结论。',
                      '输入仍为冷启动非时序；时序输出余弦不代表已完成时序场景评测。']
    else:
        occ = []
        for r in spec['occ_classes']:
            occ.append({**r, **{k: cell(result, 'occ.' + k, 'percent', class_name=r['key'], scope='EVAL')
                               for k in ('iou', 'precision', 'recall')},
                        'pillar': cell(result, 'occ.iou', 'percent', class_name=r['key'], scope='EVAL_PILLAR')})
        occ += [{'key': 'drivable', 'label': spec['occ_summary_labels']['drivable'],
                 **{k: cell(result, 'occ.drivable_' + k, 'percent') for k in ('iou', 'precision', 'recall')},
                 'pillar': cell(result, 'pillar.drivable_iou', 'percent')},
                {'key': 'nonempty', 'label': spec['occ_summary_labels']['nonempty'],
                 **{k: cell(result, 'occ.' + k, 'percent', class_id='NN', scope='EVAL') for k in ('iou', 'precision', 'recall')},
                 'pillar': cell(result, 'occ.iou', 'percent', class_id='NN', scope='EVAL_PILLAR')}]
        tables.append(table('occ', 'occ头', [('label', '类别'), ('iou', 'IoU（%）'), ('precision', 'precision（%）'),
                            ('recall', 'recall（%）'), ('pillar', 'Pillar IoU（%）')], occ))
        vision = [{**r, **{k: cell(result, 'vision.' + metric, unit, task=r['key']) for k, metric, unit in
                           [('map', 'map', 'ratio'), ('accuracy', 'accuracy', 'percent'),
                            ('macro_recall', 'macro_recall', 'percent'), ('major', 'major_classes', 'count')]}}
                  for r in spec['vision_tasks']]
        vision.append({'key': 'avg_map', 'label': 'avg mAP（总体）', 'map': cell(result, 'vision.avg_map', 'ratio')})
        tables.append(table('vision', '视觉头：父类汇总', [('label', '类别'), ('map', 'mAP'), ('accuracy', '准确率（%）'),
                            ('macro_recall', 'macro_recall（%）'), ('major', '参与子类数')], vision))
        labels = {r['key']: r['label'] for r in spec['vision_tasks']}
        major = []
        for r in spec['vision_major_classes']:
            dims = {'task': r['task'], 'class_id': r['class_id'], 'included_in_map': True}
            major.append({**r, 'parent': labels[r['task']], 'class_label': 'class_' + r['class_id'] + ' ' + r['label'],
                          **{k: cell(result, 'vision.class.' + metric, unit, **dims) for k, metric, unit in
                             [('support', 'support', 'count'), ('share', 'support_percent', 'percent'),
                              ('recall', 'recall', 'percent'), ('ap', 'ap', 'ratio')]}})
        tables.append(table('vision_major', '参与 mAP 的子类明细（major 类）', [('parent', '父类'), ('class_label', '子类'),
                            ('support', '样本数'), ('share', '占比（%）'), ('recall', 'recall（%）'), ('ap', 'AP')], major))
        notes = ['沿用原模板occ头、轨迹头、视觉头及major子类章节；每个指标只保留本次一个版本的值。',
                 'mAP/AP保留0–1比例；IoU、precision、recall、准确率及占比按百分数展示。',
                 'major类沿用模板口径：验证集占比≥5%的子类参与父类mAP；未执行时不计算任何占位均值。',
                 '轨迹头保留原模板章节；未取得实际评测配置和结果，不复制旧版本“默认不发”的结论。']
        gt_state = 'not_applicable'
    cells = [v for t in tables for row in t['rows'] for v in row.values() if isinstance(v, dict) and 'status' in v]
    return {'schema': 'pcm.single-report/v2', 'evaluation_type': kind, 'created_at': utc_now(),
            'template': contract['source'], 'identity': identity, 'coverage': coverage,
            'execution': (result or {}).get('execution', {'state': 'not_run'}),
            'collection': (result or {}).get('collection', {'status': 'empty', 'issues': []}),
            'report_status': 'template_only' if result is None else
                             'complete' if result.get('collection', {}).get('status') == 'complete'
                             and all(c['status'] in ('available', 'not_applicable', 'no_gt_samples') for c in cells) else 'partial',
            'ground_truth_status': gt_state, 'tables': tables, 'notes': notes, 'readiness': readiness,
            'ground_truth': provenance.get('gt_adaptation'),
            'filled_metric_cells': sum(c['status'] == 'available' for c in cells),
            'quality_assessment': 'not_assessed',
            'comparison': {'state': 'not_requested', 'release_inputs': 'two same-kind single reports from two runs/versions'},
            'raw_metrics': (result or {}).get('metrics', [])}


def display(value):
    if not isinstance(value, dict):
        return str(value) if value is not None else '—'
    if value.get('status') != 'available':
        return '无GT样本' if value.get('status') == 'no_gt_samples' else '—'
    n = value['value']
    return str(int(n)) if value['unit'] == 'count' else format(n, '.4f' if value['unit'] == 'ratio' else '.2f')


def render_xml(report, attachment=None):
    p = lambda s: '<p>' + escape(str(s)) + '</p>'
    kind, identity = report['evaluation_type'], report['identity']
    title = 'PCM 单次' + ('编译精度' if kind == 'compile_alignment' else '模型精度') + '评测报告 '
    title += identity['model_version'] or '待测版本'
    if report['report_status'] == 'template_only': title += '（模板·待评测）'
    parts = ['<title>' + escape(title) + '</title>', '<blockquote>' + p('模型版本：' +
             (identity['model_version'] or '待提供')) + p('运行ID：' + (identity['run_id'] or '待评测')) + '</blockquote>',
             '<h1 seq="auto">评测信息</h1>']
    observed = report['coverage']['observed']
    states = {'succeeded':'已完成','not_run':'待评测','failed':'执行失败','unknown':'未确认'}
    statuses = {'partial':'部分填报（未取得的指标留空）','complete':'已填报','template_only':'模板·待评测'}
    info = [('评测类型', '板端编译产物与ONNX FP16对齐' if kind == 'compile_alignment' else 'LPAI模型精度'),
            ('执行状态', states.get(report['execution']['state'],report['execution']['state'])),
            ('报告填报状态', statuses[report['report_status']]),
            ('数据集', identity['dataset'].get('class_name', '待提供')),
            ('数据划分文件', Path(identity['dataset'].get('data_link_file', '')).name or '待提供'),
            ('实际评测数量', observed.get('tensor_comparison') if kind == 'compile_alignment' else observed.get('model_samples'))]
    if identity['selected_scenes']: info.append(('场景数', len(identity['selected_scenes'])))
    if kind == 'model_accuracy':
        info += [('PTH权重', identity['model_inputs'].get('checkpoint') or '待提供'),
                 ('模型配置', identity['model_inputs'].get('config') or '待提供'),
                 ('匹配代码提交', identity['code_commit'] or '待确认')]
    if identity['started_at']: info.append(('开始时间（UTC）', identity['started_at']))
    if identity['finished_at']: info.append(('结束时间（UTC）', identity['finished_at']))
    if identity['model_inputs'].get('packup_tar'):
        info.append(('本次编译产物', Path(identity['model_inputs']['packup_tar']).name))
    parts.append('<table><tbody>' + ''.join('<tr><td>'+p(k)+'</td><td>'+p(v if v is not None else '未评测')+'</td></tr>' for k,v in info) + '</tbody></table>')
    if identity['reference_only']: parts.append(p('本次为提供的参考模型小场景评测，无历史版本对比和发版通过判定。'))
    if identity['historical_result']: parts.append(p('数据来源为历史任务；不作为新触发的评测结果。'))
    def emit(t):
        return '<table><thead><tr>' + ''.join('<th>'+p(c[1])+'</th>' for c in t['columns']) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join('<td>'+p(display(row.get(c[0])))+'</td>' for c in t['columns']) + '</tr>' for row in t['rows']) + '</tbody></table>'
    if kind == 'compile_alignment':
        parts += ['<h1 seq="auto">对齐部分（fp16）</h1>', emit(report['tables'][0]),
                  p(('GT分类指标本次未评测，表中保留“—”；' if report['ground_truth_status']=='not_requested'
                     else 'GT已按本次编译产物的ROI适配并补算；“无GT样本”表示小场景中该类别GT支持数为0；' if report['ground_truth_status']=='validated'
                     else '表中“—”表示未取得有效结果；')+'余弦行是板端相对ONNX的相似度。'),
                  '<h2 seq="auto">输出 tensor 余弦对比（板端 vs ONNX）</h2>', emit(report['tables'][1])]
    else:
        parts += ['<h1 seq="auto">开发空间模型评测</h1>', '<h2 seq="auto">occ头</h2>', emit(report['tables'][0]),
                  '<h2 seq="auto">轨迹头</h2>', p('本次是否评测/发版：待实际配置确认；当前未取得轨迹头指标。'),
                  '<h2 seq="auto">视觉头</h2>', emit(report['tables'][1]),
                  '<h3 seq="auto">参与 mAP 的子类明细（major 类）</h3>', emit(report['tables'][2])]
    parts.append('<h1 seq="auto">统计说明与评测产物</h1><ul>' + ''.join('<li>'+escape(n)+'</li>' for n in report['notes']) + '</ul>')
    for field in ('onnx_sha256', 'packup_sha256', 'data_link_sha256', 'sample_selection_sha256'):
        if identity.get(field): parts.append(p(field+'：'+identity[field]))
    parts.append('<p>参考模板：<a href="'+escape(report['template']['url'],quote=True)+'">PCM发版报告模板</a>；本报告仅填本次结果，后续同类型的两份单次报告用于版本对比。</p>')
    if attachment: parts.append('<source path="'+escape(attachment,quote=True)+'" name="单次评测报告结构化结果.json"/>')
    return '\n'.join(parts)


def write_report(kind, output, *, result_file=None, readiness_file=None, template_only=False, template_file=None, request_file=None):
    if kind == 'compile_alignment':
        if not result_file or template_only:
            raise ValueError('An independent compile report requires one executed result')
        if template_file:
            raise ValueError('Compile reports do not consume a release-report template')
        from .compile_report import write_compile_report
        return write_compile_report(output,result_file=result_file,request_file=request_file)
    if (result_file is None) != template_only:
        raise ValueError('Provide one result file, or explicitly request --template-only')
    report = build_single_report(kind, load_json(result_file) if result_file else None,
                                 readiness=load_json(readiness_file) if readiness_file else None,
                                 contract=load_json(template_file or DEFAULT_TEMPLATE),
                                 request=load_json(request_file) if request_file else None)
    report['sources'] = [{'path': str(Path(p).resolve()), 'sha256': digest(Path(p))}
                         for p in (result_file, readiness_file, request_file, template_file or DEFAULT_TEMPLATE) if p]
    output = Path(output).resolve()
    if output.suffix != '.xml' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Use a new XML draft path and JSON sidecar')
    relative = output.with_suffix('.json').relative_to(Path.cwd().resolve()).as_posix()
    save_json(output.with_suffix('.json'), report)
    output.write_text(render_xml(report, '@./'+relative), encoding='utf-8')
    return {'xml': str(output), 'json': str(output.with_suffix('.json')), 'evaluation_type': kind,
            'report_status': report['report_status'], 'filled_metric_cells': report['filled_metric_cells']}
