"""One report for both atomic chains, including honest pending/failed sections."""
from copy import deepcopy
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

from .business_layout import table
from .common import digest, json_digest, load_json, save_json
from .reference_report import cell, display
from .strict_model_template import model_tables, render as render_model

LABELS = {'model_accuracy': '模型精度评测', 'compile_alignment': '编译精度评测'}


def _report(workflow, chain):
    if not chain.get('report'):
        return None
    job = next(j for j in workflow['jobs'] if j['kind'] == chain['kind'])
    native = chain.get('execution', {})
    if native.get('state') != 'succeeded' or not native.get('result_file'):
        raise ValueError('不能从未成功的执行中合并指标')
    if digest(Path(native['result_file'])) != native.get('result_sha256'):
        raise ValueError('本次评测结果已变化，停止合并报告')
    path = chain['report'].get('report_json') or chain['report'].get('json')
    value = load_json(path)
    if chain['kind'] == 'model_accuracy':
        version, run_id = value['version_name'], value['run_id']
        schema = 'roadmodel.model-single-report/v1'
    else:
        version, run_id = value['identity']['model_version'], value['identity']['run_id']
        schema = 'roadmodel.compile-single-report/v3'
    if (value.get('schema') != schema or version != job['version']
            or run_id != job['plan']['run_id']):
        raise ValueError('两条报告的版本或运行身份与本次任务不匹配')
    return value


def _compile_tables(compiled, spec):
    gt = compiled['ground_truth'] if compiled else None
    lookup = {r['name']: r for r in gt['classes']} if gt else {}
    rows = []
    for item in spec['gt_classes']:
        r = lookup.get(item['key'])
        state = ('missing' if not compiled else 'not_computed' if gt['state'] != 'validated'
                 else 'missing' if r is None else 'no_gt' if r['seen'] == 0 else 'measured')
        values = [display(cell((r or {}).get(k), state=state, unit='percent_number')) for k in ('onnx_iou', 'board_iou')]
        rows.append([item['label'], *values])
    for name, key in [('可行驶区域iou', 'drivable_iou'), ('present_miou', 'present_miou')]:
        matches = [r for r in gt['summary'] if r['metric_id'] == key] if gt else []
        r = matches[0] if len(matches) == 1 else {}
        state = 'missing' if not compiled else 'not_computed' if gt['state'] != 'validated' else 'measured'
        rows.append([name, *[display(cell(r.get(k), state=state, unit='percent_number')) for k in ('onnx', 'board')]])
    cosine = compiled['tensor_alignment']['mean_frame_mean_cosine'] if compiled else None
    rows.append(['余弦 mean(all tensors)', '—', display(cell(cosine*100 if cosine is not None else None, unit='percent_number'))])
    if not compiled:
        rows = [[r[0], *['待评测' for _ in r[1:]]] for r in rows]
    summary = table(['类别/指标', 'ONNX fp16', '本次板端'], rows)
    outputs = {r['name']: r for r in compiled['tensor_alignment']['outputs']} if compiled else {}
    tensors = []
    for item in spec['outputs']:
        r = outputs.get(item['key'])
        value = display(cell([r['mean_cosine']*100, r['min_cosine']*100] if r else None, unit='mean_min_percent')) if compiled else '待评测'
        tensors.append([item['key'], item['meaning'], item['group'], value])
    return [summary, table(['输出 tensor', '含义', '分组', '本次板端 mean/min'], tensors)]


def build_combined(workflow, result, reference, spec):
    if len(workflow['jobs']) != 2 or {j['kind'] for j in workflow['jobs']} != set(LABELS):
        raise ValueError('需要同一任务的模型和编译两条链路')
    versions = {j['version'] for j in workflow['jobs']}
    if len(versions) != 1:
        raise ValueError('禁止合并不同版本的模型和编译结果')
    version = next(iter(versions))
    chains = {c['kind']: c for c in result['chains']}
    if set(chains) != set(LABELS):
        raise ValueError('缺少本次某条链路的执行状态')
    reports = {kind: _report(workflow, chains[kind]) for kind in LABELS}
    model = reports['model_accuracy']
    # Empty cells are template placeholders, never a fabricated evaluation result.
    template_model = model or {'occ_sections': [], 'all_metrics': [], 'vision_metrics': []}
    tables = model_tables(template_model, reference, spec['model_accuracy'])
    body = render_model(reference, tables, model_version=version,
                        vision_samples=(model or {}).get('head_results', {}).get('vision', {}).get('coverage', {}).get('observed', {}).get('model_samples', 0))
    if not model:
        body = '<p>模型精度评测尚未完成：'+escape('；'.join(chains['model_accuracy'].get('issues', [])) or chains['model_accuracy']['state'])+'</p>'+body
    compiled = reports['compile_alignment']
    nodes = list(ET.fromstring('<root>'+reference+'</root>'))
    start = next(i for i, n in enumerate(nodes) if n.tag == 'h1' and ''.join(n.itertext()) == '板端产物与fp16')
    tail = deepcopy(nodes[start:])
    replacements = _compile_tables(compiled, spec['compile_alignment'])
    out, index = [], 0
    for node in tail:
        if node.tag == 'title' or node.tag in ('source', 'img', 'whiteboard', 'sheet', 'bitable'):
            continue
        if node.tag == 'table':
            if index >= 2:
                raise ValueError('编译模板表结构发生变化')
            node = ET.fromstring(replacements[index])
            index += 1
        for child in node.iter():
            child.attrib.pop('id', None)
            child.attrib.pop('seq-marker', None)
        out.append(ET.tostring(node, encoding='unicode'))
        if node.tag == 'h1' and not compiled:
            issues = '；'.join(chains['compile_alignment'].get('issues', [])) or chains['compile_alignment']['state']
            out.append('<p>编译精度评测尚未完成：'+escape(issues)+'</p>')
    if index != 2:
        raise ValueError('需要原PCM模板中的两张编译表')
    xml = '<title>PCM 单次评测报告 '+escape(version)+'</title>'+body+''.join(out)
    dataset_coverage = {}
    if workflow.get('head_datasets'):
        native = chains['model_accuracy'].get('execution', {})
        sources = workflow.get('head_datasets', {})
        for kind in ('occ','vision'):
            source = sources.get(kind, {})
            declared = native.get('head_datasets', {}).get(kind, {})
            if declared and (declared.get('version') != source.get('version')
                             or declared.get('index_sha256') != source.get('index_sha256')):
                raise ValueError('模型结果的数据版本或索引与本次对应输入不同')
            frames = native.get('executed_frames_by_head', {}).get(kind, [])
            dataset_coverage[kind] = {'version': source.get('version'), 'source_ref': source.get('source_ref'),
                'index_sha256': source.get('index_sha256'), 'scenes':len({r[0] for r in frames}),
                'frames':len(frames), 'selected_frames':frames}
        job = next(j for j in workflow['jobs'] if j['kind']=='compile_alignment')
        frames = job.get('plan', {}).get('frames', []) if compiled else []
        actual = compiled.get('coverage', {}).get('actual_frames', 0) if compiled else 0
        dataset_coverage['compile'] = {**{k:v for k,v in dataset_coverage['occ'].items()
                                        if k in ('version','source_ref','index_sha256')},
            'scenes':len({r[0] for r in frames}) if actual else 0, 'frames':actual}
        document = ET.fromstring('<root>'+xml+'</root>')
        for node in list(document):
            kind = {'occ头':'occ','视觉头':'vision','板端产物与fp16':'compile'}.get(''.join(node.itertext()))
            if node.tag in ('h1','h2') and kind:
                coverage = dataset_coverage[kind]
                note = ET.Element('p')
                note.text = f"评测数据集：{coverage['version']}；实际范围：{coverage['scenes']}场景 / {coverage['frames']}帧。"
                document.insert(list(document).index(node)+1,note)
        xml = ''.join(ET.tostring(n,encoding='unicode') for n in document)
    if len(list(ET.fromstring('<root>'+xml+'</root>').iter('table'))) != 5:
        raise ValueError('合并报告应保留原模型三表和编译两表')
    data = {'schema': 'roadmodel.combined-delivery/v1', 'version': version,
            'workflow_sha256': workflow['workflow_sha256'], 'inputs': workflow['inputs'],
            'scope': workflow['scope'], 'all_evaluations_complete': all(reports.values()) and all(c['state'] == 'SUCCEEDED' for c in chains.values()),
            'chain_states': {k: chains[k]['state'] for k in LABELS},
            'model_head_statuses': deepcopy(chains['model_accuracy'].get('execution', {}).get('head_statuses', {})),
            'atomic_reports': reports, 'new_inference_frames': 0}
    if dataset_coverage:
        data['dataset_coverage'] = dataset_coverage
    data['task_sha256'] = json_digest({'run_id': workflow['run_id'], 'inputs': workflow['inputs'],
        'scope': workflow['scope'], 'version': version,
        'frames': {j['kind']: (j.get('request') or {}).get('evaluation', {}) for j in workflow['jobs']}})
    return xml, data


def write_combined(workflow, result, output, *, reference_file, template_file):
    reference = load_json(reference_file)['data']['document']['content']
    xml, data = build_combined(workflow, result, reference, load_json(template_file))
    signature = json_digest({'xml': xml, 'data': data})
    root = Path(output)/signature[:20]
    if root.exists():
        if load_json(root/'report.json') != data or (root/'report.xml').read_text(encoding='utf-8') != xml:
            raise ValueError('已有合并报告内容变化')
    else:
        save_json(root/'report.json', data)
        (root/'report.xml').write_text(xml, encoding='utf-8')
    return {'report_xml': str(root/'report.xml'), 'report_json': str(root/'report.json'),
            'source_sha256': signature, 'all_evaluations_complete': data['all_evaluations_complete']}
