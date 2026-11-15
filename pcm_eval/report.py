"""Historical local debugging view; formal reports use pcm_eval.single_report."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
from pathlib import Path

from .common import digest, load_json, save_json, utc_now


def group(result, kind):
    if result is None:
        return {'state': 'not_run', 'metrics': [], 'quality_assessment': 'not_assessed'}
    if result.get('schema') != 'pcm.single-eval/v1' or result.get('evaluation_type') != kind:
        raise ValueError('Wrong single-result schema or evaluation type: ' + kind)
    if any(not isinstance(m.get('value'), (int, float)) or isinstance(m['value'], bool)
           or not math.isfinite(m['value']) for m in result.get('metrics', [])):
        raise ValueError('Report metrics must be finite numeric values')
    invalid = [m for m in result['metrics'] if m.get('validity', '').startswith('invalid')]
    return {'state': 'complete' if result['collection']['status'] == 'complete' and not invalid else 'partial',
            'version_name': result.get('version_name'), 'execution': result['execution'],
            'coverage': result['coverage'], 'collection': result['collection'], 'metrics': result['metrics'],
            'provenance': result.get('provenance', {}), 'quality_assessment': 'not_assessed'}


def build_report(compile_result, model_result=None, *, analysis=None, readiness=None, checks=None):
    groups = {'model_accuracy': group(model_result, 'model_accuracy'),
              'compile_alignment': group(compile_result, 'compile_alignment')}
    return {'schema': 'pcm.single-report/v1', 'created_at': utc_now(), 'evaluation_groups': groups,
            'model_readiness': readiness, 'compile_difference_analysis': analysis,
            'integration_checks': checks or [],
            'model_artifact_relationship': {'state': 'not_verified',
                                            'reason': 'No verified PTH-to-ONNX export binding was supplied'},
            'quality_assessment': 'not_assessed', 'comparison': {'state': 'not_requested'}}


def render_html(report):
    e = lambda value: escape(str(value), quote=True)
    model, compile = (report['evaluation_groups'][k] for k in ('model_accuracy', 'compile_alignment'))
    metrics = compile['metrics']
    def metric(name):
        values = [m['value'] for m in metrics if m['name'] == name and not m.get('dimensions')
                  and not m.get('validity', '').startswith('invalid')]
        return '{:.6f}'.format(values[0]) if len(values) == 1 else '未提供'
    observed = compile.get('coverage', {}).get('observed', {}).get('tensor_comparison', '未提供')
    by_output = {}
    for m in metrics:
        if m.get('validity', '').startswith('invalid'):
            continue
        output = m.get('dimensions', {}).get('output')
        if output and m['name'] in ('tensor.mean_cosine', 'tensor.min_cosine'):
            by_output.setdefault(output, {})[m['name']] = m['value']
    tensor_rows = ''.join('<tr><td><code>' + e(name) + '</code></td><td>'
                          + e('{:.6f}'.format(values['tensor.mean_cosine'])) + '</td><td>'
                          + e('{:.6f}'.format(values['tensor.min_cosine'])) + '</td></tr>'
                          for name, values in sorted(by_output.items(), key=lambda kv: kv[1].get('tensor.min_cosine', 2))
                          if 'tensor.mean_cosine' in values and 'tensor.min_cosine' in values)
    model_body = '<p>尚未执行。缺失信息保留为空，不填入历史训练指标。</p>'
    if report.get('model_readiness'):
        missing = report['model_readiness'].get('missing_fields', [])
        labels = {'version_name': '待测模型版本', 'execution.repo_dir': 'LPAI 代码目录',
                  'execution.output_root': '评测输出目录', 'model.checkpoint': 'PTH 权重路径',
                  'source.commit_id': '匹配的代码提交', 'model.checkpoint_sha256': '权重哈希',
                  'model.config_sha256': '配置文件哈希'}
        model_body += '<p>待补齐：' + '、'.join(e(labels.get(x, x)) for x in missing) + '</p>'
    if model['state'] != 'not_run':
        rows = ''.join('<tr><td>' + e(m['name']) + '</td><td>' + e(m['value']) + '</td><td>'
                       + e(m['unit']) + '</td></tr>' for m in model['metrics']
                       if not m.get('validity', '').startswith('invalid'))
        model_body = '<p>版本：' + e(model.get('version_name')) + '；采集状态：' + e(model['state']) + '</p>'
        if model.get('provenance', {}).get('lpai_task', {}).get('historical_result'):
            model_body += '<p>来源标记为历史任务结果。</p>'
        model_body += '<table><tr><th>指标</th><th>值</th><th>单位</th></tr>' + rows + '</table>'
    differences = ''
    for row in (report.get('compile_difference_analysis') or {}).get('checks', []):
        extra = row.get('last_axis_argmax')
        text = ('；最后一维最大值索引不同 ' + str(extra['different']) + '/' + str(extra['count'])) if extra else ''
        differences += '<tr><td><code>' + e(row['output']) + '</code><br>' + e(row['frame_id']) + '</td><td>'
        differences += e('{:.6f}'.format(row['cosine'])) + '</td><td>'
        differences += e('{:.6f}'.format(row['mean_abs_diff'])) + '</td><td>'
        differences += e('{:.6f}'.format(row['max_abs_diff'])) + e(text) + '</td></tr>'
    checks = ''
    for check in report.get('integration_checks', []):
        if check.get('schema') == 'pcm.repeatability-check/v1':
            checks += '<p><b>同输入复跑：</b>' + e(check['frames']) + ' 帧；'
            checks += e(check['input_checks']) + ' 份输入哈希' + ('全部一致' if check['inputs_identical'] else '存在差异') + '；'
            checks += e(check['output_checks']) + ' 份两端输出哈希' + ('全部一致' if check['outputs_identical'] else '存在差异') + '。</p>'
        elif check.get('schema') == 'pcm.batch-resume-verification/v1':
            checks += '<p><b>分批续跑：</b>' + e(check['batches']) + ' 批、' + e(check['frames']) + ' 帧；'
            checks += '已完成批次' + ('未重复执行' if check['first_batch_attempts_unchanged'] else '发生变化') + '；'
            checks += e(check['finite_outputs']) + ' 份输出有限值检查通过。原始产物全部保留。</p>'
            checks += '<p>失败重试已通过单元测试；本次没有人为制造实板失败。</p>'
        else:
            checks += '<pre>' + e(json.dumps(check, ensure_ascii=False, indent=2)) + '</pre>'
    time = datetime.fromisoformat(report['created_at']).astimezone(timezone(timedelta(hours=8)))
    version = compile.get('version_name', '未绑定')
    reference = compile.get('provenance', {}).get('execution_plan', {}).get('source', {}).get('reference_only')
    identity = compile.get('provenance', {}).get('live_run', {})
    identity_rows = ''.join('<tr><th>' + e(k) + '</th><td><code>' + e(identity[k]) + '</code></td></tr>'
                            for k in ('run_id', 'onnx_sha256', 'packup_sha256', 'data_link_sha256') if k in identity)
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PCM 单次评测报告</title><style>
body{font:15px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#1e293b;background:#f1f5f9;margin:0}
main{max-width:1080px;margin:32px auto;padding:0 24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}
h1{font-size:28px;margin-bottom:4px}h2{font-size:21px}h3{font-size:16px}p{margin:10px 0}.muted{color:#64748b}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #e2e8f0;vertical-align:top}
th{color:#475569;background:#f8fafc}code{overflow-wrap:anywhere;font-size:12px}pre{white-space:pre-wrap;font-size:12px;background:#f8fafc;padding:14px}
.stats{display:flex;gap:32px;flex-wrap:wrap}.stats b{font-size:26px;display:block}.note{border-left:4px solid #d97706;padding:10px 14px;background:#fffbeb}
@media print{body{background:white}section{break-inside:avoid;margin:12px 0;padding:12px}main{margin:0;padding:0}}
</style><main><h1>PCM 单次评测报告</h1><p class="muted">''' + e(time.strftime('%Y-%m-%d %H:%M:%S UTC+08:00')) + '''</p>
<p>版本：''' + e(version) + (' · 提供的参考模型' if reference else '') + '''</p>
<p class="note">质量判定：未评定。报告分别展示模型精度与编译对齐；未进行基线版本比较。</p>
<section><h2>1. 模型精度 · LPAI</h2>''' + model_body + '''</section>
<section><h2>2. 编译精度 · 板端 / FP16 ONNX</h2><p>执行：''' + e(compile.get('execution', {}).get('state', 'unknown')) + '；采集：' + e(compile['state']) + '''</p>
<div class="stats"><div><b>''' + e(observed) + '''</b>实际评测帧</div><div><b>''' + e(len(by_output)) + '''</b>输出张量</div><div><b>''' + metric('tensor.frame_mean_cosine.mean') + '''</b>平均逐帧平均余弦</div></div>
<p>本报告只覆盖所选样本。正式发版评测范围与时长待业务方确认。GT 指标状态见原始结果。</p>
<h3>运行与模型身份</h3><table>''' + identity_rows + '''</table>
<h3>各输出余弦</h3><table><tr><th>输出</th><th>平均</th><th>最低</th></tr>''' + tensor_rows + '''</table></section>
<section><h2>3. 差异明细</h2><p>按每个输出的最低余弦排序。最大值索引变化是张量诊断，不能当作真值准确率或最终车辆行为。</p>
<table><tr><th>输出 / 帧</th><th>余弦</th><th>平均绝对误差</th><th>最大绝对误差 / 诊断</th></tr>''' + differences + '''</table>
<p>差异原因尚未定位到具体算子或编译步骤，不能仅归因于 FP16 舍入。</p></section>
<section><h2>4. 执行验证</h2>''' + (checks or '<p>未附加其他执行验证。</p>') + '''</section>
<p class="muted">PTH 与 ONNX 的导出对应关系待核验。各项原始指标来源、范围及校验信息保存在同名 JSON 文件中。</p></main></html>'''


def write_report(compile_file, output, *, model_file=None, analysis_file=None, readiness_file=None, check_files=()):
    paths = [Path(p).resolve() for p in (compile_file, model_file, analysis_file, readiness_file, *check_files) if p]
    value = build_report(load_json(compile_file), load_json(model_file) if model_file else None,
                         analysis=load_json(analysis_file) if analysis_file else None,
                         readiness=load_json(readiness_file) if readiness_file else None,
                         checks=[load_json(p) for p in check_files])
    value['sources'] = [{'path': str(p), 'sha256': digest(p)} for p in paths]
    output = Path(output).resolve()
    if output.suffix.lower() != '.html' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Use a new .html report path with no existing JSON sidecar')
    html = render_html(value)
    save_json(output.with_suffix('.json'), value)
    output.write_text(html, encoding='utf-8')
    return {'html': str(output), 'json': str(output.with_suffix('.json'))}
