"""Two business inputs; versioned, bounded internal atomic execution contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from .common import digest, load_json
from .planning import KINDS, build_plan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / 'configs/two_link_catalog.json'
FIELDS = {'model_delivery_url', 'dataset_url'}


def url(value, kind):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 1000:
        raise ValueError('请填写完整链接，长度不超过1000字符')
    value = value.strip()
    parts = urlsplit(value)
    hosts = {'li.feishu.cn'} if kind == 'model' else {'lpai.lixiang.com'}
    if (parts.scheme != 'https' or parts.hostname not in hosts or parts.username or parts.password
            or parts.port not in (None, 443) or any(c in value for c in '\r\n\x00')):
        raise ValueError('模型请使用公司飞书交付链接，数据集请使用LPAI链接')
    return urlunsplit(('https', parts.hostname, parts.path.rstrip('/'), parts.query, ''))


def validate_inputs(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('入口仅接受模型交付链接和数据集链接两个字段')
    return {'model_delivery_url': url(value['model_delivery_url'], 'model'),
            'dataset_url': url(value['dataset_url'], 'dataset')}


def resolve_pair(inputs, task_id, *, catalog_file=None):
    if not isinstance(task_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}',task_id):
        raise ValueError('Runtime task ID must be a bounded safe identifier')
    inputs = validate_inputs(inputs)
    catalog = load_json(catalog_file or DEFAULT_CATALOG)
    models = [x for x in catalog['models'] if url(x['url'], 'model') == inputs['model_delivery_url']]
    datasets = [x for x in catalog['datasets'] if url(x['url'], 'dataset') == inputs['dataset_url']]
    model = models[0] if len(models) == 1 else None
    dataset = datasets[0] if len(datasets) == 1 else None
    jobs = []
    for kind in KINDS:
        issues = []
        if not model: issues.append('该模型链接尚无已核验的权重、配置与执行绑定，需解析交付记录')
        if not dataset: issues.append('该数据集链接尚无已核验的数据版本和GT绑定，需解析LPAI任务')
        spec = deepcopy((model or {}).get('chains', {}).get(kind, {}))
        issues.extend(spec.get('issues', []))
        if model and spec.get('version') != model['version']:
            issues.append('链路产物版本与模型交付版本不一致，禁止替换版本')
        if dataset and spec.get('dataset_id') != dataset['id']:
            issues.append('链路GT/数据版本未绑定本次数据集，禁止沿用其他版本')
        if spec.get('active_execution'):
            from .active_execution import verify
            material_root=Path(catalog.get('material_root',ROOT)).resolve()
            try:
                bound=verify(material_root,spec['active_execution'],kind,(model or {}).get('version'),(dataset or {}).get('id'))
                if bound['model_delivery_url']!=inputs['model_delivery_url'] or bound['dataset_url']!=inputs['dataset_url']:
                    issues.append('在途任务与提交链接不一致')
            except (ValueError,OSError,KeyError):
                issues.append('在途评测绑定或冻结输入变化，需核对原执行；不会重新派发')
            jobs.append({'kind':kind,'ready':not issues,'issues':issues,'plan':None,'request':None,
                         'version':(model or {}).get('version'),'dataset_id':(dataset or {}).get('id'),
                         'material_root':str(material_root),'active_execution':spec['active_execution']})
            continue
        if spec.get('retained_result'):
            # Administrator-only binding; never accepted as a third form field.
            from .retained import verify
            material_root = Path(catalog.get('material_root', ROOT)).resolve()
            try:
                verify(material_root, spec['retained_result'], kind, (model or {}).get('version'), (dataset or {}).get('id'))
            except (ValueError, OSError, KeyError):
                issues.append('已登记结果或证据变化，需重新核验，不能自动重跑')
            jobs.append({'kind':kind,'ready':not issues,'issues':issues,'plan':None,'request':None,
                         'version':(model or {}).get('version'),'dataset_id':(dataset or {}).get('id'),
                         'material_root':str(material_root),'retained_result':spec['retained_result'],
                         'heads':spec.get('heads',[]),'warnings':[],
                         'delivery_record':catalog.get('retained_delivery')})
            continue
        if kind == 'compile_alignment' and spec.get('shared_bench_guard_verified') is not True:
            issues.append('共享台架互斥尚未验收，暂不允许机器人派发编译推理')
        request = None
        plan = None
        if not spec.get('request_file'): issues.append('缺少本版本已验证的原子执行配置')
        if spec.get('request_file'):
            path = (ROOT / spec['request_file']).resolve()
            path.relative_to(ROOT)
            if not path.is_file() or digest(path) != spec.get('request_sha256'):
                issues.append('原子执行配置缺失或内容变化，需重新核验绑定')
            else:
                request = load_json(path)
                if request.get('kind') != kind or request.get('version_name') != (model or {}).get('version'):
                    issues.append('执行配置类型或版本不匹配')
                frames = request.get('evaluation', {}).get('expected_frames')
                limit = request.get('evaluation', {}).get('max_frames')
                if type(limit) is not int or type(frames) is not int or not 1 <= frames <= limit <= 20:
                    issues.append('仅允许已冻结的至多20帧小场景配置')
                if not issues:
                    plan = build_plan(request, task_id + '-' + ('model' if kind == 'model_accuracy' else 'compile'))
                    issues.extend(x['message'] for x in plan['issues'] if x.get('severity') == 'error')
                    issues.extend('缺少执行字段: ' + x for x in plan['missing_fields'])
        jobs.append({'kind': kind, 'ready': not issues, 'issues': issues, 'plan': plan,
                     'request': request, 'heads': spec.get('heads', []),
                     'warnings': spec.get('warnings', []),
                     'version': (model or {}).get('version'), 'dataset_id': (dataset or {}).get('id')})
        for field in ('binding_file','gt_preparation_file'):
            if spec.get(field): jobs[-1][field] = str((ROOT/spec[field]).resolve())
    return {'schema': 'roadmodel.dual-resolution/v1', 'inputs': inputs, 'jobs': jobs,
            'ready': all(x['ready'] for x in jobs), 'scope': 'small',
            'dataset_cache_policy': 'reuse_existing_then_fetch_only_required_files',
            'atomic_skills': ['roadmodel-model-accuracy', 'roadmodel-compile-alignment']}
