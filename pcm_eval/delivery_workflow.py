"""Link-driven preparation and independently checkpointed atomic evaluations.

Online artifacts identify materials. Only locally installed, hashed requests
choose executables, environments and data adapters.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from .common import digest, json_digest, load_json, save_json, utc_now
from .delivery_source import LarkSource, delivery_url
from .planning import KINDS, build_plan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT/'configs/delivery_catalog.json'


def material_identity(materials):
    """Exclude mutable title/URL/token; include the explicitly linked model versions."""
    return {
        'version': materials['version'],
        'model_assets': sorted([{k: r[k] for k in ('namespace', 'uid', 'version')}
                                for r in materials['model_assets']], key=lambda r: (r['uid'], r['version'])),
        'checkpoints': sorted([{'name': r['name'], 'size': r['size'], 'token': r.get('file_token') or r.get('token')}
                               for r in materials['chains']['model_accuracy']['checkpoints']], key=lambda r: r['name']),
        'onnx': sorted([{'name': r['name'], 'size': r['size'], 'token': r.get('file_token') or r.get('token')} for r in
                        materials['chains']['compile_alignment']['onnx']], key=lambda r: r['name']),
        'packages': sorted([{'name': r['name'], 'size': r['size'], 'token': r.get('file_token') or r.get('token')} for r in
                            materials['chains']['compile_alignment']['packages']], key=lambda r: r['name']),
    }


def _checked_file(root, record):
    path = Path(record['path'])
    path = path.resolve() if path.is_absolute() else (root/path).resolve()
    if not path.is_file() or digest(path) != record.get('sha256'):
        raise ValueError('已核验的执行材料缺失或内容变化：'+path.name)
    return path


def dataset_inventory(dataset, root):
    path = _checked_file(root, dataset['index'])
    index = load_json(path)
    frames = {}
    for head, split in [('occ', 'eval'), ('vision', 'vision_eval'), ('trajectory', 'trajectory_eval')]:
        scenes = index.get(split, {})
        if not isinstance(scenes, dict):
            raise ValueError('验证集索引结构无效：'+split)
        rows = []
        for scene, info in scenes.items():
            if (not isinstance(scene, str) or not scene or Path(scene).is_absolute()
                    or '..' in scene.replace('\\', '/').split('/') or '\\' in scene
                    or not isinstance(info, dict) or not isinstance(info.get('frames'), list)):
                raise ValueError('验证集场景标识或帧清单无效')
            for frame in info['frames']:
                fr = frame.get('frame_index')
                if type(fr) is not int or fr < 0:
                    raise ValueError('验证集帧标识无效')
                rows.append([scene, fr])
        if len(rows) != len(set(map(tuple, rows))):
            raise ValueError('验证集包含重复帧：'+split)
        frames[head] = rows
    return {'id': dataset['id'], 'version': dataset['version'], 'index_sha256': digest(path),
            'frames_by_head': frames, 'locations': deepcopy(dataset.get('locations', {})),
            'gt_heads': list(dataset.get('gt_heads', []))}


def prepare_delivery(model_url, dataset_ref=None, *, catalog_file=None, source=None, run_id=None,
                     validation=False, dataset_source=None):
    catalog_path = Path(catalog_file or DEFAULT_CATALOG).resolve()
    root = Path(load_json(catalog_path).get('material_root', ROOT)).resolve()
    catalog = load_json(catalog_path)
    materials = (source or LarkSource()).read(delivery_url(model_url))
    identity = material_identity(materials)
    if dataset_ref is None:
        refs = list(dict.fromkeys(r['url'] for r in materials['dataset_assets']))
        if len(refs) == 1:
            dataset_ref = refs[0]
    if dataset_ref is not None and (not isinstance(dataset_ref, str) or len(dataset_ref) > 1000
                                   or any(c in dataset_ref for c in '\r\n\x00')):
        raise ValueError('评测数据链接/路径无效')
    datasets = [d for d in catalog.get('datasets', []) if dataset_ref in d.get('aliases', [])]
    dataset, data_error = None, None
    if len(datasets) == 1:
        try:
            dataset = dataset_inventory(datasets[0], root)
        except (OSError, ValueError, KeyError) as e:
            data_error = str(e)
    elif not dataset_ref:
        data_error = ('发现多个数据集版本候选，请明确本次使用的数据集链接' if len(refs)>1
                      else '交付记录没有绑定评测数据，请提供一份评测数据链接')
    else:
        data_error = '本次数据来源尚未完成索引、GT和运行位置核验；不会替换为历史数据'
    if dataset_source is not None:
        # A live in-space probe identifies the submitted mount, not a cache alias.
        receipt = dataset_source.get('mount_receipt', {})
        frames = dataset_source.get('frames_by_head', {})
        from .space_preparation_queue import signature as frame_signature
        if (dataset_source.get('source_ref') != dataset_ref
                or receipt.get('state') != 'MOUNT_VERIFIED'
                or receipt.get('execution_location') != 'lpai'
                or receipt.get('workstation_data_used') is not False
                or receipt.get('asset_binding_sha256') != dataset_source.get('asset_binding_sha256')
                or receipt.get('index_sha256') != dataset_source.get('index_sha256')
                or receipt.get('frame_manifest_sha256') != frame_signature(frames)):
            raise ValueError('本次数据与开发空间挂载、索引及冻结帧清单不一致')
        verified = [d for d in catalog.get('datasets', [])
                    if d.get('index', {}).get('sha256') == dataset_source['index_sha256']
                    and d.get('version') == dataset_source['version']]
        bound = verified[0] if len(verified) == 1 else {}
        dataset = {'id': bound.get('id', dataset_source['id']), 'version': dataset_source['version'],
                   'index_sha256': dataset_source['index_sha256'], 'frames_by_head': frames,
                   'gt_heads': bound.get('gt_heads', []),
                   'locations': {'lpai': receipt['mount_path']}, 'mount_receipt': receipt}
        data_error = None
    run_id = run_id or 'rm-'+__import__('uuid').uuid4().hex[:20]
    if not re.fullmatch(r'rm-[A-Za-z0-9_-]{1,28}', run_id):
        raise ValueError('运行ID应为rm-开头的安全标识')
    jobs = []
    for kind in KINDS:
        keys = ('version', 'model_assets', 'checkpoints') if kind == 'model_accuracy' else ('version', 'onnx', 'packages')
        profiles = [p for p in catalog.get('models', [])
                    if all(p.get('materials', {}).get(k) == identity.get(k) for k in keys)]
        profile = profiles[0] if len(profiles) == 1 else None
        issues = list(materials['issues'])+list(materials['chains'][kind]['issues'])
        if not profile:
            issues.append('已识别交付材料，尚需核验本架构的权重/代码/接口执行适配；不会套用旧版本')
            if kind=='model_accuracy' and len(materials['model_assets'])>1:
                issues.append('模型链接有多个候选，现有版本/架构绑定不能唯一确认：'+'；'.join(r['url'] for r in materials['model_assets']))
        if data_error:
            issues.append(data_error)
        spec = deepcopy((profile or {}).get('chains', {}).get(kind, {}))
        job = {'kind': kind, 'version': materials['version'], 'issues': issues,
               'environment': 'lpai_devspace' if kind == 'model_accuracy' else 'workstation_and_bench',
               'request': None, 'plan': None, 'dataset_id': (dataset or {}).get('id')}
        if dataset:
            required = spec.get('validation_heads' if validation else 'heads', ['occ'] if kind == 'model_accuracy' else [])
            job['heads'] = {h: dataset['frames_by_head'].get(h, []) for h in required}
            for head in required:
                if not job['heads'][head] or head not in dataset['gt_heads']:
                    issues.append('本份数据缺少'+head+'验证帧或已核验GT，不自动拼入其他数据')
        if spec.get('dataset_id') != (dataset or {}).get('id') and profile and dataset:
            issues.append('本次数据尚未绑定该执行适配，需核验预处理、GT与实际运行位置')
        if not spec.get('request'):
            issues.append('缺少已核验的原子执行请求')
        if spec.get('request'):
            try:
                request = load_json(_checked_file(root, spec['request']))
                if request.get('kind') != kind or request.get('version_name') != materials['version']:
                    raise ValueError('执行请求与本次交付类型或版本不一致')
                if request.get('dataset', {}).get('data_link_sha256') != (dataset or {}).get('index_sha256'):
                    # An explicitly pinned subset may be used for validation only.
                    if not (validation and spec.get('validation_only')):
                        issues.append('执行索引与本次完整数据索引不一致，不能把缓存小样本当作全量')
                if spec.get('validation_only') and not validation:
                    issues.append('现有适配仅验收小样本，完整验证集执行适配尚未验收')
                if not spec.get('validation_only') and dataset:
                    actual = request['evaluation'].get('selected_frames') or request['evaluation'].get('selection', {}).get('frames')
                    expected = dataset['frames_by_head'].get(spec.get('scope_head', 'occ'), [])
                    if actual != expected:
                        issues.append('执行请求没有冻结本次完整验证帧清单')
                for key, role in [('binding', 'binding_file'), ('gt_preparation', 'gt_preparation_file')]:
                    if spec.get(key):
                        job[role] = str(_checked_file(root, spec[key]))
                job.update(request=request, plan=build_plan(request, run_id+('-model' if kind == 'model_accuracy' else '-compile')))
                if kind == 'model_accuracy' and dataset_source is not None:
                    if request['dataset'].get('path') != receipt['mount_path']:
                        issues.append('模型执行请求尚未绑定本次LPAI挂载，需完成空间内预处理与GT适配')
                issues.extend('缺少执行字段：'+f for f in job['plan']['missing_fields'])
                issues.extend(x['message'] for x in job['plan']['issues'] if x.get('severity') == 'error')
            except (OSError, ValueError, KeyError) as e:
                issues.append(str(e))
        job['issues'] = list(dict.fromkeys(issues))
        job['ready'] = not job['issues']
        jobs.append(job)
    result = {'schema': 'roadmodel.delivery-workflow/v1', 'run_id': run_id, 'created_at': utc_now(),
              'inputs': {'model_delivery_url': materials['source_url'], 'dataset_ref': dataset_ref},
              'materials': materials, 'dataset': dataset, 'catalog_sha256': digest(catalog_path),
              'scope': 'validation' if validation else 'full', 'jobs': jobs,
              'ready': all(j['ready'] for j in jobs), 'execution_started': False}
    result['workflow_sha256'] = json_digest(result)
    return result


def _write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    temp.replace(path)


def run_delivery(workflow, output, *, executor=None, reporter=None, publisher=None, blocked_jobs=None):
    """Sequential by default; stage recovery never replays an ambiguous inference.

Reporter/publisher are separate stages. A failed report can be retried using
    the immutable successful execution receipt without another model forward.
"""
    from .runner import execute
    root = Path(output).resolve()
    signature = workflow.get('workflow_sha256')
    if (workflow.get('schema') != 'roadmodel.delivery-workflow/v1'
            or signature != json_digest({k: v for k, v in workflow.items() if k != 'workflow_sha256'})
            or {j['kind'] for j in workflow['jobs']} != set(KINDS) or len(workflow['jobs']) != 2):
        raise ValueError('工作流身份或两条链路不完整')
    root.mkdir(parents=True, exist_ok=True)
    frozen = root/'workflow.json'
    if frozen.exists():
        if load_json(frozen) != workflow:
            raise ValueError('已有运行目录绑定其他输入，不覆盖或重新执行')
    else:
        save_json(frozen, workflow)
    # Exclusive lock covers all mutations; an abandoned lock needs reconciliation.
    lock = root/'workflow.lock'
    try:
        with lock.open('x') as f:
            f.write(str(__import__('os').getpid()))
    except FileExistsError as e:
        raise ValueError('原运行正在执行或曾中断，先核验原收据，不重复派发') from e
    results = []
    try:
        for job in workflow['jobs']:
            target = root/job['kind']
            target.mkdir(exist_ok=True)
            path = target/'chain.json'
            prior = load_json(path) if path.exists() else None
            if prior and prior['state'] == 'SUCCEEDED':
                results.append(prior)
                continue
            if job['kind'] in (blocked_jobs or {}):
                detail = {'kind': job['kind'], 'state': 'BLOCKED', 'issues': blocked_jobs[job['kind']]}
                _write_atomic(path, detail)
                results.append(detail)
                continue
            if not job['ready']:
                detail = {'kind': job['kind'], 'state': 'BLOCKED', 'issues': job['issues']}
                _write_atomic(path, detail)
                results.append(detail)
                continue
            stage = target/'inference.json'
            if stage.exists():
                inference = load_json(stage)
                if inference.get('plan_sha256') != job['plan']['plan_sha256']:
                    raise ValueError('原执行收据与本次计划不一致')
                if inference['state'] != 'SUCCEEDED':
                    detail = {'kind': job['kind'], 'state': 'NEEDS_REVIEW',
                              'issues': ['原推理收据尚未核验成功，不重复推理'], 'inference': inference}
                    _write_atomic(path, detail)
                    results.append(detail)
                    continue
                native = inference['receipt']
            else:
                save_json(stage, {'state': 'DISPATCHING', 'plan_sha256': job['plan']['plan_sha256']})
                try:
                    native = (executor or execute)(job['plan'], target/'execution')
                    if native.get('state') != 'succeeded' or not native.get('result_file'):
                        raise ValueError('推理或结果收集没有完成，需核验原执行收据')
                    result = load_json(native['result_file'])
                    if (result.get('evaluation_type') != job['kind'] or result.get('version_name') != job['version']
                            or result.get('execution', {}).get('state') != 'succeeded'
                            or result.get('collection', {}).get('status') != 'complete'):
                        raise ValueError('评测结果类型、版本、执行状态或收集完整性不匹配')
                    native = deepcopy(native)
                    native['result_sha256'] = digest(Path(native['result_file']))
                    _write_atomic(stage, {'state': 'SUCCEEDED', 'plan_sha256': job['plan']['plan_sha256'], 'receipt': native})
                except Exception as e:
                    detail = {'kind': job['kind'], 'state': 'NEEDS_REVIEW', 'issues': [str(e)[:300]]}
                    _write_atomic(path, detail)
                    results.append(detail)
                    continue
            detail = {'kind': job['kind'], 'state': 'REPORT_PENDING', 'execution': native, 'issues': []}
            try:
                if digest(Path(native['result_file'])) != native['result_sha256']:
                    raise ValueError('留存结果已变化，停止生成报告')
                report_path = target/'report-receipt.json'
                if report_path.exists():
                    report = load_json(report_path)
                else:
                    attempt = target/('report-'+__import__('uuid').uuid4().hex[:10])
                    report = (reporter or generate_report)(job, native, attempt)
                    save_json(report_path, report)
                detail.update(report=report, state='PARTIAL' if native.get('incomplete_heads') else 'SUCCEEDED')
                if native.get('incomplete_heads'):
                    detail['issues'] = native['incomplete_heads']
                if publisher:
                    publication = target/'publication-receipt.json'
                    if publication.exists():
                        pub = load_json(publication)
                    else:
                        journal = target/'publication-started.json'
                        if journal.exists():
                            raise ValueError('原发布结果待核验，不重复写入在线报告')
                        save_json(journal, {'state': 'WRITING', 'workflow_sha256': signature})
                        pub = publisher(job, report, target/'publication')
                        if pub.get('state') != 'VERIFIED' or not pub.get('report_url'):
                            raise ValueError('报告发布尚未通过远端回读')
                        save_json(publication, pub)
                    detail['publication'] = pub
            except Exception as e:
                detail.update(state='NEEDS_REVIEW' if (target/'publication-started.json').exists() else 'PARTIAL',
                              issues=[str(e)[:300]])
            _write_atomic(path, detail)
            results.append(detail)
        states = [r['state'] for r in results]
        final = ('SUCCEEDED' if all(s == 'SUCCEEDED' for s in states) else
                 'NEEDS_REVIEW' if 'NEEDS_REVIEW' in states else
                 'PARTIAL' if any(s in ('SUCCEEDED', 'PARTIAL') for s in states) else 'BLOCKED')
        summary = {'schema': 'roadmodel.delivery-execution/v1', 'state': final,
                   'workflow_sha256': signature, 'chains': results, 'updated_at': utc_now()}
        _write_atomic(root/'progress.json', summary)
        return summary
    finally:
        lock.unlink()


def generate_report(job, native, target):
    target.mkdir(parents=True)
    if job['kind'] == 'compile_alignment':
        from .compile_report import write_compile_report
        save_json(target/'request.json', job['request'])
        value = write_compile_report(target/'report.xml', result_file=native['result_file'],
                                     request_file=target/'request.json')
        _portable_siblings(Path(value['xml']))
        return value
    from .atomic_result import prepare_model_result
    from .model_report import write_model_report
    if job.get('native_adapter') == 'lpai_mounted_dataset':
        value = write_model_report(target/'document', result_file=native['result_file'],
                                   head_result_files=native.get('head_result_files'),
                                   head_statuses=native.get('head_statuses'))
        _portable_siblings(Path(value['report_xml']))
        return value
    result = prepare_model_result(job, native, target/'normalized-result.json')
    value = write_model_report(target/'document', result_file=result, binding_file=job.get('binding_file'),
                               gt_preparation_file=job.get('gt_preparation_file'))
    _portable_siblings(Path(value['report_xml']))
    return value


def _portable_siblings(xml):
    import xml.etree.ElementTree as ET
    tree = ET.fromstring('<root>'+xml.read_text(encoding='utf-8')+'</root>')
    for node in tree.iter():
        path = node.get('path', '')
        if path.startswith('@./') and (xml.parent/Path(path).name).is_file():
            node.set('path', '@./'+Path(path).name)
    xml.write_text(''.join(ET.tostring(n, encoding='unicode') for n in tree), encoding='utf-8')
