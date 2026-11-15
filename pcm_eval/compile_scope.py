"""Parameterize the validated native compiler controller for one frozen scope."""
from copy import deepcopy
from pathlib import Path
import shutil

from .common import digest, json_digest, load_json, save_json
from .lpai_native_model import select_frames

ROOT = Path(__file__).resolve().parents[1]


def verify_native_materials(binding, source):
    current = source['current']
    actual = {r['stage']: r['sha256'] for r in current['stages']}
    expected = {r['stage']: r['sha256'] for r in binding['subnet_archives']}
    if (binding['version'] != 'v3.2.2' or set(actual) != set(range(5))
            or len(binding['subnet_archives']) != 5 or actual != expected
            or current['onnx_sha256'] != binding['source_onnx_sha256']):
        raise ValueError('原生控制器的ONNX或分网包与本次交付不同')


def build_plan(request, dataset, run_id, *, selection=None):
    frames = select_frames(dataset['frames_by_head']['occ'], selection)
    if not frames:
        raise ValueError('编译评测缺少验证帧')
    plan = {'schema': 'roadmodel.compile-scope-plan/v1', 'run_id': run_id, 'request': deepcopy(request),
            'frames': frames, 'expected_frames': len(frames), 'dataset_version': dataset['version'],
            'index_sha256': dataset['index_sha256'], 'scope': 'validation' if selection else 'full'}
    plan['plan_sha256'] = json_digest(plan)
    return plan


def prepare(plan, output, profile):
    import numpy as np
    if plan['plan_sha256'] != json_digest({k: v for k, v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('编译范围计划哈希不一致')
    root = Path(output).resolve()
    proof = root/'scope-preparation.json'
    if proof.exists():
        value = load_json(proof)
        if value['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('编译范围准备目录已绑定其他输入')
        for item in value['code_files']:
            if digest(root/item['path']) != item['sha256']:
                raise ValueError('已准备的编译控制器代码变化')
        if not value.get('scope_files'):
            raise ValueError('旧版编译准备收据缺少冻结范围校验，保留原目录待核验')
        for item in value['scope_files']:
            if digest(root/item['path']) != item['sha256']:
                raise ValueError('已准备的编译范围文件变化')
        (root/'compile/frames').mkdir(exist_ok=True)
        return value
    dataset_root = Path(profile['dataset_root'])
    index = Path(profile['index_path'])
    if digest(index) != plan['index_sha256']:
        raise ValueError('工作站编译数据索引与提交数据不同')
    source = load_json(index)
    wanted = set(map(tuple, plan['frames']))
    subset = {key: {} for key in source}
    actual = []
    for scene, info in source['eval'].items():
        frames = [f for f in info['frames'] if (scene, f['frame_index']) in wanted]
        if frames:
            subset['eval'][scene] = dict(info, frames=frames)
            actual.extend([[scene, f['frame_index']] for f in frames])
    if actual != plan['frames']:
        raise ValueError('编译帧范围或顺序与本次索引不同')
    subset['joint_eval'] = subset['eval']
    root.mkdir(parents=True, exist_ok=False)
    selected_index = save_json(root/'selected-data-link.json', subset)
    material_binding = save_json(root/'native-materials.json', {
        'version': plan['request']['version_name'],
        'source_onnx_sha256': plan['request']['model']['onnx_sha256'],
        'subnet_archives': [{'stage': r['stage'], 'sha256': r['sha256']}
                            for r in plan['request']['model']['subnet_archives']]})
    frozen = {'task_id': plan['run_id'], 'model_version': plan['request']['version_name'], 'frames': actual,
              'expected_frames': len(actual), 'data_link_sha256': digest(selected_index),
              'source_data_link_sha256': plan['index_sha256'], 'dataset_version': plan['dataset_version'],
              'native_materials_sha256': digest(material_binding)}
    request_file = save_json(root/'request.json', frozen)
    save_json(root/'data-state.json', {'state': 'READY', 'source': str(dataset_root), 'selected_frames': actual})
    code = root/'code'
    code.mkdir()
    legacy = Path(profile['native_task_code'])
    for name in ('formal_subnet_chain.py', 'subnet_contract.py', 'gt_adapter.py'):
        shutil.copy2(legacy/name, code/name)
    for name in ('formal_compile_remote.py', 'fast_compile_controller.py', 'fast_subnet_driver.py',
                 'batch_subnet_driver.py', 'board_batch_subnet.py', 'board_validate_subnet.py'):
        shutil.copy2(ROOT/'tools'/name, code/name)
    original_gt = (legacy/'formal_compile_remote.py').read_text()
    selected_gt = (code/'formal_compile_remote.py').read_text()
    marker = "            receipt = load(root / 'execution_receipt.json')"
    end = '            cursor += 1'
    original_body = original_gt[original_gt.index(marker):original_gt.index(end, original_gt.index(marker))]
    selected_body = selected_gt[selected_gt.index(marker):selected_gt.index(end, selected_gt.index(marker))]
    if original_body != selected_body:
        raise ValueError('原生GT计算代码发生变化，不能继续适配范围')
    shutil.copytree(ROOT/'pcm_eval', code/'pcm_eval', ignore=shutil.ignore_patterns('__pycache__'))
    settings = load_json(Path(profile['native_release'])/'gt-runtime.json')
    settings['frozen_request_sha256'] = digest(request_file)
    save_json(code/'gt-runtime.json', settings)
    # Serial reference inference uses the same validated ORT session/metric rules.
    # Parallel-reference validation is bound to its old request, so that optional
    # configuration is deliberately absent from this new task.
    compile_root = root/'compile'
    compile_root.mkdir()
    (compile_root/'frames').mkdir()
    from .gt_adapter import Counts
    empty = Counts()
    with (compile_root/'checkpoint.npz').open('xb') as handle:
        np.savez(handle, completed=0, **{b+'_confusion': empty.confusion for b in ('board', 'onnx')},
                 **{b+'_ds': empty.ds for b in ('board', 'onnx')})
    environment = {'ROADMODEL_COMPILE_TASK_ROOT': str(root), 'ROADMODEL_COMPILE_DATASET_ROOT': str(dataset_root),
                   'ROADMODEL_COMPILE_DATA_LINK': str(selected_index)}
    files = [{'path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
             for p in code.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    value = {'state': 'OFFLINE_PREPARED', 'plan_sha256': plan['plan_sha256'], 'frames': actual,
             'expected_frames': len(actual), 'scope': plan['scope'], 'environment': environment,
             'controller': str(code/'fast_compile_controller.py'), 'task_root': str(root),
             'code_files': files, 'bench_contacted': False, 'inference_frames': 0,
             'scope_files': [{'path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
                             for p in (request_file, selected_index, material_binding, root/'data-state.json')],
             'GT_body_source_sha256': digest(legacy/'formal_compile_remote.py'), 'original_GT_body_unchanged': True,
             'model_version': plan['request']['version_name']}
    save_json(proof, value)
    return value


def execute(plan, output, profile, *, bench_available=False):
    from .compile_execution import execute_prepared
    prepared = prepare(plan, Path(output)/'native-scope', profile)
    return execute_prepared(plan, output, prepared, profile, bench_available=bench_available)
