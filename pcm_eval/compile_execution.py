"""Observe one native dispatch and collect it without repeating board inference."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import time
import uuid

from .common import digest, json_digest, load_json, save_json, utc_now
from .validation import require_result


def controller_alive(pid, controller):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(bytes([0]))
    except OSError:
        return False
    return str(controller).encode() in args


def _verify_collection(root, plan, native_root):
    binding = load_json(root/'collection-binding.json')
    result_file = root/'normalized-result.json'
    if (binding['plan_sha256'] != plan['plan_sha256']
            or binding['checkpoint_sha256'] != digest(native_root/'compile/checkpoint.npz')
            or binding['result_sha256'] != digest(result_file)
            or binding['source_manifest_sha256'] != digest(root/'native-source-manifest.json')):
        raise ValueError('编译收集结果或检查点已变化，不重新推理')
    manifest = load_json(root/'native-source-manifest.json')
    if [r['frame'] for r in manifest['frames']] != plan['frames']:
        raise ValueError('编译收集范围与冻结计划不同')
    for row in manifest['frames']:
        for item in row['files']:
            path = Path(item['path']).resolve()
            path.relative_to((native_root/'compile/frames').resolve())
            if digest(path) != item['sha256']:
                raise ValueError('编译原始结果已变化，停止复用')
    result = load_json(result_file)
    require_result(result)
    live = result.get('provenance', {}).get('live_run', {})
    if (result['evaluation_type'] != 'compile_alignment'
            or result['version_name'] != plan['request']['version_name']
            or result['execution']['state'] != 'succeeded'
            or result['collection']['status'] != 'complete'
            or live.get('run_id') != plan['run_id'] or live.get('selected_frames') != plan['frames']):
        raise ValueError('编译结果身份或完整性不匹配')
    return result_file


def collect_completed(plan, output, prepared):
    from .compile_aggregate import aggregate_compile
    root = Path(prepared['task_root'])
    progress = load_json(root/'compile/progress.json')
    if (progress.get('state') != 'SUCCEEDED' or progress.get('task_id') != plan['run_id']
            or progress.get('expected_frames') != len(plan['frames'])
            or progress.get('completed_frames') != len(plan['frames'])
            or (root/'compile/controller.lock').exists()):
        raise ValueError('编译原生任务未完整结束，保留检查点及每帧收据')
    output = Path(output)
    result_root = output/'aggregation'
    if not result_root.exists():
        # A failed collection preserves its own directory; retrying collection
        # alone cannot touch the dispatch marker or any native frame.
        staging = output/('collection-'+uuid.uuid4().hex)
        request = deepcopy(plan['request'])
        request['evaluation'].update(expected_frames=len(plan['frames']), max_frames=len(plan['frames']),
                                     selection={'frames': plan['frames']}, compute_gt_metrics=True)
        aggregate_compile([root/'compile/frames'/f'frame_{i:06d}' for i in range(len(plan['frames']))],
                          plan['frames'], staging, request=request, run_id=plan['run_id'],
                          checkpoint=root/'compile/checkpoint.npz')
        # Aggregation artifacts carry paths, so collect directly into its final
        # attempt directory and bind that directory before publishing a pointer.
        save_json(staging/'collection-binding.json', {'plan_sha256': plan['plan_sha256'],
            'checkpoint_sha256': digest(root/'compile/checkpoint.npz'),
            'source_manifest_sha256': digest(staging/'native-source-manifest.json'),
            'result_sha256': digest(staging/'normalized-result.json')})
        result_file = _verify_collection(staging, plan, root)
        pointer = output/'collection.json'
        if pointer.exists():
            raise ValueError('已有编译收集指针，先核验原结果')
        save_json(pointer, {'plan_sha256': plan['plan_sha256'], 'directory': str(staging.resolve())})
        return result_file
    # Earlier releases without a binding must be reconciled, never overwritten.
    return _verify_collection(result_root, plan, root)


def execute_prepared(plan, output, prepared, profile, *, bench_available=False):
    if plan['plan_sha256'] != json_digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise ValueError('编译计划内容发生变化')
    output = Path(output).resolve()
    root = Path(prepared['task_root']).resolve()
    if prepared['plan_sha256'] != plan['plan_sha256'] or root != output/'native-scope':
        raise ValueError('编译准备收据不属于本次计划或目录')
    receipt_file = output/'execution-receipt.json'
    pointer_file = output/'collection.json'
    if receipt_file.exists() or pointer_file.exists():
        pointer = load_json(pointer_file)
        directory = Path(pointer['directory']).resolve()
        directory.relative_to(output)
        if pointer['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('编译收集指针属于其他计划')
        result_file = _verify_collection(directory, plan, root)
        if receipt_file.exists():
            receipt = load_json(receipt_file)
            if (receipt.get('plan_sha256') != plan['plan_sha256'] or receipt.get('state') != 'succeeded'
                    or receipt['result_file'] != str(result_file)
                    or receipt['result_sha256'] != digest(result_file)):
                raise ValueError('编译执行收据与留存结果不同')
            return receipt
    else:
        marker = root/'dispatch.json'
        child = None
        if marker.exists():
            dispatch = load_json(marker)
            if dispatch.get('plan_sha256') != plan['plan_sha256']:
                raise ValueError('原编译派发属于其他计划')
        else:
            if bench_available is not True:
                raise ValueError('台架未获准使用，仅允许离线准备和已有结果收集')
            if (root/'STOP_AFTER_FRAME').exists():
                raise ValueError('编译任务已暂停，不能隐式恢复')
            if (root/'compile/controller.lock').exists() or (root/'compile/progress.json').exists():
                raise ValueError('已有未绑定的原生控制器状态，不重复启动')
            dispatch = {'state': 'DISPATCHING', 'plan_sha256': plan['plan_sha256'], 'started_at': utc_now()}
            save_json(marker, dispatch)
            with (root/'controller.log').open('xb') as log:
                child = subprocess.Popen([profile['python'], prepared['controller']], cwd=root/'code',
                    env={**os.environ, **prepared['environment']}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            dispatch.update(state='STARTED', pid=child.pid)
            save_json(marker, dispatch, overwrite=True)
        if child is not None:
            while child.poll() is None:
                time.sleep(5)
            if child.returncode:
                raise ValueError('编译控制器退出异常，保留原生收据，不重复推理')
        else:
            while controller_alive(dispatch.get('pid'), prepared['controller']):
                time.sleep(5)
        result_file = collect_completed(plan, output, prepared)
    dispatch = load_json(root/'dispatch.json')
    receipt = {'state': 'succeeded', 'run_id': plan['run_id'], 'plan_sha256': plan['plan_sha256'],
               'started_at': dispatch.get('started_at'), 'finished_at': utc_now(),
               'result_file': str(result_file), 'result_sha256': digest(result_file)}
    save_json(receipt_file, receipt)
    return receipt
