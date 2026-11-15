"""Run frozen OCC/vision manifests entirely inside a mounted LPAI space."""
from copy import deepcopy
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

from .common import digest, json_digest, load_json, save_json, utc_now
from .delivery_workflow import _write_atomic


def select_frames(frames, selection=None):
    rows = [list(row) for row in frames]
    if len(rows) != len(set(map(tuple, rows))):
        raise ValueError('数据索引存在重复帧')
    if not selection:
        return rows
    count, per_scene = selection['scene_count'], selection['frames_per_scene']
    if type(count) is not int or type(per_scene) is not int or min(count, per_scene) < 1:
        raise ValueError('验证范围必须是正整数场景数和每场景帧数')
    scene_offset, frame_offset = selection.get('scene_offset', 0), selection.get('frame_offset', 0)
    if any(type(v) is not int or v < 0 for v in (scene_offset, frame_offset)):
        raise ValueError('验证偏移必须是非负整数')
    scenes = list(dict.fromkeys(s for s, _ in rows))[scene_offset:scene_offset+count]
    return [row for scene in scenes for row in [r for r in rows if r[0] == scene][frame_offset:frame_offset+per_scene]]


def build_plan(profile, dataset, run_id, *, selection=None, vision_dataset=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run_id):
        raise ValueError('Invalid native run identity')
    if dataset['mount_receipt']['state'] != 'MOUNT_VERIFIED':
        raise ValueError('本次数据尚未核验实际只读挂载')
    if dataset.get('vision_gt_binding'):
        from .vision_gt_binding import declared_frames
        bound_frames = declared_frames(dataset,dataset['vision_gt_binding'])
        if dataset['frames_by_head'].get('vision') != bound_frames:
            raise ValueError('视觉帧清单与逐帧GT收据不同')
        if not selection:
            raise ValueError('当前逐帧GT绑定只核验了小场景，不可冒充全量视觉范围')
    heads = {h: select_frames(dataset['frames_by_head'].get(h, []), selection) for h in profile['heads']}
    if vision_dataset is not None:
        source_state = vision_dataset['mount_receipt']['state']
        if source_state not in ('MOUNT_VERIFIED','ASSET_FILES_VERIFIED'):
            raise ValueError('视觉数据尚未核验实际只读挂载')
        if source_state == 'ASSET_FILES_VERIFIED':
            if not selection or vision_dataset.get('asset_selection') != selection:
                raise ValueError('视觉资产准备仅允许已授权的冻结小场景，不允许扩大为全量')
            heads['vision'] = deepcopy(vision_dataset['frames_by_head']['vision'])
        else:
            heads['vision'] = select_frames(vision_dataset['frames_by_head'].get('vision', []), selection)
    if dataset.get('vision_gt_binding'):
        # This receipt already freezes the exact verified small-frame scope;
        # offsets used to choose fresh OCC/board frames must not select it twice.
        scenes = {s for s, _ in bound_frames}
        if len(scenes)>selection['scene_count'] or any(sum(s==scene for s,_ in bound_frames)>selection['frames_per_scene'] for scene in scenes):
            raise ValueError('逐帧GT收据超过本次小场景范围')
        heads['vision'] = bound_frames
    if not heads.get('occ'):
        raise ValueError('本版原生结果适配需要OCC验证帧')
    if profile['source_commit'] != profile['binding']['code']['evaluation_commit']:
        raise ValueError('模型执行代码与交付绑定不一致')
    for name in ('main_checkpoint', 'classifier_checkpoint'):
        if profile[name]['sha256'] != profile['binding'][name]['sha256']:
            raise ValueError('权重与交付绑定不一致')
    batch_size = profile.get('batch_size', 100)
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError('Invalid native batch size')
    plan = {'schema': 'roadmodel.lpai-native-plan/v1', 'run_id': run_id, 'profile': deepcopy(profile),
            'dataset': deepcopy(dataset), 'heads': heads, 'batch_size': batch_size,
            'scope': 'validation' if selection else 'full', 'selection': selection}
    if vision_dataset is not None:
        plan['datasets_by_head'] = {'occ': deepcopy(dataset), 'vision': deepcopy(vision_dataset)}
    if 'vision' in heads:
        # Candidate scope is not scored coverage. The mount-side preparer must
        # still find exact per-image labels before any vision inference.
        plan['vision_candidates'] = heads['vision'] if vision_dataset is not None else heads['vision'] or heads['occ']
    plan['plan_sha256'] = json_digest(plan)
    return plan


def prepare_occ_input(plan, selected, output):
    """Create only adapted GT and a datalink. Read raw inputs at their mounted paths."""
    import numpy as np
    from .model_dataset import native_sparse_gt
    root = Path(plan['dataset']['mount_receipt']['mount_path']).resolve()
    index = Path(plan['dataset']['index_path'])
    index.resolve().relative_to(root)
    if digest(index) != plan['dataset']['index_sha256']:
        raise ValueError('本次挂载索引内容变化')
    original = load_json(index)
    subset = {key: {} for key in original}
    adaptations = []
    for scene, frame_id in selected:
        if Path(scene).is_absolute() or '..' in Path(scene).parts:
            raise ValueError('Invalid scene path')
        scene_info = original['eval'][scene]
        matches = [r for r in scene_info['frames'] if r['frame_index'] == frame_id]
        if len(matches) != 1:
            raise ValueError('Frozen frame is not unique in the submitted dataset')
        frame = dict(matches[0])
        folder = root/scene/'dataset'/str(frame_id)
        source = (folder/frame['occ_file']).resolve()
        source.relative_to(root)
        if not (folder/(str(frame_id)+'_f_ca_120.jpg')).is_file():
            raise ValueError('选中帧缺少本版模型的120度图像')
        source_gt = np.load(source, allow_pickle=False)
        adapted, receipt = native_sparse_gt(source_gt, plan['profile']['binding']['gt_contract'])
        target = Path(output)/'adapted-occ'/scene/(str(frame_id)+'.npy')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not np.array_equal(np.load(target, allow_pickle=False), adapted):
                raise ValueError('已有GT适配结果与本次输入不同')
        else:
            np.save(target, adapted, allow_pickle=False)
        frame['occ_file'] = str(target.resolve())
        frame['height_field_file'] = None
        subset['eval'].setdefault(scene, dict(scene_info, frames=[]))['frames'].append(frame)
        adaptations.append({**receipt, 'scene': scene, 'frame_index': frame_id,
                            'source_sha256': digest(source), 'adapted_sha256': digest(target)})
    subset['joint_eval'] = subset['eval']
    link = Path(output)/'data-link.json'
    if link.exists() and load_json(link) != subset:
        raise ValueError('已有原生索引与本次冻结帧不同')
    if not link.exists():
        save_json(link, subset)
    save_json(Path(output)/'gt-preparation.json', {'state': 'validated', 'frames': adaptations,
        'execution_location': 'lpai', 'raw_data_copied_to_workstation': False},
        overwrite=(Path(output)/'gt-preparation.json').exists())
    return link


def launch_batch(plan, kind, selected, folder):
    profile = plan['profile']
    root = Path(profile['source_root'])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(profile['environment'])
    dataset = plan.get('datasets_by_head', {}).get(kind, plan['dataset'])
    env.update(DATASET_PATH=','.join(dataset.get('data_roots', [dataset['mount_receipt']['mount_path']])),
               DATASET=profile.get('dataset_classes_by_head', {}).get(kind, profile.get('dataset_class', 'SmartChassisDatasetV7')), NUM_WORKERS='0',
               PERSISTENT_WORKERS='False', PIN_MEMORY='False', BATCH_SIZE='1',
               CUDA_VISIBLE_DEVICES=str(profile['gpu_index']), CKPT_SKIP_KEYS='')
    if kind == 'occ':
        env.update(DATA_LINK_FILE=str(prepare_occ_input(plan, selected, folder)), TRAIN_MODE='occ', HEAD_MODE='gs_only')
    elif kind == 'vision':
        from .vision_preparation import prepare
        from .vision_gt_binding import BindingRejected
        allowed = plan.get('vision_candidates', plan['heads'].get('vision', []))
        if not selected or len(selected) != len(set(map(tuple, selected))) or any(row not in allowed for row in selected):
            raise BindingRejected('VISION_SCOPE_DIFFERS_FROM_PLAN', {})
        prepared = prepare(dataset, selected, folder/'vision-input')
        if prepared['state'] != 'READY':
            raise BindingRejected('VISION_GT_BLOCKED', {'issues': prepared['issues']})
        link = prepared['index_path']
        env.update(DATA_LINK_FILE=str(link), TRAIN_MODE='vision', HEAD_MODE='gs_only')
    else:
        raise ValueError('Unsupported native model head')
    launch = {'schema': 'roadmodel.head-launch/v1', 'kind': kind,
              'runtime_root': str(Path(__file__).resolve().parents[1]),
              'run_id': plan['run_id']+'-'+folder.name, 'version_name': profile['version'],
              'source_root': str(root), 'source_commit': profile['source_commit'],
              'main_checkpoint': profile['main_checkpoint']['path'],
              'classifier_checkpoint': profile['classifier_checkpoint']['path'],
              'checkpoint_sha256': {Path(profile[k]['path']).name: profile[k]['sha256']
                                    for k in ('main_checkpoint', 'classifier_checkpoint')},
              'py_config': profile['py_config'], 'environment': env, 'expected_frames': len(selected),
              'selected_frames': selected, 'dataset_version': dataset['version'],
              'dataset_source_ref': dataset.get('source_ref'),
              'dataset_class': env['DATASET'], 'dataset_index_sha256': dataset['index_sha256']}
    if kind == 'vision':
        launch['vision_preparation'] = prepared
    launch['argv'] = [sys.executable, 'eval.py', '--py-config', profile['py_config'], '--work-dir', str(folder/'output'),
                      '--resume-from', launch['main_checkpoint'], '--classifier-resume', launch['classifier_checkpoint'],
                      '--sample-freq', '1', '--zip-vis', 'none']
    return launch


PREFLIGHT = r'''import pathlib,sys,os,json,torch
p=pathlib.Path(sys.argv[1]);a=json.loads((p/'launch.json').read_text());os.environ.update(a['environment']);os.environ['eval']='true';os.chdir(a['source_root']);sys.path.insert(0,a['source_root']);sys.path.insert(0,a['runtime_root'])
import model
from mmengine import Config
from mmseg.models import build_segmentor
from dataset import get_dataloader
from misc.ckpt_load_utils import overlay_export_heads
cfg=Config.fromfile(a['py_config'],lazy_import=False);net=build_segmentor(cfg.model);sd=torch.load(a['main_checkpoint'],map_location='cpu',weights_only=True)['state_dict'];net.load_state_dict(sd,strict=True);overlay_export_heads(net,classifier_resume=a['classifier_checkpoint'])
if a['kind']=='vision':
 from pcm_eval.vision_preparation import verify
 tasks=net.classifier_core_names if net.classifier_core_only else net.classifier_all_names
 if not tasks or net.classifier is None or net.no_classifier:raise ValueError('The bound model has no active classification tasks')
 verify(a['vision_preparation'],required_tasks=tasks)
_,dl=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
assert len(dl.dataset)==a['expected_frames'],(len(dl.dataset),a['expected_frames'])
(p/'model-preflight.json').write_text(json.dumps({'state':'passed','dataset_size':len(dl.dataset),'strict_main_keys':len(sd),'pc_range':list(cfg.pc_range),'torch':torch.__version__,'cuda':torch.version.cuda}))
'''


def run_batch(plan, kind, selected, folder):
    from .parsers import collect_model
    folder = Path(folder)
    normalized = folder/'normalized-result.json'
    if normalized.exists():
        value = load_json(normalized)
        if value['provenance']['native_run']['selected_frames'] != selected:
            raise ValueError('原结果的帧范围与本次计划不同')
        return value
    receipt_file = folder/'execution_receipt.json'
    if receipt_file.exists():
        saved = load_json(receipt_file)
        if saved['state'] != 'succeeded' or saved['selected_frames'] != selected:
            raise ValueError('原生执行结果待核验，不重复推理')
        launch = load_json(folder/'launch.json')
    else:
        launch = launch_batch(plan, kind, selected, folder)
        save_json(folder/'launch.json', launch)
        receipt = {**launch, 'schema': 'roadmodel.model-execution/v1', 'source': 'lpai_devspace_native_eval',
                   'state': 'preflight', 'started_at': utc_now()}
        _write_atomic(receipt_file, receipt)
        env = {**os.environ, **launch['environment']}
        started = time.monotonic()
        try:
            with (folder/'preflight-console.log').open('xb') as log:
                result = subprocess.run([sys.executable, '-c', PREFLIGHT, str(folder)], env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError('模型原生加载或数据预检失败')
            receipt['state'] = 'running'
            _write_atomic(receipt_file, receipt)
            with (folder/'eval-console.log').open('xb') as log:
                result = subprocess.run(launch['argv'], cwd=launch['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError('模型原生评测失败')
            receipt.update(state='succeeded', exit_code=0, duration_seconds=time.monotonic()-started, finished_at=utc_now())
            _write_atomic(receipt_file, receipt)
        except Exception as error:
            receipt.update(state='failed', error=str(error), finished_at=utc_now())
            _write_atomic(receipt_file, receipt)
            raise
    console = (folder/'eval-console.log').read_text(errors='replace')
    count = len(selected)
    if not re.search(r'100%[^\r\n]*\b'+str(count)+'/'+str(count)+r'\b', console):
        raise ValueError('原生日志未确认冻结帧全部完成')
    save_json(folder/'coverage_receipt.json', {'state': 'verified', 'run_id': launch['run_id'],
        'selected_frames': selected, 'observed_frames': count, 'expected_frames': count,
        'console_sha256': digest(folder/'eval-console.log'), 'loader_preflight_sha256': digest(folder/'model-preflight.json')},
        overwrite=(folder/'coverage_receipt.json').exists())
    logs = list((folder/'output').glob('*/*.log'))
    if len(logs) != 1:
        raise ValueError('原生评测日志不唯一')
    result = collect_model(folder, plan['profile']['version'], expected_frames=count, log_file=logs[0])
    if result['collection']['status'] != 'complete':
        raise ValueError('原生评测结果解析不完整')
    save_json(normalized, result)
    return result


def run(plan, output):
    from .model_aggregate import aggregate_occ
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if plan['plan_sha256'] != json_digest({k: v for k, v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('Native plan hash differs')
    lock = root/'execution.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        frozen = root/'plan.json'
        if frozen.exists() and load_json(frozen) != plan:
            raise ValueError('原运行目录已绑定其他输入')
        if not frozen.exists():
            save_json(frozen, plan)
        profile = plan['profile']
        for name in ('main_checkpoint', 'classifier_checkpoint'):
            if digest(Path(profile[name]['path'])) != profile[name]['sha256']:
                raise ValueError('交付权重哈希不一致')
        source = load_json(profile['source_manifest'])
        if source['commit'] != profile['source_commit']:
            raise ValueError('Source snapshot commit differs')
        for name, expected in source['files'].items():
            path = (Path(profile['source_root'])/name).resolve()
            path.relative_to(Path(profile['source_root']).resolve())
            if digest(path) != expected:
                raise ValueError('模型源文件发生变化：'+name)
        outputs, incomplete, statuses, executed = {}, [], {}, {}
        for kind, frames in plan['heads'].items():
            if kind == 'vision' and plan.get('vision_candidates'):
                from .vision_preparation import prepare
                from .vision_gt_binding import BindingRejected
                candidates = plan['vision_candidates']
                try:
                    preparation = prepare(plan.get('datasets_by_head', {}).get(kind, plan['dataset']), candidates, root/'vision-preparation')
                    statuses[kind] = {'state': preparation['state'], 'requested_frames': candidates,
                                      'preparation': str(root/'vision-preparation/preparation.json'),
                                      'issues': preparation['issues'], 'completed_frames': 0}
                    if preparation['state'] != 'READY':
                        codes = sorted({r['code'] for r in preparation['issues']})
                        incomplete.append('vision：本次索引 vision_eval 与选定图像的帧级GT绑定未就绪：'+','.join(codes))
                        continue
                    frames = candidates
                except (BindingRejected, OSError, ValueError) as error:
                    code = error.code if isinstance(error, BindingRejected) else 'VISION_PREPARATION_FAILED'
                    statuses[kind] = {'state': 'BLOCKED', 'code': code, 'completed_frames': 0,
                                      'requested_frames': candidates}
                    incomplete.append('vision：帧级GT绑定未通过：'+code)
                    continue
            if not frames:
                # Empty indexed coverage is not proof that annotations do not
                # exist elsewhere (for example scene-level tags.json).
                if kind == 'vision':
                    incomplete.append('vision：本次索引 vision_eval 未列出验证帧；尚未建立可用于计分的帧级GT绑定')
                else:
                    incomplete.append(kind+'未列出本次验证帧；GT可用性待核验')
                statuses[kind] = {'state': 'BLOCKED', 'completed_frames': 0, 'requested_frames': []}
                continue
            batch_size = plan['batch_size'] if kind == 'occ' else len(frames)
            paths = []
            for index, start in enumerate(range(0, len(frames), batch_size)):
                folder = root/kind/f'batch-{index:04d}'
                try:
                    run_batch(plan, kind, frames[start:start+batch_size], folder)
                except Exception:
                    if kind != 'vision':
                        raise
                    # An unsuccessful vision attempt is retained for review;
                    # it must not discard already verified OCC results.
                    statuses[kind] = {**statuses.get(kind, {}), 'state': 'NEEDS_REVIEW',
                                      'code': 'VISION_EXECUTION_NEEDS_REVIEW',
                                      'completed_frames': 0, 'execution_receipt': str(folder/'execution_receipt.json')}
                    incomplete.append('vision：原生加载、推理或收集待核验；保留原执行收据，不自动重跑')
                    break
                paths.append(folder/'normalized-result.json')
                _write_atomic(root/'progress.json', {'state': 'RUNNING', 'head': kind,
                    'completed_frames': min(start+batch_size, len(frames)), 'expected_frames': len(frames)})
            if statuses.get(kind, {}).get('state') == 'NEEDS_REVIEW':
                continue
            if kind == 'occ':
                merged = root/'occ-aggregate'
                if not (merged/'normalized-result.json').exists():
                    aggregate_occ(paths, frames, merged, run_id=plan['run_id'])
                outputs[kind] = str(merged/'normalized-result.json')
            else:
                outputs[kind] = str(paths[0])
            executed[kind] = frames
            statuses[kind] = {**statuses.get(kind, {}), 'state': 'SUCCEEDED',
                              'completed_frames': len(frames), 'selected_frames': frames}
        receipt = {'state': 'succeeded', 'result_file': outputs['occ'],
                   'head_result_files': [p for k, p in outputs.items() if k != 'occ'],
                   'incomplete_heads': incomplete, 'plan_sha256': plan['plan_sha256'],
                   'head_statuses': statuses, 'executed_frames_by_head': executed,
                   'execution_location': 'lpai', 'workstation_data_used': False}
        receipt['head_datasets'] = {kind: {'version': dataset.get('version'), 'source_ref': dataset.get('source_ref'),
            'index_sha256': dataset.get('index_sha256'), 'selected_frames': plan['heads'].get(kind, [])}
            for kind, dataset in plan.get('datasets_by_head', {'occ': plan['dataset'], 'vision': plan['dataset']}).items()}
        _write_atomic(root/'result.json', receipt)
        _write_atomic(root/'progress.json', {**receipt, 'state': 'PARTIAL' if incomplete else 'SUCCEEDED'})
        return receipt
    finally:
        lock.unlink()
