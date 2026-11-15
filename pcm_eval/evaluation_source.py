"""Live dataset identity and source preparation; platform mutations stay explicit."""
from .common import json_digest
from .dataset_mount import resolve_dataset_input, mount_plan
from .delivery_workflow import prepare_delivery
from copy import deepcopy


def _prepare_one(request, task, *, profile, query, probe, source=None, catalog_file=None,
            space_actions=None, native_profile=None, compile_profile=None, selection=None):
    plan = None
    dataset = None
    issues = []
    if space_actions is not None:
        from .space_lifecycle import advance
        journal = advance(request['dataset_ref'], profile, space_actions.get('journal', {}),
                          query=query, observe_ui=space_actions['observe'], request_ui=space_actions['request'],
                          probe=probe, save=space_actions['save'], assert_idle=space_actions['idle'])
        if journal['state'] != 'DATA_READY':
            return {'state': journal['state'], 'stage': journal.get('stage', '准备开发空间'),
                    'issues': [journal.get('reason', journal.get('stage', '等待开发空间'))]}
        dataset = journal['dataset_source']
        plan = journal['mount_plan']
    else:
        try:
            space = query(['devspaces', 'get', '--namespace', profile['namespace'],
                           '--cluster', profile['cluster'], '--name', profile['space_name']])
            if (space['metadata']['uid'] != profile['space_uid']
                    or space['spec']['clusterName'] != profile['cluster']):
                raise ValueError('开发空间身份与本人工单运行配置不一致')
            asset = resolve_dataset_input(request['dataset_ref'], query, space)
            plan = mount_plan(asset, space)
            if plan['space_phase'] != 'ready':
                issues.append('开发空间尚未启动。按本工单手动启动或由Codex通过Computer Use协助启动，空间就绪后核验本次挂载')
            elif plan['state'] != 'MOUNT_CONFIGURED':
                issues.append('开发空间尚未挂载本次数据版本，需按本工单通过LPAI页面配置只读挂载')
            else:
                mounted = probe(plan)
                dataset = mounted.get('dataset_source')
                if not dataset or mounted.get('space_uid') != profile['space_uid']:
                    raise ValueError('空间内未返回本次数据索引核验清单')
        except (ValueError, OSError, KeyError) as error:
            issues.append('模型开发空间准备待核验：'+str(error)[:250])
    workflow = prepare_delivery(request['model_delivery_url'], request['dataset_ref'], run_id=task,
                                validation=selection is not None, source=source, catalog_file=catalog_file,
                                dataset_source=dataset)
    if native_profile and dataset:
        from .delivery_workflow import material_identity
        from .lpai_native_model import build_plan
        identity = material_identity(workflow['materials'])
        if all(identity[k] == native_profile['materials'][k] for k in ('version', 'model_assets', 'checkpoints')):
            from copy import deepcopy
            execution = deepcopy(native_profile['execution'])
            execution['gpu_index'] = dataset['runtime']['gpu_index']
            execution['space_access'] = dataset['runtime']['execution_space']
            native = build_plan(execution, dataset, task+'-model', selection=selection)
            job = next(j for j in workflow['jobs'] if j['kind'] == 'model_accuracy')
            job.update(native_adapter='lpai_mounted_dataset', plan=native, ready=True, issues=[],
                       request={'kind': 'model_accuracy', 'version_name': workflow['materials']['version'],
                           'evaluation': {'selected_frames': native['heads']['occ'], 'expected_frames': len(native['heads']['occ'])},
                           'dataset': {'path': dataset['mount_receipt']['mount_path'], 'data_link_sha256': dataset['index_sha256']}})
    if compile_profile and dataset:
        from .compile_scope import build_plan as build_compile_plan
        from .delivery_workflow import material_identity
        identity = material_identity(workflow['materials'])
        job = next(j for j in workflow['jobs'] if j['kind'] == 'compile_alignment')
        if (job['request'] and dataset['index_sha256'] in compile_profile['supported_index_sha256']
                and all(identity[k] == compile_profile['materials'][k] for k in ('version', 'onnx', 'packages'))):
            from .lpai_native_model import select_frames
            frames = select_frames(dataset['frames_by_head']['occ'], selection)
            request = job['request']
            request['evaluation'].update(expected_frames=len(frames), max_frames=len(frames),
                                         selection={'frames': frames}, compute_gt_metrics=True)
            scope_plan = build_compile_plan(request, dataset, task+'-compile', selection=selection)
            job.update(native_adapter='compile_full_scope', plan=scope_plan,
                       ready=not workflow['materials']['chains']['compile_alignment']['issues'],
                       issues=list(workflow['materials']['chains']['compile_alignment']['issues']))
    workflow['dataset_source'] = dataset
    workflow['dataset_preparation'] = {'plan': plan, 'state': 'VERIFIED' if dataset else 'WAITING_SPACE', 'issues': issues}
    if issues:
        job = next(j for j in workflow['jobs'] if j['kind'] == 'model_accuracy')
        job['issues'] = list(dict.fromkeys(job['issues']+issues))
        job['ready'] = False
    workflow['ready'] = all(j['ready'] for j in workflow['jobs'])
    workflow.pop('workflow_sha256')
    workflow['workflow_sha256'] = json_digest(workflow)
    return workflow


def prepare(request, task, *, profile, query, probe, source=None, catalog_file=None,
            space_actions=None, native_profile=None, compile_profile=None, selection=None, vision_source=None):
    # Prepare both mounts before producing a plan. A retry must re-probe OCC too,
    # because adding the visual mount may have replaced the container/SSH port.
    vision_ref = request.get('vision_dataset_ref')
    vision = None
    if vision_ref and vision_source is None:
        actions = space_actions['for_dataset'](vision_ref) if space_actions and 'for_dataset' in space_actions else None
        vision_request = {'model_delivery_url': request['model_delivery_url'], 'dataset_ref': vision_ref}
        prepared = _prepare_one(vision_request, task, profile=profile, query=query,
            probe=lambda p: probe({**p, 'evaluation_head': 'vision'}), source=source,
            catalog_file=catalog_file, space_actions=actions)
        if 'jobs' not in prepared:
            return prepared
        vision = prepared.get('dataset_source')
        if not vision:
            return {'state': 'WAITING_SPACE', 'stage': '等待视觉分类数据集挂载核验',
                    'issues': prepared['dataset_preparation']['issues']}
    actions = space_actions['for_dataset'](request['dataset_ref']) if space_actions and 'for_dataset' in space_actions else space_actions
    workflow = _prepare_one(request, task, profile=profile, query=query,
        probe=lambda p: probe({**p, 'evaluation_head': 'occ'}), source=source, catalog_file=catalog_file,
        space_actions=actions, native_profile=None if vision else native_profile,
        compile_profile=compile_profile, selection=selection)
    if 'jobs' not in workflow:
        return workflow
    if vision_ref and vision_source is not None:
        if not workflow.get('dataset_source'):
            return {'state':'WAITING_SPACE','stage':'等待OCC挂载与开发空间就绪','issues':workflow['dataset_preparation']['issues']}
        vision = vision_source(vision_ref, selection, workflow['dataset_source'])
        if vision.get('source_ref') != vision_ref or vision.get('runtime') != workflow['dataset_source']['runtime']:
            raise ValueError('视觉资产准备来源或执行空间与本次输入不同')
    if vision_ref:
        workflow['inputs']['vision_dataset_ref'] = vision_ref
    if vision and native_profile and workflow.get('dataset_source'):
        from .delivery_workflow import material_identity
        from .lpai_native_model import build_plan
        identity = material_identity(workflow['materials'])
        if all(identity[k] == native_profile['materials'][k] for k in ('version','model_assets','checkpoints')):
            occ = workflow['dataset_source']
            execution = deepcopy(native_profile['execution'])
            execution['gpu_index'] = occ['runtime']['gpu_index']
            execution['space_access'] = occ['runtime']['execution_space']
            # Both datasets must be read by the same dedicated execution space.
            if vision['runtime']['execution_space'] != occ['runtime']['execution_space']:
                return {'state': 'WAITING_SPACE', 'stage': '开发空间已变化，重新核验两份挂载', 'issues': []}
            native = build_plan(execution, occ, task+'-model', selection=selection, vision_dataset=vision)
            job = next(j for j in workflow['jobs'] if j['kind']=='model_accuracy')
            job.update(native_adapter='lpai_mounted_dataset', plan=native, ready=True, issues=[],
                request={'kind':'model_accuracy','version_name':workflow['materials']['version'],
                    'evaluation':{'selected_frames':native['heads']['occ'],'expected_frames':len(native['heads']['occ'])},
                    'dataset':{'path':occ['mount_receipt']['mount_path'],'data_link_sha256':occ['index_sha256']},
                    'head_datasets':{'occ':occ,'vision':vision}})
        workflow['head_datasets'] = {'occ':workflow['dataset_source'],'vision':vision}
    elif not vision_ref and workflow.get('dataset_source'):
        # New two-input tickets bind every head to the submitted dataset. Missing
        # visual GT is handled by per-image validation, never a historical source.
        workflow['head_datasets'] = {'occ':workflow['dataset_source'],'vision':workflow['dataset_source']}
    workflow['ready'] = all(j['ready'] for j in workflow['jobs'])
    workflow.pop('workflow_sha256', None)
    workflow['workflow_sha256'] = json_digest(workflow)
    return workflow
