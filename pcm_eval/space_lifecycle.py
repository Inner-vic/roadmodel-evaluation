"""Task-owned LPAI lifecycle, with explicit UI readback and durable intents."""
from copy import deepcopy
import time

from .common import json_digest
from .dataset_mount import mount_plan, resolve_dataset_input


def template_identity(template):
    required = ('namespace', 'cluster', 'image', 'resources', 'accesses', 'idle_shutdown')
    if any(not template.get(k) for k in required):
        raise ValueError('空间环境模板缺少集群、镜像、资源、现有访问凭证或闲时关闭配置')
    if template['idle_shutdown'].get('enabled') is not True:
        raise ValueError('评测空间模板必须明确闲时关闭策略')
    return json_digest(template)


def advance(ref, profile, journal, *, query, observe_ui, request_ui, probe, save, assert_idle):
    """A CLI 404 alone never authorizes creating a duplicate space."""
    journal = deepcopy(journal)
    official = profile.get('lifecycle_provider') == 'official_cli'
    template = profile['environment_template']
    template_sha = template_identity(template)
    def finish(state, **fields):
        journal.update(state=state, template_sha256=template_sha, source_ref=ref, **fields)
        save(journal)
        return journal
    if journal.get('source_ref', ref) != ref or journal.get('template_sha256', template_sha) != template_sha:
        return finish('NEEDS_REVIEW', reason='原空间准备请求的输入或环境模板发生变化')
    try:
        space = query(['devspaces', 'get', '--namespace', profile['namespace'],
                       '--cluster', profile['cluster'], '--name', profile['space_name']])
    except (ValueError, OSError):
        observation = observe_ui(profile)
        if observation is None or time.time()-observation.get('observed_at', 0) > 300:
            request_ui('observe', {'profile': profile, 'dataset_ref': ref, 'template_sha256': template_sha})
            return finish('WAITING_UI', stage='等待Codex通过LPAI页面核验空间，CLI查询失败不等于空间不存在')
        expected_source = 'LPAI_OFFICIAL_API' if official else 'LPAI_UI'
        if (observation.get('source') != expected_source or observation.get('space_name') != profile['space_name']
                or observation.get('namespace') != profile['namespace']):
            return finish('NEEDS_REVIEW', reason='空间回读来源或身份不一致')
        space = observation.get('space')
        if space is None and observation.get('exists') is not False:
            return finish('NEEDS_REVIEW', reason='页面未明确确认空间存在状态')
    if space is None:
        action, context = 'create', {'profile': profile, 'template': template, 'dataset_ref': ref}
    else:
        if space['metadata']['name'] != profile['space_name'] or space['spec']['clusterName'] != profile['cluster']:
            return finish('NEEDS_REVIEW', reason='页面空间与本次任务绑定不同')
        if official and space['metadata'].get('uid') != profile['space_uid']:
            return finish('NEEDS_REVIEW', reason='官方接口返回的空间UID与本次绑定不同')
        asset = resolve_dataset_input(ref, query, space)
        plan = mount_plan(asset, space)
        phase = plan['space_phase']
        journal['mount_plan'] = plan
        if phase not in ('ready', 'stopped'):
            if not official:
                request_ui('observe', {'profile': profile, 'dataset_ref': ref, 'waiting_phase': phase})
            return finish('PREPARING', stage='等待空间启动和挂载就绪')
        if phase == 'ready' and plan['state'] == 'MOUNT_CONFIGURED':
            receipt = probe(plan)
            source = receipt.get('dataset_source')
            if (receipt.get('state') != 'MOUNT_VERIFIED' or not source
                    or receipt.get('asset_binding_sha256') != asset['binding_sha256']
                    or source.get('source_ref') != ref):
                return finish('NEEDS_REVIEW', reason='空间内实际挂载与本次数据来源不一致')
            return finish('DATA_READY', stage='本次空间与数据已核验，可以接续评测', dataset_source=source)
        if plan['state'] == 'MOUNT_CONFIGURED':
            action = 'start'
        elif phase == 'ready':
            if assert_idle(space) is not True:
                return finish('PREPARING', stage='等待当前空间的活动任务释放资源')
            action = 'stop'
        else:
            action = 'configure_and_start'
        context = {'profile': profile, 'space': space, 'asset': asset, 'dataset_ref': ref}
    intent = journal.get('intent')
    key = json_digest({'action': action, 'source_ref': ref, 'template': template_sha,
                       'observed_phase': (space or {}).get('status', {}).get('phase')})
    if intent:
        if intent['key'] == key:
            return finish('PREPARING' if official else 'WAITING_UI',
                          stage='等待官方接口确认原空间请求状态，不重复提交' if official else '等待已提交的空间动作回读，保留本工单继续执行')
        # A different observed platform phase reconciles the earlier action.
        journal.setdefault('completed_intents', []).append(journal.pop('intent'))
    journal['intent'] = {'action': action, 'key': key}
    finish('PREPARING' if official else 'WAITING_UI',
           stage=('官方接口准备执行空间动作 ' if official else 'Codex准备执行空间动作 ')+action)
    request_ui(action, {**context, 'intent_key': key, 'template_sha256': template_sha})
    return journal
