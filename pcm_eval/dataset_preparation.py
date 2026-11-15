"""Agent-owned LPAI dataset lifecycle with durable mutation intents.

The Runtime supplies supported platform/UI actions and its durable journal.
No state asks the requester to operate LPAI. Missing capabilities are deployment
errors; they must not turn into a successful evaluation or a cache fallback.
"""
import copy
import hashlib
import json

from .dataset_mount import mount_plan


def _signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def advance(asset, space, journal, *, save, actions):
    """Perform at most one mutation per tick; always reconcile before advancing.

    actions: create(asset), assert_idle(space), stop(space), configure_and_start(space, asset),
start(space), probe(plan). configure_and_start adds a read-only mount using the
official selector, retaining all existing space configuration. Creation uses an
explicit administrator-owned template in the Runtime, never inferred resources.
"""
    journal=copy.deepcopy(journal)
    def done(state,**extra):
        journal.update(state=state,**extra);save(journal);return journal
    if journal.get('asset_hash') not in (None,asset['binding_sha256']):
        return done('NEEDS_REVIEW',reason='本次输入版本身份已变化，停止沿用原准备任务')
    journal['asset_hash']=asset['binding_sha256']
    if space is None:
        if journal.get('intent'):
            return done('NEEDS_REVIEW',reason='空间创建结果待平台查询核验，不重复创建')
        if 'create' not in actions or not actions.get('environment_template_sha256'):
            return done('NEEDS_REVIEW',reason='Runtime未配置已审核的空间创建执行器和环境模板，需修复接入；不要求用户代创建')
        journal['intent']={'action':'create','environment_template_sha256':actions['environment_template_sha256']}
        done('PREPARING',stage='Agent按已审核环境模板创建评测空间')
        try:actions['create'](asset)
        except Exception as error:return done('NEEDS_REVIEW',reason='空间创建结果待回读，不自动重试：'+type(error).__name__)
        return journal
    if journal.get('space_uid') not in (None,space['metadata']['uid']):
        return done('NEEDS_REVIEW',reason='开发空间身份变化，需核验原执行而非另建同名空间重试')
    journal['space_uid']=space['metadata']['uid']
    plan=mount_plan(asset,space);phase=plan['space_phase']
    intent=journal.get('intent')
    if intent:
        action=intent['action']
        if action=='create':
            journal.pop('intent')
        elif action=='stop' and phase=='stopped':
            journal.pop('intent')
        elif action=='configure_and_start' and plan['state']=='MOUNT_CONFIGURED':
            before=intent['before_spec'];after=space['spec']
            if any(before.get(k)!=after.get(k) for k in set(before)|set(after) if k!='dataVols'):
                return done('NEEDS_REVIEW',reason='挂载变更同时改动了其他空间配置')
            if any(v not in after.get('dataVols',[]) for v in before.get('dataVols',[])):
                return done('NEEDS_REVIEW',reason='原有挂载被改变，停止派发评测')
            journal.pop('intent')
        elif action=='start' and phase=='ready':
            journal.pop('intent')
        else:
            # The mutation may already have been accepted remotely. Never issue
            # it again after a transport failure, timeout, or service restart.
            return done('NEEDS_REVIEW' if journal.get('state')=='NEEDS_REVIEW' else 'PREPARING',stage='等待平台动作回读',plan=plan)
    if phase not in ('stopped','ready'):
        return done('PREPARING',stage='等待LPAI空间状态稳定',plan=plan)
    if plan['state']=='MOUNT_CONFIGURED' and phase=='ready':
        try:
            receipt=actions['probe'](plan)
            if (receipt.get('state')!='MOUNT_VERIFIED'
                or receipt.get('asset_binding_sha256')!=asset['binding_sha256']
                or receipt.get('space_uid')!=plan['space_uid']
                or receipt.get('execution_location')!='lpai'
                or receipt.get('workstation_data_used') is not False
                or receipt.get('mount_path')!=plan['mount']['mountPath']):
                raise ValueError('挂载回读收据与本次数据及空间不一致')
            return done('DATA_READY',mount_receipt=receipt,stage='空间内数据挂载及索引已核验')
        except Exception as error:
            return done('NEEDS_REVIEW',reason=str(error)[:250])
    action='start' if plan['state']=='MOUNT_CONFIGURED' else 'stop' if phase=='ready' else 'configure_and_start'
    try:
        # Never stop another task to prepare this one. The trusted scheduler
        # validates ownership/leases; an unknown occupancy is not idle.
        if action in ('stop','configure_and_start'):
            if actions['assert_idle'](space) is not True:
                return done('PREPARING',stage='等待当前空间任务释放资源',plan=plan)
        callback=actions[action]
    except KeyError:
        return done('NEEDS_REVIEW',reason='Agent Runtime缺少空间准备执行器，需修复接入；不会要求用户代操作')
    journal['intent']={'action':action,'before_spec':copy.deepcopy(space['spec']),
                       'before_spec_sha256':_signature(space['spec'])}
    done('PREPARING',stage='Agent执行空间准备：'+action,plan=plan)
    try:
        callback(space,asset) if action=='configure_and_start' else callback(space)
    except Exception as error:
        return done('NEEDS_REVIEW',reason='平台操作结果待回读，不自动重试：'+type(error).__name__)
    return journal
