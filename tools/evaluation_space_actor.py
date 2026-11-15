"""Consume one durable UI action using the existing Codex Computer Use runtime."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from pcm_eval.common import load_json
from pcm_eval.dataset_mount import resolve_dataset
from pcm_eval.lpai import request
from pcm_eval.space_browser import run_browser_action
from evaluation_agent import exchange


def execute(cfg, row):
    context = json.loads(row['context'])
    profile = cfg['profile']
    if context['profile'] != profile:
        raise ValueError('空间动作模板与登记配置不同')
    if cfg.get('space_backend') == 'official_cli':
        import re
        from pcm_eval.space_api import OfficialSpaces
        if not re.fullmatch(r'[a-f0-9]{64}', row['id']):
            raise ValueError('Invalid durable space action identity')
        client = OfficialSpaces(profile, cli=cfg['lpai_cli'])
        return client.execute(row['action'], context,
            Path(cfg['local_state'])/'official-space-actions'/row['id'], task=row['task'],
            idle=lambda space: exchange(cfg, 'idle', task=row['task'], plan={
                'space_uid': profile['space_uid'], 'asset': {'source_ref': context['dataset_ref']}})['idle'])
    query = lambda args: request(args, cli=cfg['lpai_cli'])
    asset = context.get('asset') or resolve_dataset(context['dataset_ref'], query)
    space = context.get('space') or {'metadata': {'name': profile['space_name'], 'namespace': profile['namespace'], 'uid': profile['space_uid']},
                                   'spec': {}, 'status': {}}
    space = deepcopy(space)
    space['environment_template'] = profile['environment_template']
    result = run_browser_action(row['action'], space, asset,
        schema_path=ROOT/'schemas/space-preparation-result.schema.json', cwd=ROOT,
        binary=cfg.get('codex_binary'), timeout=180)
    reply = result.get('reply', {})
    if result['state'] == 'OBSERVED' and 'space_exists' in reply and 'mounts' in reply:
        if type(reply['space_exists']) is not bool:
            result.update(state='NEEDS_REVIEW', reason='页面没有明确确认空间存在状态')
            return result
        if not reply['space_exists']:
            observed = None
        else:
            phase = {'运行中': 'ready', '未启动': 'stopped', '已停止': 'stopped', '启动中': 'starting',
                     '镜像拉取中': 'starting', '排队中': 'pending', '容器创建中': 'starting', '停止中': 'stopping'}.get(reply.get('visible_state'))
            if phase is None:
                result.update(state='NEEDS_REVIEW', reason='页面运行状态无法明确识别')
                return result
            access = {k: reply.get(k) for k in ('ssh_host', 'ssh_port', 'credential_name')}
            if (access['ssh_host'] != profile['ssh_host'] or type(access['ssh_port']) is not int
                    or not 1 <= access['ssh_port'] <= 65535
                    or access['credential_name'] != profile['environment_template']['accesses']['ssh_credential']):
                result.update(state='NEEDS_REVIEW', reason='页面未返回已审核集群及现有凭证对应的实际SSH入口')
                return result
            # This is a page binding, never a fabricated platform UUID. The
            # actual SSH mount probe must succeed before evaluation is allowed.
            binding = 'lpai-ui:'+profile['namespace']+':'+profile['space_name']
            observed = {'metadata': {'name': profile['space_name'], 'namespace': profile['namespace'], 'uid': binding},
                        'spec': {'clusterName': profile['cluster'], 'dataVols': [
                            {'name': v['name'], 'type': 'dataset', 'version': v['version'], 'mountPath': v['path'],
                             'mode': 'ReadOnly', 'envKey': v['env'], 'envValue': v['path']} for v in reply['mounts']]},
                        'status': {'phase': phase}, 'verification': {'source': 'LPAI_UI', 'identity_kind': 'page_binding'},
                        'runtime_access': {'host': access['ssh_host'], 'port': access['ssh_port'],
                                           'credential_name': access['credential_name']}}
        result['observation'] = {'source': 'LPAI_UI', 'space_name': profile['space_name'], 'namespace': profile['namespace'],
                                 'observed_at': time.time(), 'exists': reply['space_exists'], 'space': observed}
    return result


def main(config):
    cfg = load_json(config)
    row = exchange(cfg, 'claim_space_action')
    if not row:
        return {'state': 'IDLE'}
    try:
        detail = execute(cfg, row)
    except Exception as error:
        detail = {'state': 'NEEDS_REVIEW', 'reason': str(error)[:250], 'real_browser_call_completed': False}
    exchange(cfg, 'complete_space_action', action_id=row['id'], detail=detail)
    return {'action_id': row['id'], 'action': row['action'], **detail}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.config), ensure_ascii=False))
