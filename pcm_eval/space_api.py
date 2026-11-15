"""Official LPAI devspace adapter, pinned to an existing space UID.

The installed official lpai-agent exposes this API. It is not the separate lpai
training Python package. No browser tokens, shell commands or raw HTTP are used.
"""
from copy import deepcopy
from pathlib import Path
import re
import time

from .common import json_digest, load_json, save_json, utc_now
from .dataset_mount import mount_plan, resolve_dataset
from .lpai import request


PROVIDER = 'LPAI_OFFICIAL_API'


class OfficialSpaces:
    def __init__(self, profile, *, cli='lpai-agent', invoke=None):
        self.profile = deepcopy(profile)
        self.cli = cli
        self.invoke = invoke or (lambda args: request(args, cli=cli, preserve_envelope=True))

    def args(self, operation):
        if operation not in ('get', 'start', 'stop'):
            raise ValueError('Unsupported official devspace operation')
        return ['devspaces', operation, '--namespace', self.profile['namespace'],
                '--cluster', self.profile['cluster'], '--name', self.profile['space_name']]

    def read(self):
        value = self.invoke(self.args('get'))
        space = deepcopy(value.get('data', value))
        meta, spec = space.get('metadata', {}), space.get('spec', {})
        profile = self.profile
        if profile.get('lpai_account') and meta.get('annotations',{}).get('lpai.org/owner','').lower()!=profile['lpai_account'].lower():
            raise ValueError('Official space owner differs from the initialized LPAI account')
        uid = profile['space_uid']
        if (not uid or uid.startswith('lpai-ui:') or meta.get('uid') != uid
                or meta.get('name') != profile['space_name'] or meta.get('namespace') != profile['namespace']
                or spec.get('clusterName') != profile['cluster']):
            raise ValueError('Official space identity differs from the pinned UID/namespace/cluster')
        phase = str(space.get('status', {}).get('phase', '')).lower()
        phase = {'running': 'ready', 'image_pulling': 'starting', 'waiting': 'starting'}.get(phase, phase)
        if phase not in ('ready', 'stopped', 'pending', 'starting', 'stopping', 'failed'):
            raise ValueError('Official space phase is unknown')
        space['status']['phase'] = phase
        credential = profile['environment_template']['accesses']['ssh_credential']
        access = [x for x in spec.get('accesses', []) if x.get('name') == 'ssh']
        if len(access) != 1 or spec.get('sshSecretName') != credential or access[0].get('pubkeyName') != credential:
            raise ValueError('Official SSH binding differs from the existing approved credential')
        address = access[0].get('address', '')
        host, port = None, None
        match = re.fullmatch(r'(?:ssh://)?(?:root@)?([A-Za-z0-9.-]+):(\d+)/?', address)
        if match:
            host, port = match[1], int(match[2])
        else:
            host_match = re.search(r'\broot@([A-Za-z0-9.-]+)\b', address)
            port_match = re.search(r'(?:^|\s)-p\s+(\d+)\b', address)
            if host_match and port_match:
                host, port = host_match[1], int(port_match[1])
        if host != profile['ssh_host'] or port is None or not 1 <= port <= 65535:
            raise ValueError('Official API did not return the approved SSH host and a valid port')
        # Retain only lifecycle/mount identity, never unrelated access URLs or
        # platform configuration in work-order observations.
        space = {'metadata': {k: meta[k] for k in ('name', 'uid', 'namespace')},
                 'spec': {'clusterName': spec['clusterName'], 'dataVols': [
                     {k:v.get(k) for k in ('name','type','version','mountPath','pvcName','mode','envKey','envValue','fsType')}
                     for v in spec.get('dataVols', [])]}, 'status': {'phase': phase}}
        space['verification'] = {'source': PROVIDER, 'identity_kind': 'uid_binding'}
        space['runtime_access'] = {'host': host, 'port': port, 'credential_name': credential}
        return space

    def observation(self, space=None):
        space = self.read() if space is None else space
        return {'source': PROVIDER, 'space_name': self.profile['space_name'],
                'namespace': self.profile['namespace'], 'observed_at': time.time(), 'exists': True,
                'space': space}

    def execute(self, action, context, output, *, task, idle=None):
        if action not in ('observe', 'start', 'stop'):
            return {'state': 'NEEDS_ASSISTANCE', 'provider': PROVIDER,
                    'reason': '本次需要创建空间或修改挂载，需Codex协助完成；尚未发送启动请求'}
        if context['profile'] != self.profile:
            raise ValueError('Space action profile changed')
        space = self.read()
        if action == 'observe':
            return {'state': 'OBSERVED', 'provider': PROVIDER, 'real_api_call_completed': True,
                    'observation': self.observation(space)}
        asset = context.get('asset')
        if action == 'start':
            if asset is None:
                asset = resolve_dataset(context['dataset_ref'], lambda args: request(args, cli=self.cli))
            if asset['source_ref'] != context['dataset_ref']:
                raise ValueError('Dataset action identity differs')
            mount = mount_plan(asset, space)
            if mount['state'] != 'MOUNT_CONFIGURED':
                return {'state': 'NEEDS_ASSISTANCE', 'provider': PROVIDER,
                        'reason': 'Requested dataset is not configured on this space; no start request was sent'}
        phase = space['status']['phase']
        complete_phase = 'ready' if action == 'start' else 'stopped'
        if phase == complete_phase:
            return {'state': 'OBSERVED', 'provider': PROVIDER, 'real_api_call_completed': True,
                    'new_requests': 0, 'observation': self.observation(space)}
        source_phase = 'stopped' if action == 'start' else 'ready'
        if phase != source_phase:
            return {'state': 'OBSERVED', 'provider': PROVIDER, 'real_api_call_completed': True,
                    'new_requests': 0, 'observation': self.observation(space)}
        if action == 'stop' and (idle is None or idle(space) is not True):
            return {'state': 'NEEDS_ASSISTANCE', 'provider': PROVIDER, 'reason': 'Space is not verified idle; no stop request was sent'}
        root = Path(output)
        journal = root/'request.json'
        identity = {'action': action, 'task': task, 'space_uid': space['metadata']['uid'],
                    'namespace': self.profile['namespace'], 'cluster': self.profile['cluster'],
                    'dataset_ref': context.get('dataset_ref'), 'intent_key': context.get('intent_key')}
        signature = json_digest(identity)
        if journal.exists():
            prior = load_json(journal)
            if prior['request_sha256'] != signature:
                raise ValueError('Original space request identity changed')
            if prior['state'] in ('REQUESTING', 'UNKNOWN'):
                return {'state': 'NEEDS_REVIEW', 'provider': PROVIDER, 'new_requests': 0,
                        'reason': 'Original request remains unknown; no second write was sent'}
            return {'state': 'WAITING_APPROVAL' if prior['state'] == 'WAITING_APPROVAL' else 'ACTION_RETURNED',
                    'provider': PROVIDER, 'real_api_call_completed': True, 'new_requests': 0,
                    'request_sha256': signature, 'observation': self.observation(space)}
        # Exclusive creation happens before the platform write. Timeouts and
        # unknown responses never cause automatic resubmission.
        saved = {'state': 'REQUESTING', **identity, 'request_sha256': signature, 'requested_at': utc_now()}
        save_json(journal, saved)
        try:
            reply = self.invoke(self.args(action))
            state = str(reply.get('status', reply.get('state', ''))).lower()
            data = reply.get('data') if isinstance(reply.get('data'), dict) else {}
            pending = state == 'pending_approval' or data.get('status') == 'pending_approval' or data.get('state') == 'pending_approval'
            saved['state'] = 'WAITING_APPROVAL' if pending else 'SUBMITTED'
            saved['platform_references'] = {k:v for obj in (reply, data) for k,v in obj.items()
                if k in ('request_id','approval_id') and isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,160}',v)}
            save_json(journal, saved, overwrite=True)
            observed = self.read()
            return {'state': 'WAITING_APPROVAL' if pending else 'ACTION_RETURNED',
                    'provider': PROVIDER, 'real_api_call_completed': True, 'new_requests': 1,
                    'request_sha256': signature, 'observation': self.observation(observed),
                    'reason': 'Platform reports pending_approval; preserve the original '+action+' request' if pending else ''}
        except Exception:
            saved['state'] = 'UNKNOWN'
            save_json(journal, saved, overwrite=True)
            return {'state': 'NEEDS_REVIEW', 'provider': PROVIDER, 'new_requests': 1,
                    'reason': 'Official space request outcome is unknown; keep the original request and reconcile by read-only observation'}
