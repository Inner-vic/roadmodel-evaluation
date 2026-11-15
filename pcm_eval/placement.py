"""Location-neutral compilation preflight and migration-plan validation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath

from .common import json_digest, utc_now

PROBE_CODE = r'''
import importlib.util,json,pathlib,socket,sys
cfg=json.loads(sys.argv[1]); paths=cfg.get('paths',{})
out={'paths':{k:{'path':v,'exists':pathlib.Path(v).exists()} for k,v in paths.items()},
 'dependencies':{name:importlib.util.find_spec(name) is not None for name in ['numpy','PIL','onnxruntime','paramiko']},
 'route':{'host':cfg['board_host'],'port':cfg.get('board_port',22)}}
try:
 with socket.create_connection((cfg['board_host'],cfg.get('board_port',22)),timeout=5):
  out['route']['tcp_reachable']=True
except OSError as error:out['route'].update(tcp_reachable=False,error_type=type(error).__name__)
out['ready']=all(x['exists'] for x in out['paths'].values()) and all(out['dependencies'].values()) and out['route']['tcp_reachable']
print(json.dumps(out))
'''


def migration_plan(request, target, probe):
    if request.get('kind') != 'compile_alignment':
        raise ValueError('Compilation migration requires a compile request')
    issues = []
    for key in ('workspace_root', 'repo_dir', 'output_root', 'python', 'driver_script', 'lock_file'):
        value = target.get(key)
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            issues.append('target.' + key)
    for key in ('path', 'data_link_file', 'required_manifest'):
        value = target.get('dataset', {}).get(key)
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            issues.append('target.dataset.' + key)
    route = target.get('board_route')
    if route not in ('direct','workstation_gateway'):
        issues.append('board_route: relay requires a verified gateway; it is not inferred from local PC connectivity')
    gateway=target.get('gateway',{})
    if route=='workstation_gateway' and not all(gateway.get(k) for k in ('ssh_alias','board_host','lock_file')):
        issues.append('gateway: configured SSH authentication and the workstation bench lock are required')
    if probe.get('ready') is not True:
        issues.append('runtime_probe: mount/dependencies/board route not verified')
    if route=='direct' and not target.get('exclusive_board_access'):
        issues.append('exclusive_board_access: a lock shared by every board caller or an explicit exclusive reservation is required')
    migrated = deepcopy(request)
    migrated['execution'].update({k: target[k] for k in ('workspace_root','repo_dir','output_root','python','driver_script','lock_file') if k in target})
    migrated['execution']['data_access'] = 'mounted'
    if route=='workstation_gateway':migrated['execution']['gateway']=gateway
    migrated['dataset'].update(target.get('dataset', {}))
    migrated['dataset']['export_root'] = migrated['dataset']['path']
    for key in ('onnx', 'packup_tar'):
        value = target.get('model', {}).get(key)
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            issues.append('target.model.' + key)
        else:
            migrated['model'][key] = value
    if not target.get('ssh_alias'):
        issues.append('target.ssh_alias: execution connection is required')
    else:
        migrated['execution']['ssh_alias'] = target['ssh_alias']
    return {'schema':'roadmodel.placement-plan/v1','created_at':utc_now(),'ready':not issues,'issues':issues,
            'source_request_sha256':json_digest(request),'target':target,'probe':probe,'request':migrated,
            'onnx_provider':'CPUExecutionProvider','raw_dataset_copy_to_workstation':False,
            'validation_scope':'same frozen small frames; no full run'}
