"""Per-installation account and dedicated devspace initialization.

No credentials are persisted. Creation uses the existing Codex CUA adapter;
official readback, not a browser success message, establishes the UID binding.
"""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import getpass
import json
from pathlib import Path
import platform
import re
import sqlite3
import time
import uuid

from .common import json_digest, load_json, save_json, utc_now
from .space_api import OfficialSpaces


SCHEMA = 'roadmodel.installation-request/v1'
NAME = r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}'


def machine_identity():
    return json_digest({'host':platform.node().lower(),'user':getpass.getuser().lower(),'os':platform.system()})


def validate_request(value):
    value = deepcopy(value)
    def reject_credentials(item):
        if isinstance(item,dict):
            for k,v in item.items():
                if re.search(r'(?i)(?:^|_)(?:token|password|secret|api_key|private_key)(?:$|_)',k):
                    raise ValueError('Store authentication in the standard CLI/SSH configuration, not initialization files')
                reject_credentials(v)
        elif isinstance(item,list):
            for v in item:reject_credentials(v)
    reject_credentials(value)
    allowed = {'schema', 'account', 'template', 'ssh', 'adopt_existing', 'dataset_ref'}
    if set(value)-allowed or value.get('schema') != SCHEMA:
        raise ValueError('Invalid initialization request; credentials and arbitrary settings are not accepted')
    account, template, ssh = (value[k] for k in ('account', 'template', 'ssh'))
    if set(account) != {'lpai_account', 'lpai_login', 'feishu_user_id', 'feishu_app_id'}:
        raise ValueError('Bind both the LPAI account and Feishu user')
    account['lpai_account'] = account['lpai_account'].strip().lower()
    account['lpai_login'] = account['lpai_login'].strip().lower()
    if not re.fullmatch(NAME,account['lpai_login']):
        raise ValueError('Provide the actual SSO login shown in the LPAI personal center')
    if not re.fullmatch(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9.-]+', account['lpai_account']):
        raise ValueError('Provide the actual LPAI owner account email')
    if not re.fullmatch(r'ou_[A-Za-z0-9_-]{3,100}', account['feishu_user_id']):
        raise ValueError('Invalid Feishu identity')
    if not re.fullmatch(r'cli_[A-Za-z0-9]{3,100}', account['feishu_app_id']):
        raise ValueError('Bind the Feishu application as well as its app-scoped user identity')
    required = {'id', 'namespace', 'cluster', 'project', 'resource_queue', 'image', 'resources',
                'accesses', 'idle_shutdown', 'ssh_host'}
    if set(template) != required or not all(template.values()):
        raise ValueError('Initialization requires the complete approved environment template')
    for field in ('id', 'namespace', 'cluster', 'project', 'resource_queue'):
        if not re.fullmatch(NAME, template[field]):
            raise ValueError('Invalid template identity: '+field)
    if not re.fullmatch(r'[A-Za-z0-9.-]+', template['ssh_host']):
        raise ValueError('Invalid approved SSH host')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:@-]+', template['image']) or 'REPLACE' in template['image']:
        raise ValueError('Choose an actual accessible evaluation image')
    resources = template['resources']
    permitted = {'cpu','gpu','memory','gpuType','request','limit','affinity','taints',
                 'resourceComboId','resourceComboNum','resourceQueue'}
    if not isinstance(resources, dict) or set(resources)-permitted or not resources.get('gpuType'):
        raise ValueError('Use the reviewed GPU resource specification')
    if resources.get('resourceQueue') != template['resource_queue']:
        raise ValueError('Resource queue differs from the template')
    if template['accesses'] != {'visibility':'personal', 'ports':[22,8888]}:
        raise ValueError('Use personal visibility and the existing SSH/notebook ports')
    if template['idle_shutdown'] != {'enabled':True, 'minutes':240}:
        raise ValueError('The initialization template must preserve the requested four-hour automatic shutdown')
    if set(ssh) != {'credential_name', 'identity_file'} or not re.fullmatch(NAME, ssh['credential_name']):
        raise ValueError('Select this account\'s existing SSH credential and local key reference')
    if not isinstance(ssh['identity_file'], str) or not ssh['identity_file'] or any(c in ssh['identity_file'] for c in '\r\n\x00'):
        raise ValueError('Invalid local SSH key reference')
    if value.get('adopt_existing'):
        adoption = value['adopt_existing']
        if (set(adoption) != {'name','uid'} or not re.fullmatch(NAME, adoption['name'])
                or not re.fullmatch(r'[a-fA-F0-9-]{8,80}', adoption['uid'])):
            raise ValueError('Adoption requires the existing space name and real UID')
    if value.get('dataset_ref'):
        from .dataset_mount import dataset_reference
        if dataset_reference(value['dataset_ref'])['namespace'] != template['namespace']:
            raise ValueError('Initial dataset and environment namespaces differ')
    return value


def profile_from_binding(binding):
    request, space = binding['request'], binding['space']
    template = deepcopy(request['template'])
    template.pop('id');host = template.pop('ssh_host')
    template['accesses']['ssh_credential'] = request['ssh']['credential_name']
    return {'namespace':space['namespace'], 'cluster':space['cluster'], 'space_name':space['name'],
            'space_uid':space['uid'], 'ssh_host':host, 'ssh_port':space['port'],
            'identity_file':request['ssh']['identity_file'], 'lpai_account':request['account']['lpai_account'],
            'lifecycle_provider':'official_cli', 'environment_template':template}


def validate_space(raw, request, name, expected_uid=None, query=None):
    """Validate full official details before stripping them down to a binding."""
    request = validate_request(request);t = request['template'];m=raw.get('metadata',{});s=raw.get('spec',{})
    annotations, labels = m.get('annotations',{}), m.get('labels',{})
    issues=[]
    checks = {
        'owner':annotations.get('lpai.org/owner','').lower()==request['account']['lpai_account'],
        'name':m.get('name')==name, 'namespace':m.get('namespace')==t['namespace'],
        'cluster':s.get('clusterName')==t['cluster'],
        'uid':bool(m.get('uid')) and (expected_uid is None or m['uid']==expected_uid),
        'image':t['image'] in (s.get('image'), annotations.get('lpai.org/base-image')),
        'project':annotations.get('lpai.org/gems-project')==t['project'],
        'visibility':labels.get('lpai.org/visible')=='personal',
        'resource_queue':labels.get('lpai.org/resourceQueue')==t['resource_queue'],
        'four_hour_shutdown':s.get('autoCulling',{}).get('enable') is True
            and str(s.get('autoCulling',{}).get('windowMinutes'))=='240',
    }
    def contains(actual, expected):
        if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and contains(actual[k],v) for k,v in expected.items())
        return actual==expected
    checks['resources']=contains(s.get('resources'),t['resources'])
    issues.extend(k for k,v in checks.items() if not v)
    if issues:return {'state':'CONFIGURATION_MISMATCH','issues':issues}
    if request.get('dataset_ref'):
        from .dataset_mount import resolve_dataset, mount_plan
        if query is None:return {'state':'CONFIGURATION_MISMATCH','issues':['initial_dataset_readback_required']}
        def read_asset(args):
            value=query(args);return value.get('data',value)
        asset=resolve_dataset(request['dataset_ref'],read_asset)
        if mount_plan(asset,raw)['state']!='MOUNT_CONFIGURED':
            return {'state':'CONFIGURATION_MISMATCH','issues':['initial_dataset_mount']}
    draft={'request':request,'space':{'namespace':t['namespace'],'cluster':t['cluster'],
            'name':name,'uid':m['uid'],'port':22}}
    profile=profile_from_binding(draft)
    # Reuse the UID, approved-host and existing-credential checks used at runtime.
    observed=OfficialSpaces(profile,invoke=lambda args:{'data':raw}).read()
    return {'state':'VERIFIED','space':{'namespace':t['namespace'],'cluster':t['cluster'],
        'name':name,'uid':m['uid'],'port':observed['runtime_access']['port']},
        'observed_phase':observed['status']['phase']}


class Installation:
    def __init__(self, root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        identity=self.root/'installation.json'
        try:save_json(identity,{'schema':'roadmodel.installation/v1','id':uuid.uuid4().hex,'machine_binding_sha256':machine_identity()})
        except FileExistsError:pass
        stored=load_json(identity);self.id=stored['id']
        if stored.get('machine_binding_sha256') != machine_identity():
            raise ValueError('This state belongs to another machine/login; initialize in a new local state directory')
        if not re.fullmatch(r'[a-f0-9]{32}',self.id):raise ValueError('Invalid installation identity')
        with self.connect() as db:
            db.execute('create table if not exists bindings(key text primary key,request text not null,detail text not null)')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.root/'installation.sqlite3',timeout=15)
        db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    def register(self, request, feishu_identity):
        request=validate_request(request)
        if feishu_identity != {'open_id':request['account']['feishu_user_id'],'app_id':request['account']['feishu_app_id']}:
            raise ValueError('The currently verified Feishu login differs from this initialization account')
        key=json_digest({'installation':self.id,'account':request['account'],
                         'namespace':request['template']['namespace'],'cluster':request['template']['cluster']})
        with self.connect() as db:
            db.execute('begin immediate')
            row=db.execute('select * from bindings where key=?',(key,)).fetchone()
            if row:
                if json.loads(row['request'])!=request:
                    raise ValueError('This installation/account is already registered; do not silently change its template, key or space')
                return json.loads(row['detail'])
            name=request.get('adopt_existing',{}).get('name') or 'roadmodel-eval-'+key[:20]
            detail={'schema':'roadmodel.environment-binding/v1','key':key,'installation_id':self.id,
                'state':'REGISTERED','space_name':name,'request':request,'created_at':utc_now(),
                'machine_binding_sha256':machine_identity(),
                'stop_after_evaluation':False,'binding_file':str(self.root/'bindings'/(key+'.json'))}
            db.execute('insert into bindings values(?,?,?)',(key,json.dumps(request),json.dumps(detail)))
            return detail

    def get(self,key):
        with self.connect() as db:row=db.execute('select detail from bindings where key=?',(key,)).fetchone()
        if not row:raise ValueError('Unknown installation/account binding')
        return json.loads(row['detail'])

    def save(self,detail):
        with self.connect() as db:
            db.execute('begin immediate')
            row=db.execute('select detail from bindings where key=?',(detail['key'],)).fetchone()
            if not row:raise ValueError('Unknown initialization')
            current=json.loads(row['detail'])
            if current.get('space'):
                if detail.get('space') and detail['space']['uid']!=current['space']['uid']:
                    raise ValueError('A bound space UID cannot be replaced')
                detail.setdefault('space',current['space'])
            if current.get('create_intent'):
                if detail.get('create_intent') not in (None,current['create_intent']):
                    raise ValueError('Original creation intent cannot be replaced')
                detail['create_intent']=current['create_intent']
            db.execute('update bindings set detail=? where key=?',(json.dumps(detail),detail['key']))

    def observe(self,key,query):
        detail=self.get(key);request=detail['request'];t=request['template']
        args=['devspaces','get','--namespace',t['namespace'],'--cluster',t['cluster'],'--name',detail['space_name']]
        try:
            reply=query(args);raw=reply.get('data',reply)
        except (ValueError,OSError):
            # A CLI failure never proves absence and never authorizes creation.
            detail['state']=('BLOCKED_CAPACITY' if detail.get('creation_failure')=='CAPACITY_EXCEEDED' else
                'BOUND_UNAVAILABLE' if detail.get('space') else
                'CREATE_UNKNOWN' if detail.get('create_intent') else 'WAITING_BROWSER_OBSERVATION')
            self.save(detail);return detail
        expected=detail.get('space',{}).get('uid') or request.get('adopt_existing',{}).get('uid')
        checked=validate_space(raw,request,detail['space_name'],expected,query=query)
        if checked['state']!='VERIFIED':
            detail.update(state=checked['state'],issues=checked['issues']);self.save(detail);return detail
        detail.update(state='BOUND',space=checked['space'],verified_at=utc_now(),observed_phase=checked['observed_phase'])
        detail.pop('issues',None)
        detail.pop('creation_failure',None)
        detail['profile']=profile_from_binding(detail)
        # The exclusive account record is authoritative. This replaceable export
        # contains only approved identity/path references, never private keys.
        self.save(detail)
        save_json(detail['binding_file'],detail,overwrite=Path(detail['binding_file']).exists())
        return detail

    def record_observation(self,key,receipt):
        detail=self.get(key);request=detail['request']
        browser=receipt.get('source')=='CODEX_CUA' and receipt.get('real_browser_call_completed') is True
        api=(receipt.get('source')=='LPAI_NOTEBOOK_API' and receipt.get('real_api_call_completed') is True
             and receipt.get('http_status')==404 and receipt.get('exists') is False
             and receipt.get('owner')==request['account']['lpai_account'])
        if (not (browser or api)
                or receipt.get('space_name')!=detail['space_name']
                or receipt.get('lpai_login','').lower()!=request['account']['lpai_login'].lower()
                or receipt.get('namespace')!=request['template']['namespace']
                or receipt.get('cluster')!=request['template']['cluster']
                or type(receipt.get('exists')) is not bool):
            raise ValueError('Initialization requires the actual browser account and matching space observation')
        if not time.time()-300 <= receipt.get('observed_at',0) <= time.time()+30:
            raise ValueError('Browser account/absence observation expired')
        detail['browser_observation']=deepcopy(receipt);self.save(detail);return detail

    def claim_creation(self,key):
        with self.connect() as db:
            db.execute('begin immediate');row=db.execute('select detail from bindings where key=?',(key,)).fetchone()
            if not row:raise ValueError('Unknown initialization')
            d=json.loads(row['detail']);o=d.get('browser_observation',{})
            if d.get('create_intent') or d.get('space') or d['state']=='BOUND':return None
            if d['request'].get('adopt_existing'):raise ValueError('Never create a replacement for an explicitly adopted UID')
            if o.get('exists') is not False or o.get('observed_at',0)<time.time()-300:
                raise ValueError('Creation requires a fresh confirmed absence under the intended account')
            d['create_intent']={'id':uuid.uuid4().hex,'requested_at':utc_now()}
            d['state']='CREATE_REQUESTED'
            db.execute('update bindings set detail=? where key=?',(json.dumps(d),key))
            return d

    def retry_failed_creation(self,key,receipt,query):
        """Operator-only recovery of one returned 5xx, after full-list reconciliation.

        This is never called by the background browser adapter. A timeout or
        missing GET alone cannot authorize another submission.
        """
        detail=self.get(key);intent=detail.get('create_intent',{})
        if (detail.get('space') or detail.get('creation_retry') or not intent
                or receipt.get('source')!='CODEX_CUA' or receipt.get('real_browser_call_completed') is not True
                or receipt.get('space_name')!=detail['space_name']
                or receipt.get('create_intent_id')!=intent['id'] or receipt.get('http_status')!=500
                or receipt.get('lpai_login')!=detail['request']['account']['lpai_login']
                or receipt.get('namespace')!=detail['request']['template']['namespace']
                or receipt.get('cluster')!=detail['request']['template']['cluster']
                or receipt.get('exists') is not False
                or not time.time()-300 <= receipt.get('observed_at',0) <= time.time()+30
                or receipt.get('observed_at',0)<datetime.fromisoformat(intent['requested_at']).timestamp()+120
                or time.time()-datetime.fromisoformat(intent['requested_at']).timestamp()<120):
            raise ValueError('Creation retry requires a fresh operator-confirmed returned 500 and reconciled absence')
        template=detail['request']['template']
        reply=query(['devspaces','list','--namespace',template['namespace'],'--cluster',template['cluster'],
                     '--page-size','100','--page-num','1'])
        data=reply.get('data',reply)
        if not isinstance(data,dict) or not isinstance(data.get('items'),list) or data.get('total')!=len(data['items']):
            raise ValueError('A complete official space list is required before retry')
        if any(s.get('metadata',{}).get('name')==detail['space_name'] for s in data['items']):
            raise ValueError('Space exists; reconcile it instead of retrying creation')
        with self.connect() as db:
            db.execute('begin immediate')
            current=json.loads(db.execute('select detail from bindings where key=?',(key,)).fetchone()['detail'])
            if current.get('space') or current.get('creation_retry'):raise ValueError('Creation retry already consumed')
            current.update(state='CREATE_REQUESTED',creation_retry={**deepcopy(receipt),'authorized_at':utc_now()})
            db.execute('update bindings set detail=? where key=?',(json.dumps(current),key))
        return current

    def record_creation_result(self,key,receipt):
        detail=self.get(key)
        browser = receipt.get('source')=='CODEX_CUA' and receipt.get('real_browser_call_completed') is True
        api = (receipt.get('source')=='LPAI_NOTEBOOK_API' and receipt.get('real_api_call_completed') is True
               and receipt.get('platform_writes')==0 and receipt.get('owner')==detail['request']['account']['lpai_account']
               and receipt.get('namespace')==detail['request']['template']['namespace']
               and receipt.get('cluster')==detail['request']['template']['cluster'])
        if (not detail.get('create_intent') or not (browser or api)
                or receipt.get('space_name')!=detail['space_name']
                or receipt.get('create_intent_id')!=detail['create_intent']['id']
                or receipt.get('outcome')!='CAPACITY_EXCEEDED' or receipt.get('changed') is not False
                or not datetime.fromisoformat(detail['create_intent']['requested_at']).timestamp() <= receipt.get('observed_at',0) <= time.time()+30):
            raise ValueError('Creation capacity result requires matching real, fresh official evidence')
        detail.update(state='BLOCKED_CAPACITY',creation_failure='CAPACITY_EXCEEDED',
            creation_result=deepcopy(receipt),issues=['Platform per-account space limit reached; no space was created or deleted'])
        self.save(detail);return detail


def validate_runtime_binding(cfg):
    path=cfg.get('environment_binding_file')
    if not path:return  # Existing reviewed installations remain compatible.
    binding=load_json(path)
    if (binding.get('state')!='BOUND' or binding.get('machine_binding_sha256')!=machine_identity()
            or cfg['owner']!=binding['request']['account']['feishu_user_id']
            or cfg.get('feishu_app_id')!=binding['request']['account']['feishu_app_id']
            or cfg['profile']!=binding['profile'] or cfg.get('installation_id')!=binding['installation_id']):
        raise ValueError('Runtime account/environment differs from initialized binding; initialize this account first')


def export_agent_config(binding,base,output,*,local_state):
    if binding['state']!='BOUND':raise ValueError('Space has not passed official ownership/template readback')
    if base.get('feishu_app_id')!=binding['request']['account']['feishu_app_id']:
        raise ValueError('Reviewed deployment and verified Feishu login must use the same application; open_id is app-scoped')
    cfg=deepcopy(base)
    cfg.update(owner=binding['request']['account']['feishu_user_id'],profile=binding['profile'],
        environment_binding_file=binding['binding_file'],installation_id=binding['installation_id'],
        space_backend='official_cli',local_state=str(Path(local_state).resolve()),
        agent_id='roadmodel-'+binding['key'][:20],browser_backend_verified=False)
    validate_runtime_binding(cfg)
    save_json(output,cfg)
    return cfg


def export_server_config(binding,base,output):
    if binding['state']!='BOUND' or base.get('feishu_app_id')!=binding['request']['account']['feishu_app_id']:
        raise ValueError('Server deployment must match the verified Feishu application and initialized space')
    cfg=deepcopy(base);profile=binding['profile']
    cfg.update(owner=binding['request']['account']['feishu_user_id'],agent_id='roadmodel-'+binding['key'][:20],
        profile=profile,space_backend='official_cli',installation_binding_id=binding['key'])
    cfg['model_space']={'host':profile['ssh_host'],'port':profile['ssh_port'],'user':'root','identity_file':profile['identity_file']}
    save_json(output,cfg);return cfg
