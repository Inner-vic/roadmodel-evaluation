"""First-install account binding and template-based dedicated-space setup.

The regular Feishu card remains a two-input card. This command is deployment
initialization, with optional Codex CUA assistance; it never runs an evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT)]
from pcm_eval.common import load_json, save_json
from pcm_eval.installation import Installation, export_agent_config, export_server_config
from pcm_eval.installation_browser import initialization_prompt, run_initialization_browser
from pcm_eval.lpai import request


def verified_feishu_identity(executable,profile=None):
    # Standard auth only. No login token, device code or raw auth response is saved.
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if sys.platform=='win32' else {}
    command=[executable]+(['--profile',profile] if profile else [])
    result=subprocess.run([*command,'auth','status','--json','--verify'],capture_output=True,
        text=True,encoding='utf-8',timeout=60,**kwargs)
    if result.returncode:raise ValueError('Complete this machine\'s standard Feishu user authorization first')
    value=json.loads(result.stdout);data=value.get('data',value)
    if data.get('verified') is not True:raise ValueError('Feishu user login has not been verified')
    user=data.get('identities',{}).get('user',{})
    if not user.get('openId') or user.get('verified') is not True or not data.get('appId'):
        raise ValueError('No verified Feishu user/application identity was returned')
    return {'open_id':user['openId'],'app_id':data['appId']}


def initialize(args):
    if args.auth_via_deployment:
        from tools.evaluation_agent import exchange
        identity=exchange(load_json(args.auth_via_deployment),'verify_user_identity')
        if set(identity)!={'app_id','open_id'} or not all(identity.values()):
            raise ValueError('Deployment did not return a verified application-scoped identity')
    else:
        identity=verified_feishu_identity(args.lark_cli,args.lark_profile)
    installation=Installation(args.state_dir)
    detail=installation.register(load_json(args.request),identity)
    if args.record_observation:
        detail=installation.record_observation(detail['key'],load_json(args.record_observation))
    if args.record_creation_result:
        detail=installation.record_creation_result(detail['key'],load_json(args.record_creation_result))
    query=lambda argv:request(argv,cli=args.lpai_cli,preserve_envelope=True)
    detail=installation.observe(detail['key'],query)
    if args.retry_confirmed_failure:
        detail=installation.retry_failed_creation(detail['key'],load_json(args.retry_confirmed_failure),query)
    browser_result=None
    capacity_without_post = (detail['state']=='BLOCKED_CAPACITY'
        and detail.get('creation_result',{}).get('source')=='LPAI_NOTEBOOK_API'
        and detail['creation_result'].get('platform_writes')==0)
    if args.create_provider=='official_sdk' and (detail['state']=='WAITING_BROWSER_OBSERVATION' or capacity_without_post):
        if not args.sdk_python or not detail['request'].get('dataset_ref'):
            raise ValueError('SDK creation requires its existing interpreter and the explicit initial dataset link')
        from pcm_eval.notebook_api import NotebookApi
        from pcm_eval.dataset_mount import resolve_dataset
        api=NotebookApi(args.sdk_python,args.request,Path(args.state_dir)/'notebook-api'/detail['key'])
        def query_asset(argv):
            reply=query(argv);return reply.get('data',reply)
        asset=resolve_dataset(detail['request']['dataset_ref'],query_asset)
        body=api.prepare(detail,asset);observation=api.observe(detail,body)
        if observation.get('receipt'):
            detail=installation.record_observation(detail['key'],observation['receipt'])
            detail=installation.claim_creation(detail['key']) or detail
            browser_result=api.create(body)
            if browser_result.get('state')=='BLOCKED_CAPACITY' and browser_result.get('platform_writes')==0:
                detail=installation.record_creation_result(detail['key'],{
                    'source':'LPAI_NOTEBOOK_API','real_api_call_completed':True,'platform_writes':0,
                    'space_name':detail['space_name'],'create_intent_id':detail['create_intent']['id'],
                    'owner':detail['request']['account']['lpai_account'],
                    'namespace':detail['request']['template']['namespace'],
                    'cluster':detail['request']['template']['cluster'],
                    'outcome':'CAPACITY_EXCEEDED','changed':False,'observed_at':time.time()})
            detail=installation.observe(detail['key'],query)
        elif observation['exists'] is True:detail=installation.observe(detail['key'],query)
    if args.run_browser and detail['state']=='WAITING_BROWSER_OBSERVATION':
        browser_result,receipt=run_initialization_browser('observe',detail,cwd=ROOT,binary=args.codex_binary)
        if receipt:
            detail=installation.record_observation(detail['key'],receipt)
            if receipt['exists']:detail=installation.observe(detail['key'],query)
    if args.claim_create or (args.run_browser and detail.get('browser_observation',{}).get('exists') is False):
        claimed=installation.claim_creation(detail['key'])
        if claimed:
            detail=claimed
            if args.run_browser:
                browser_result,receipt=run_initialization_browser('create',detail,cwd=ROOT,binary=args.codex_binary)
                if receipt and receipt.get('outcome')=='CAPACITY_EXCEEDED':
                    detail=installation.record_creation_result(detail['key'],receipt)
                # Even a browser failure/timeout must reconcile the original
                # name by GET, never retry creating another space.
                detail=installation.observe(detail['key'],query)
    if args.agent_config_output:
        if not args.base_agent_config or not args.local_state:raise ValueError('Provide the reviewed deployment base and new local state path')
        export_agent_config(detail,load_json(args.base_agent_config),args.agent_config_output,local_state=args.local_state)
    if args.server_config_output:
        if not args.base_server_config:raise ValueError('Provide the reviewed server deployment base')
        export_server_config(detail,load_json(args.base_server_config),args.server_config_output)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    save_json(output/'initialization.json',detail,overwrite=True)
    if detail['state'] in ('WAITING_BROWSER_OBSERVATION','CREATE_REQUESTED','CREATE_UNKNOWN'):
        action='create' if detail.get('create_intent') else 'observe'
        (output/'codex-browser-action.txt').write_text(initialization_prompt(action,detail),encoding='utf-8')
    value={'state':detail['state'],'space_name':detail['space_name'],'binding_file':detail.get('binding_file'),
        'issues':detail.get('issues',[]),'creation_intent_preserved':bool(detail.get('create_intent')),
        'browser_state':browser_result.get('state') if browser_result else None,
        'next_action':('Initialization complete; install the exported account-bound deployment configuration' if detail['state']=='BOUND'
            else 'Account space limit reached; preserve the original intent and existing spaces' if detail['state']=='BLOCKED_CAPACITY'
            else 'Codex reads the saved plan and assists through actual CUA; refresh this same initialization afterwards'),
        'new_inference_frames':0,'dataset_copied':False}
    print(json.dumps(value,ensure_ascii=True))
    return 0 if detail['state']=='BOUND' else 2


def add_arguments(parser):
    parser.add_argument('--request',required=True);parser.add_argument('--state-dir',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--lark-cli',default='lark-cli')
    parser.add_argument('--lark-profile');parser.add_argument('--lpai-cli',default='lpai-agent');parser.add_argument('--codex-binary')
    parser.add_argument('--auth-via-deployment',help='Read verified Bot-app user identity through an existing reviewed deployment; no tokens are copied')
    parser.add_argument('--create-provider',choices=('codex_cua','official_sdk'),default='codex_cua')
    parser.add_argument('--sdk-python',help='Existing official SDK interpreter; authentication stays in its standard configuration')
    parser.add_argument('--retry-confirmed-failure',help='Operator-only: one reconciled retry of a returned 500; never used for unknown/timeout responses')
    parser.add_argument('--record-observation');parser.add_argument('--record-creation-result');parser.add_argument('--claim-create',action='store_true')
    parser.add_argument('--run-browser',action='store_true',help='Use existing Codex CUA; do not change browser permissions')
    parser.add_argument('--base-agent-config');parser.add_argument('--agent-config-output');parser.add_argument('--local-state')
    parser.add_argument('--base-server-config');parser.add_argument('--server-config-output')
    return parser


def main(argv=None):
    args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    try:return initialize(args)
    except (ValueError,OSError,KeyError) as error:
        # Only our validation errors are returned; do not serialize CLI stderr.
        message=str(error) if isinstance(error,ValueError) else type(error).__name__
        print(json.dumps({'state':'NEEDS_INITIALIZATION','reason':message[:300]},ensure_ascii=True));return 2


if __name__=='__main__':raise SystemExit(main())
