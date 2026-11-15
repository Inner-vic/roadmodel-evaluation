"""Desktop material and publication agent; reuses the existing Bot transport."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from pcm_eval.common import json_digest, load_json, save_json
from pcm_eval.evaluation_source import prepare
from pcm_eval.delivery_source import LarkSource
from pcm_eval.delivery_workflow import material_identity
from types import SimpleNamespace
from pcm_eval.lpai import request as lpai_request
from pcm_eval.combined_publication import publish_combined
from pcm_eval.publication import cli_command
from delivery_demo_agent import rpc
from model_space_desktop import own_lock

RPC = '''import json,sys,pathlib,subprocess,shlex
sys.path.insert(0,sys.argv[1])
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime(sys.argv[2]);p=json.load(sys.stdin)
cfg=json.loads(r.config_file.read_text())
if p['agent']!=cfg['agent_id'] or p['owner']!=cfg['owner']:raise ValueError('Agent identity differs')
action=p['action']
if action=='verify_user_identity':
 binding=cfg['workstation_lark']
 result=subprocess.run([binding['executable'],'--profile',binding['profile'],'auth','status','--json','--verify'],capture_output=True,text=True,timeout=60)
 if result.returncode:raise ValueError('Deployment Feishu authorization is not verified')
 data=json.loads(result.stdout);data=data.get('data',data);user=data.get('identities',{}).get('user',{})
 if data.get('verified') is not True or user.get('verified') is not True or user.get('openId')!=cfg['owner'] or not data.get('appId'):raise ValueError('Deployment Feishu identity differs')
 v={'app_id':data['appId'],'open_id':user['openId']}
elif action=='claim':v=r.claim_materials(p['agent'],p['owner'])
elif action=='complete':r.complete_materials(p['agent'],p['task'],p['digest'],p['detail']);v={'state':'ACCEPTED'}
elif action=='claim_publication':v=r.claim_publication(p['agent'],p['owner'])
elif action=='complete_publication':r.complete_publication(p['agent'],p['task'],p['digest'],p['detail']);v={'state':'ACCEPTED'}
elif action=='space_observation':v=r.space_observation()
elif action=='record_space_observation':r.record_space_observation(p['observation']);v={'state':'ACCEPTED'}
elif action=='space_action':v=r.request_space_action(p['task'],p['space_action'],p['context'])
elif action=='claim_space_action':v=r.claim_space_action(p['agent'],p['owner'])
elif action=='complete_space_action':r.complete_space_action(p['agent'],p['action_id'],p['detail']);v={'state':'ACCEPTED'}
elif action=='read_delivery':
 from pcm_eval.delivery_source import LarkSource
 r.get(p['task'],p['owner'])
 with r.connect() as db:row=db.execute("select request from materials where task=? and state='READING'",(p['task'],)).fetchone()
 if not row:raise ValueError('This work order has no claimed material read')
 binding=cfg['workstation_lark'];v=LarkSource(executable=binding['executable'],profile=binding['profile']).read(json.loads(row['request'])['model_delivery_url'])
elif action=='publish_remote':
 r.get(p['task'],p['owner'])
 with r.connect() as db:row=db.execute("select * from publications where task=? and state='WRITING'",(p['task'],)).fetchone()
 if not row or row['digest']!=p['digest']:raise ValueError('Publication is not claimed or differs')
 sys.path.insert(0,str(pathlib.Path(sys.argv[1])/'tools'))
 from evaluation_agent import publish
 binding=cfg['workstation_lark'];v=publish({'local_state':str(r.root/'server-publications'),'lark_executable':binding['executable'],'lark_profile':binding['profile']},dict(row))
elif action=='stage_visual':
 r.get(p['task'],p['owner'])
 with r.connect() as db:row=db.execute("select request from materials where task=? and state='READING'",(p['task'],)).fetchone()
 body=p['body'];policy=cfg.get('visual_asset_staging',{})
 if not row or not policy.get('enabled') or body['source_ref']!=json.loads(row['request']).get('vision_dataset_ref') or body['source_ref']!=policy['source_ref']:raise ValueError('Visual asset input is not authorized for this work order')
 if body['selection']!=r.get(p['task'],p['owner'])['options']['selection'] or len(body['selected_frames'])>policy['max_frames']:raise ValueError('Visual asset scope differs from confirmed validation')
 if len(body['files'])!=2*len(body['selected_frames']) or sum(x['bytes'] for x in body['files'])>policy['max_bytes']:raise ValueError('Visual asset file budget exceeded')
 observation=r.space_observation();space=(observation or {}).get('space') or {}
 if space.get('status',{}).get('phase')!='ready' or space.get('metadata',{}).get('uid')!=cfg['profile']['space_uid']:raise ValueError('Space is not verified ready for visual asset reads')
 access=space['runtime_access'];target={'host':access['host'],'port':access['port'],'user':'root','identity_file':cfg['profile']['identity_file']}
 from pcm_eval.lpai_model_transport import call
 code=(pathlib.Path(sys.argv[1])/'pcm_eval/visual_asset_input.py').read_text()+chr(10)+'import sys,json'+chr(10)+'print(json.dumps(stage(json.load(sys.stdin))))'
 v=json.loads(call(target,['python3','-c',code],json.dumps(body).encode(),timeout=90))
elif action=='space_journal':
 path=r.space_journal_path(p['task'],p.get('dataset_ref'));v=json.loads(path.read_text()) if path.exists() else {}
elif action=='save_space_journal':r.save_space_journal(p['task'],p['journal'],p.get('dataset_ref'));v={'state':'ACCEPTED'}
elif action=='offline_compile':
 from pcm_eval.delivery_workflow import _checked_file
 from pcm_eval.compile_preparation import prepare_compile
 root=pathlib.Path(sys.argv[1]);catalog=json.loads((root/'configs/delivery_catalog.json').read_text());identity=p['materials']
 profiles=[m for m in catalog['models'] if all(m['materials'][k]==identity[k] for k in ('version','onnx','packages'))]
 if len(profiles)!=1:v={'state':'BLOCKED','issues':['编译交付尚未绑定本版离线材料']}
 else:
  spec=profiles[0]['chains']['compile_alignment'];req=json.loads(_checked_file(root,spec['request']).read_text())
  v=prepare_compile({'version':identity['version'],'request':req},r.root/'runs'/p['task']/'compile-preparation')
  with r.connect() as db:db.execute('update chains set state=?,detail=? where task=? and kind=?',('WAITING_BENCH',json.dumps({'stage':'编译材料离线核验'+v['state'],'preparation':v,'issues':['等待台架可用']}),p['task'],'compile_alignment'))
elif action in ('probe','idle'):
 with r.connect() as db:row=db.execute("select request from materials where task=? and state='READING'",(p['task'],)).fetchone()
 plan=p['plan'];profile=cfg['profile'];observation=r.space_observation()
 observed_space=(observation or {}).get('space') or {}
 access=observed_space.get('runtime_access',{})
 host=access.get('host',profile['ssh_host']);port=access.get('port',profile['ssh_port'])
 if not row or plan['asset']['source_ref'] not in {json.loads(row['request']).get(k) for k in ('dataset_ref','vision_dataset_ref')} or plan.get('space_name',profile['space_name'])!=profile['space_name']:raise ValueError('Probe request identity differs')
 if host!=profile['ssh_host'] or type(port) is not int or not 1<=port<=65535:raise ValueError('Observed space access differs from approved cluster')
 if action=='probe':
  code=(pathlib.Path(sys.argv[1])/'pcm_eval/dataset_mount.py').read_text()+'\\nplan='+repr(plan)+'\\nresult=probe_mounted_index(plan,index_relative=plan["asset"].get("index_relative_hint"),include_manifest=True)\\nimport subprocess\\np=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)\\nrows=[tuple(map(int,line.split(","))) for line in p.stdout.splitlines() if line.strip()]\\nchoice=max(rows,key=lambda row:row[1]) if rows else None\\nif choice is None or choice[1]<6144:raise ValueError("No visible GPU has enough free memory")\\nresult["dataset_source"]["runtime"]={"gpu_index":choice[0],"free_mib":choice[1]}\\nprint(json.dumps(result))'
 else:
  code="""import pathlib,os,json,subprocess
unknown=0;own=set()
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 own.add(int(p.name))
 if int(p.name)==os.getpid():continue
 try:
  names=[pathlib.PurePosixPath(x.decode(errors='ignore')).name for x in (p/'cmdline').read_bytes().split(bytes([0]))[:3]]
  if names and names[0].startswith('python') and not any(n in ('jupyter-lab','jupyter-notebook','jupyterlab','jupyter_server','supervisord','update_pubkey.py') for n in names[1:]):unknown+=1
 except OSError:pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
busy=sum(int(x.strip()) in own for x in gpu.stdout.splitlines() if x.strip().isdigit()) if gpu.returncode==0 else -1
print(json.dumps({'idle':unknown==0 and busy==0,'unknown_python':unknown,'gpu_processes':busy}))"""
 args=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=accept-new','-p',str(port),'-i',profile['identity_file'],'root@'+host,shlex.join(['python3','-c',code])]
 result=subprocess.run(args,capture_output=True,text=True,timeout=90)
 if result.returncode:raise ValueError('LPAI mounted dataset probe failed')
 v=json.loads(result.stdout)
 if action=='probe':v['dataset_source']['runtime']['execution_space']={'host':host,'port':port,'user':'root','identity_file':profile['identity_file']}
else:raise ValueError('Unknown evaluation operation')
print(json.dumps(v))
'''


def exchange(cfg, action, **fields):
    return rpc({**cfg, 'rpc_code': RPC}, {'action': action, 'agent': cfg['agent_id'], 'owner': cfg['owner'], **fields})


def platform_query(cfg, args):
    if cfg.get('space_backend') == 'official_cli' and args[:2] == ['devspaces', 'get']:
        from pcm_eval.space_api import OfficialSpaces
        client = OfficialSpaces(cfg['profile'], cli=cfg['lpai_cli'])
        space = client.read()
        exchange(cfg, 'record_space_observation', observation=client.observation(space))
        return space
    return lpai_request(args, cli=cfg['lpai_cli'])


def prepare_materials(cfg, row):
    request = json.loads(row['request'])
    if json_digest(request) != row['digest']:
        raise ValueError('原卡提交内容哈希变化')
    try:
        materials = (exchange(cfg,'read_delivery',task=row['task']) if cfg.get('material_source')=='workstation_cli'
                     else LarkSource(executable=cfg.get('lark_executable')).read(request['model_delivery_url']))
        offline = exchange(cfg, 'offline_compile', task=row['task'], materials=material_identity(materials))
        actions = None
        if cfg['profile'].get('environment_template'):
            def for_dataset(ref):
                return {'journal': exchange(cfg, 'space_journal', task=row['task'], dataset_ref=ref),
                    'observe': lambda profile: exchange(cfg, 'space_observation'),
                    'request': lambda action, context: exchange(cfg, 'space_action', task=row['task'], space_action=action, context=context),
                    'save': lambda journal: exchange(cfg, 'save_space_journal', task=row['task'], dataset_ref=ref, journal=journal),
                    'idle': lambda space: exchange(cfg, 'idle', task=row['task'], plan={
                        'space_uid': cfg['profile']['space_uid'], 'asset': {'source_ref': ref}})['idle']}
            actions = {'for_dataset': for_dataset}
        workflow = prepare(request, row['task'], profile=cfg['profile'],
            query=lambda args: platform_query(cfg, args),
            probe=lambda plan: exchange(cfg, 'probe', task=row['task'], plan=plan),
            source=SimpleNamespace(read=lambda url: materials), space_actions=actions,
            native_profile=cfg.get('native_profile'), compile_profile=cfg.get('compile_profile'),
            selection=row.get('options', {}).get('selection'),
            vision_source=(lambda ref, selected, occ: prepare_visual_asset(cfg,row['task'],ref,selected,occ))
                          if cfg.get('visual_asset_staging',{}).get('enabled') else None)
        if 'jobs' not in workflow:
            exchange(cfg, 'complete', task=row['task'], digest=row['digest'], detail=workflow)
            return
        for job in workflow['jobs']:
            for key in ('binding_file', 'gt_preparation_file'):
                if job.get(key):
                    relative = Path(job[key]).resolve().relative_to(ROOT.resolve())
                    job[key] = cfg['release']+'/'+relative.as_posix()
        workflow.pop('workflow_sha256')
        workflow['workflow_sha256'] = json_digest(workflow)
        detail = {'workflow': workflow}
    except Exception as error:
        detail = {'error': '评测准备待核验：'+str(error)[:350],
                  'failure_kind':'preparation_environment', 'stage':'评测环境准备待核验',
                  'error_type':type(error).__name__}
    exchange(cfg, 'complete', task=row['task'], digest=row['digest'], detail=detail)


def prepare_visual_asset(cfg, task, ref, selection, occ):
    policy=cfg['visual_asset_staging']
    if policy.get('sdk_python'):
        process=subprocess.run([policy['sdk_python'],str(ROOT/'tools/visual_asset_source.py')],
            input=json.dumps({'ref':ref,'selection':selection,'policy':policy,'task':task}),
            capture_output=True,text=True,encoding='utf-8',timeout=90,
            **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
        if process.returncode:
            raise ValueError('官方资产SDK读取未完成，保留本工单核查；不会记录对象URL或凭据')
        body=json.loads(process.stdout)
    else:
        from visual_asset_source import prepare_request
        body = prepare_request(ref, selection, policy, task)
    result = exchange(cfg, 'stage_visual', task=task, body=body)
    result['runtime'] = occ['runtime']
    return result


def publish(cfg, row, *, invoke=None):
    if cfg.get('publication_location')=='workstation':
        return exchange(cfg,'publish_remote',task=row['task'],digest=row['digest'])
    payload = json.loads(row['payload'])
    if json_digest(payload) != row['digest'] or payload['task'] != row['task']:
        raise ValueError('发布请求内容与工单不一致')
    task_folder = Path(cfg['local_state'])/'publications'/row['task']
    folder = task_folder/'payloads'/row['digest']
    folder.mkdir(parents=True, exist_ok=True)
    xml, data = folder/'report.xml', folder/'report.json'
    if xml.exists() or data.exists():
        if not xml.exists() or not data.exists() or xml.read_text(encoding='utf-8') != payload['xml'] or load_json(data) != payload['data']:
            raise ValueError('原发布目录内容待核验')
    else:
        xml.write_text(payload['xml'], encoding='utf-8')
        save_json(data, payload['data'])
    report = {'report_xml': str(xml), 'report_json': str(data), 'source_sha256': payload['source_sha256']}
    sequence = 0
    def cli(args, *, cwd):
        nonlocal sequence
        sequence += 1
        command = [cfg['lark_executable']] if cfg.get('lark_executable') else cli_command()
        if cfg.get('lark_profile'):command+=['--profile',cfg['lark_profile']]
        proc = subprocess.run([*command, *args, '--as', 'user'], cwd=cwd,
                              capture_output=True, timeout=180,
                              **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
        (folder/f'cli-{sequence}.stdout.json').write_bytes(proc.stdout)
        (folder/f'cli-{sequence}.stderr.txt').write_bytes(proc.stderr)
        value = json.loads(proc.stdout or proc.stderr)
        if proc.returncode or value.get('ok') is not True or value.get('data', {}).get('warnings'):
            raise ValueError('飞书发布结果待核验，请保留原收据')
        return value['data']
    receipt = publish_combined(report, task_folder/'publication', invoke=invoke or cli)
    # publish_combined creates and reads using the same requester identity.
    return {**receipt, 'requester_read_verified': True}


def main(config, *, once=False):
    cfg = load_json(config)
    from pcm_eval.installation import validate_runtime_binding
    validate_runtime_binding(cfg)
    folder = Path(cfg['local_state'])
    with own_lock(folder):
        if cfg.get('paused'):
            status = {'state': 'PAUSED', 'pid': os.getpid(), 'updated': time.time()}
            save_json(folder/'status.json', status)
            return status
        if cfg.get('auth_probe_on_start'):
            binary = LarkSource(executable=cfg.get('lark_executable')).executable
            probe = subprocess.run([binary, 'auth', 'status', '--json', '--verify'], capture_output=True,
                encoding='utf-8', timeout=45,
                **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
            try:
                value=json.loads(probe.stdout or probe.stderr);value=value.get('data',value)
                user=value.get('identities',{}).get('user',{})
                safe={'verified':value.get('verified'),'identity':value.get('identity'),
                      'user_status':user.get('status'),'token_status':user.get('tokenStatus'),
                      'open_id':user.get('openId'),'exit_code':probe.returncode}
            except ValueError:safe={'exit_code':probe.returncode,'state':'NON_JSON'}
            safe.update(executable=binary,context_variable_names=sorted(k for k in os.environ if k.startswith(('LARK','FEISHU','EPT','CODEX'))))
            save_json(folder/'auth-probe.json',safe,overwrite=(folder/'auth-probe.json').exists())
        while True:
            status = {'state': 'CONNECTED', 'pid': os.getpid(), 'updated': time.time()}
            status['lark_executable'] = LarkSource(executable=cfg.get('lark_executable')).executable
            try:
                row = exchange(cfg, 'claim_publication')
                if row:
                    try:
                        detail = publish(cfg, row)
                    except Exception as error:
                        detail = {'state': 'NEEDS_REVIEW', 'issues': [str(error)[:250]]}
                    exchange(cfg, 'complete_publication', task=row['task'], digest=row['digest'], detail=detail)
                    status['last_publication'] = row['task']
                row = exchange(cfg, 'claim')
                if row:
                    prepare_materials(cfg, row)
                    status['last_task'] = row['task']
                if cfg.get('space_backend') == 'official_cli' or cfg.get('browser_backend_verified') is True:
                    # Opt-in only after a real CUA capability check; an installed
                    # CLI or a foreground Codex session does not prove this.
                    from evaluation_space_actor import execute
                    action = exchange(cfg, 'claim_space_action')
                    if action:
                        try:
                            detail = execute(cfg, action)
                        except Exception as error:
                            detail = {'state': 'NEEDS_REVIEW', 'reason': str(error)[:250],
                                      'real_browser_call_completed': False}
                        exchange(cfg, 'complete_space_action', action_id=action['id'], detail=detail)
                        status['last_space_action'] = {'id': action['id'], 'state': detail['state']}
            except Exception as error:
                status.update(state='NEEDS_REVIEW', reason=str(error)[:250])
            temp = folder/'status.tmp'
            temp.write_text(json.dumps(status), encoding='utf-8')
            temp.replace(folder/'status.json')
            if once or status['state'] == 'NEEDS_REVIEW':
                return status
            time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    result = main(args.config, once=args.once)
    if args.once:
        print(json.dumps(result))
