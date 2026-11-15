"""Real official lifecycle only: no card callback simulation or model inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import load_json, save_json, utc_now
from pcm_eval.space_api import OfficialSpaces
from pcm_eval.dataset_mount import resolve_dataset, mount_plan
from pcm_eval.lpai import request
from deploy_environment_flow import remote

CLI = 'C:/Users/developer/AppData/Local/bin/lpai-agent.exe'
DATASET = 'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1'
OUT = ROOT/'evidence/sdk-space-20261107'


def configuration():
    cfg = load_json(ROOT/'.runtime/evaluation-environment-desktop-20261106-r3/configs/agent.json')
    profile = deepcopy(cfg['profile'])
    profile['space_uid'] = load_json(OUT/'official-read.json')['metadata']['uid']
    profile['lifecycle_provider'] = 'official_cli'
    return profile


def check_tasks():
    return remote('''import sqlite3,json
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3') as db:
 active=db.execute("select id,state from tasks where state in ('RUNNING','PREPARING','READY','QUEUED','WAITING_REPORT')").fetchall()
assert not active,'Evaluation work is active'
print(json.dumps({'active_evaluation_tasks':0}))
''', timeout=30)


def idle(profile, space):
    import ast
    tree = ast.parse((ROOT/'tools/finish_frame_binding_20261107.py').read_text(encoding='utf-8'))
    code = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'idle_code' for t in n.targets))
    result = space_call(profile, space, code)
    save_json(OUT/'live-idle.json', result, overwrite=(OUT/'live-idle.json').exists())
    return result['idle']


def space_call(profile, space, code):
    payload = {'code':code, 'port':space['runtime_access']['port']}
    return remote('''import pathlib,json,sys,shlex,subprocess
cfg=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5/configs/evaluation.json').read_text());p=cfg['profile'];v=json.load(sys.stdin)
cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=accept-new','-p',str(v['port']),'-i',p['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',v['code']])]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=50);assert r.returncode==0,'Authorized space probe failed'
print(r.stdout)
''', json.dumps(payload).encode(), timeout=65)


def main(action):
    profile = configuration();client = OfficialSpaces(profile, cli=CLI)
    space = client.read()
    if action == 'observe':
        result = {'state':'OBSERVED','observation':client.observation(space)}
        if space['status']['phase'] == 'ready':
            asset = resolve_dataset(DATASET, lambda args: request(args, cli=CLI))
            plan = mount_plan(asset, space)
            code = (ROOT/'pcm_eval/dataset_mount.py').read_text(encoding='utf-8')
            code += '\nplan='+repr(plan)+'\nresult=probe_mounted_index(plan,include_manifest=True)\nprint(json.dumps({k:result[k] for k in ("state","asset_binding_sha256","space_uid","mount_path","index_sha256","split_counts","execution_location","workstation_data_used")}))'
            result['mount_probe'] = space_call(profile, space, code)
    else:
        check_tasks()
        context = {'profile':profile,'dataset_ref':DATASET,'intent_key':'official-api-live-validation-20261107'}
        if action == 'start':context['asset'] = resolve_dataset(DATASET, lambda args: request(args, cli=CLI))
        result = client.execute(action, context, ROOT/'.runtime/official-space-live-20261107'/action,
                                task='official-lifecycle-validation-20261107', idle=lambda s:idle(profile,s))
    save_json(OUT/'observations'/(utc_now().replace(':','-')+'-'+action+'.json'), result)
    save_json(OUT/('live-'+action+'.json'), result, overwrite=(OUT/('live-'+action+'.json')).exists())
    print(json.dumps({'action':action,'state':result['state'],'new_requests':result.get('new_requests',0),
                      'phase':result.get('observation',{}).get('space',{}).get('status',{}).get('phase'),
                      'mount_probe':result.get('mount_probe'),'reason':result.get('reason')},ensure_ascii=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('start','observe','stop'))
    main(parser.parse_args().action)
