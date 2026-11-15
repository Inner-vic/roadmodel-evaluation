"""PC-side authenticated material reader; no credentials leave this computer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import json_digest, load_json
from pcm_eval.delivery_workflow import prepare_delivery
from pcm_eval.planning import ssh_command

RPC = '''import json,sys
sys.path.insert(0,sys.argv[1])
if len(sys.argv)>3 and sys.argv[3]=='evaluation':
 from pcm_eval.evaluation import EvaluationRuntime
 r=EvaluationRuntime(sys.argv[2])
else:
 from pcm_eval.delivery_demo import DemoRuntime
 r=DemoRuntime(sys.argv[2])
p=json.load(sys.stdin)
if p['action']=='claim':v=r.claim_materials(p['agent'],p['owner'])
else:r.complete_materials(p['agent'],p['task'],p['digest'],p['detail']);v={'state':'ACCEPTED'}
print(json.dumps(v))
'''


def rpc(cfg, payload):
    protocol = cfg.get('rpc_code', RPC)
    command = ssh_command('ubuntu-workstation', [cfg['python'], '-c', protocol, cfg['release'], cfg['state_root'], cfg.get('queue_kind', 'demo')])
    if os.name == 'nt':
        command[0] = str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'System32/OpenSSH/ssh.exe')
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    p = subprocess.run(command, input=json.dumps(payload).encode('utf-8'), capture_output=True,
                       timeout=cfg.get('rpc_timeout', 45), **options)
    if p.returncode:
        raise ValueError('Demo工作站队列连接暂不可用')
    return json.loads(p.stdout)


def process(cfg, row):
    request = json.loads(row['request'])
    if json_digest(request) != row['digest']:
        raise ValueError('Confirmed material request changed')
    try:
        workflow = prepare_delivery(request['model_delivery_url'], request['dataset_ref'],
                                    run_id=row['task'], validation=True)
        for job in workflow['jobs']:
            for key in ('binding_file', 'gt_preparation_file'):
                if job.get(key):
                    relative = Path(job[key]).resolve().relative_to(ROOT.resolve())
                    job[key] = cfg['release']+'/'+relative.as_posix()
        workflow.pop('workflow_sha256')
        workflow['workflow_sha256'] = json_digest(workflow)
        detail = {'workflow': workflow}
    except Exception as error:
        detail = {'error': '材料核验未完成：'+str(error)[:350]}
    rpc(cfg, {'action': 'complete', 'agent': cfg['agent_id'], 'task': row['task'],
              'digest': row['digest'], 'detail': detail})


def main(config, once=False):
    cfg = load_json(config)
    log = Path(config).parent/'agent-heartbeat.json'
    if cfg.get('paused'):
        state = {'pid': os.getpid(), 'updated': time.time(), 'state': 'PAUSED'}
        log.write_text(json.dumps(state), encoding='utf-8')
        return state
    while True:
        state = {'pid': os.getpid(), 'updated': time.time(), 'state': 'CONNECTED'}
        try:
            row = rpc(cfg, {'action': 'claim', 'agent': cfg['agent_id'], 'owner': cfg['owner']})
            if row:
                process(cfg, row)
                state['last_task'] = row['task']
        except Exception as error:
            state.update(state='NEEDS_REVIEW', error=str(error)[:200])
        temp = log.with_suffix('.tmp')
        temp.write_text(json.dumps(state), encoding='utf-8')
        temp.replace(log)
        if once or state['state'] == 'NEEDS_REVIEW':
            return state
        time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    result = main(args.config, args.once)
    if args.once:
        print(json.dumps(result))
