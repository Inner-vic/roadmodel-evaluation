"""Desktop preparation agent; all work orders, journals and results stay on WS."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcm_eval.dataset_mount import mount_plan, resolve_dataset_input
from pcm_eval.dataset_preparation import advance
from pcm_eval.lpai import request as lpai_request
from pcm_eval.space_browser import run_browser_action


def remote(cfg,packet):
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',cfg['workstation_alias'],
          shlex.join([cfg['remote_python'],cfg['remote_protocol'],cfg['queue_root']])]
    p=subprocess.run(argv,input=json.dumps(packet).encode(),capture_output=True,timeout=130,
                     **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
    try:value=json.loads(p.stdout)
    except ValueError:raise ValueError('Workstation preparation connection unavailable')
    if p.returncode or not value.get('ok'):raise ValueError(value.get('reason','Workstation preparation protocol failed'))
    return value['data']


def process(cfg,client,job,*,deadline_seconds=600):
    req=job['request'];profile=req['profile'];detail=job.get('detail',{});journal=detail.get('journal',{})
    def rpc(op,**fields):return remote(cfg,{'op':op,'client':client,'id':job['id'],'lease':job['lease'],**fields})
    def save(value):rpc('update',state='RUNNING',detail={'stage':value.get('stage','准备LPAI环境'),'journal':value})
    query=lambda args:lpai_request(args,cli=cfg['lpai_cli'])
    initial_space=query(['devspaces','get','--namespace',profile['namespace'],'--name',profile['space_name']])
    if initial_space['metadata']['uid']!=profile['space_uid']:raise ValueError('开发空间身份与本人工单运行环境不一致')
    asset=resolve_dataset_input(req['inputs']['dataset_ref'],query,initial_space)
    started=time.monotonic()
    while time.monotonic()-started<deadline_seconds:
        space=query(['devspaces','get','--namespace',profile['namespace'],'--name',profile['space_name']])
        if space['metadata']['uid']!=profile['space_uid'] or space['spec']['clusterName']!=profile['cluster']:
            raise ValueError('开发空间身份或集群与本人运行环境不一致')
        plan=mount_plan(asset,space)
        latest=[journal]
        def browser(action):
            result=run_browser_action(action,space,asset,schema_path=ROOT/'schemas/space-preparation-result.schema.json',cwd=ROOT,
                                      binary=cfg.get('codex_binary'),timeout=180)
            rpc('update',state='RUNNING',detail={'stage':'核验LPAI网页操作结果','journal':latest[0],'browser_action':result})
            if result['state'] not in ('ACTION_RETURNED','OBSERVED'):
                raise ValueError('后台浏览器能力未通过：'+result.get('reply',{}).get('outcome',result.get('reason','UNKNOWN')))
        actions={'assert_idle':lambda s:rpc('idle',plan=plan)['idle'],
                 'stop':lambda s:browser('stop'),'start':lambda s:browser('start'),
                 'configure_and_start':lambda s,a:browser('configure_and_start'),
                 'probe':lambda p:rpc('probe',plan=p)}
        if not cfg.get('browser_backend_verified',False):
            for key in ('stop','start','configure_and_start'):actions.pop(key)
        # Save receives the new intent before the UI action. Don't close over an
        # older journal when recording the browser result.
        def persist(value):latest[0]=value;save(value)
        journal=advance(asset,space,journal,save=persist,actions=actions)
        if journal['state']=='DATA_READY':
            receipt=journal['mount_receipt'];source=receipt.pop('dataset_source',None)
            if not source:raise ValueError('空间内未返回本次冻结帧清单')
            rpc('update',state='DATA_READY',detail={'stage':'LPAI挂载及验证索引已就绪','journal':journal,'dataset_source':source})
            return {'state':'DATA_READY','job_id':job['id'],'space_uid':profile['space_uid'],
                    'split_counts':source['mount_receipt']['split_counts'],'new_inference_frames':0}
        if journal['state']=='NEEDS_REVIEW':
            rpc('update',state='NEEDS_REVIEW',detail={'stage':'环境准备待核验','reason':journal.get('reason','原平台动作需核验'),'journal':journal})
            return {'state':'NEEDS_REVIEW','job_id':job['id'],'reason':journal.get('reason')}
        time.sleep(5)
    rpc('update',state='NEEDS_REVIEW',detail={'stage':'环境准备待核验','reason':'空间准备等待超时，保留原动作收据，不重复操作','journal':journal})
    return {'state':'NEEDS_REVIEW','job_id':job['id'],'reason':'preparation_wait_timeout'}


@contextmanager
def own_lock(folder):
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'process.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:yield
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    cache=Path(cfg['connection_index'])
    with own_lock(cache):
        if cfg.get('paused'):
            (cache/'status.json').write_text(json.dumps({'state':'PAUSED','pid':os.getpid(),'updated':time.time()}),encoding='utf-8')
            return
        index=cache/'identity.json'
        if not index.exists():index.write_text(json.dumps({'client':uuid.uuid4().hex}))
        client=json.loads(index.read_text())['client']
        while True:
            state='CONNECTED';job=None
            try:
                job=remote(cfg,{'op':'claim','client':client})
                if job:
                    result=process(cfg,client,job)
                    if args.once:print(json.dumps(result,ensure_ascii=False))
            except Exception as error:
                state=type(error).__name__
                if job:
                    try:remote(cfg,{'op':'update','client':client,'id':job['id'],'lease':job['lease'],'state':'NEEDS_REVIEW',
                                   'detail':{'reason':str(error)[:250],'stage':'环境准备待核验'}})
                    except Exception:pass
                if args.once:print(json.dumps({'state':'NEEDS_REVIEW','error_type':type(error).__name__,'reason':str(error)[:250]},ensure_ascii=False))
            # Only process/connection indices on PC; no user inputs or journals.
            tmp=cache/'status.tmp';tmp.write_text(json.dumps({'state':state,'pid':os.getpid(),'updated':time.time()}));os.replace(tmp,cache/'status.json')
            if args.once or state!='CONNECTED':return
            time.sleep(10)


if __name__=='__main__':main()
