"""Admit the evaluation worker's board phase to the existing gateway FIFO."""
import json
from pathlib import Path
import sys
import time

from .common import digest, json_digest, load_json, save_json
from .evaluation import EvaluationRuntime

ENTRY = 'roadmodel-evaluation/v1'


def validate_admission_config(cfg):
    binding = cfg.get('shared_hardware_queue', {})
    if (binding.get('enabled') is not True or binding.get('action') != 'roadmodel_hardware'
            or binding.get('adapter_sha256') != digest(Path(__file__))):
        raise ValueError('新版编译链路尚未核验共享硬件队列绑定')


def services():
    base = Path('/mnt/data/bench-eval/service/gateway')
    sys.path.insert(0, str(base))
    from bench_gateway.config import load
    from bench_gateway.core import Store
    cfg = load(base)
    return cfg, Store(cfg['state_root'])


def bound_task(gateway_cfg, task_id, owner):
    if not task_id.startswith('rm-') or not task_id[3:].isalnum():
        raise ValueError('Invalid evaluation work order')
    rt = EvaluationRuntime(Path(gateway_cfg['state_root'])/'roadmodel/evaluation')
    cfg = load_json(rt.config_file)
    validate_admission_config(cfg)
    if cfg.get('bench_available') is not True:
        raise ValueError('台架执行未启用')
    task = rt.get(task_id, owner)
    if task['state'] != 'RUNNING' or not task.get('event_id'):
        raise ValueError('原卡未确认或工单没有运行')
    run = rt.root/'runs'/task_id
    workflow = load_json(run/'workflow.json')
    if workflow['workflow_sha256'] != json_digest({k:v for k,v in workflow.items() if k!='workflow_sha256'}):
        raise ValueError('冻结工作流发生变化')
    confirmed = task['inputs']
    inputs = {'model_delivery_url': confirmed['model_delivery_url'],
              'dataset_ref': confirmed['occ_dataset_url'] if 'occ_dataset_url' in confirmed else confirmed['dataset_url']}
    if 'occ_dataset_url' in confirmed:
        inputs['vision_dataset_ref'] = confirmed['vision_dataset_url']
    if workflow['run_id'] != task_id or workflow['inputs'] != inputs:
        raise ValueError('工作流与真实卡片输入不同')
    job = next(j for j in workflow['jobs'] if j['kind']=='compile_alignment')
    if not job['ready'] or job.get('native_adapter') != 'compile_full_scope':
        raise ValueError('编译材料或执行适配未就绪')
    return rt, cfg, task, workflow, job


def request_path(gateway_cfg, task_id):
    return Path(gateway_cfg['state_root'])/'roadmodel/evaluation/runs'/task_id/'hardware-request.json'


def submit_and_wait(plan, target, cfg):
    validate_admission_config(cfg)
    gateway_cfg, store = services()
    target = Path(target).resolve()
    task_id = target.parents[2].name
    rt, installed, task, workflow, job = bound_task(gateway_cfg, task_id, cfg['owner'])
    expected = (rt.root/'runs'/task_id/'effective/compile_alignment/execution').resolve()
    if target != expected or job['plan'] != plan:
        raise ValueError('硬件队列计划或输出目录不属于原工单')
    path = request_path(gateway_cfg, task_id)
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('Hardware request path contains a link')
    document = {'schema':ENTRY,'task':task_id,'owner':task['owner'],'session':task['session'],
                'event_id':task['event_id'],'inputs':task['inputs'],
                'workflow_sha256':workflow['workflow_sha256'],'plan':plan,'target':str(target)}
    if path.exists():
        if load_json(path) != document:raise ValueError('硬件队列冻结请求发生变化')
    else:save_json(path, document)
    payload = {'action':'roadmodel_hardware','automatic':True,'entry_runtime':ENTRY,
               'roadmodel_task':task_id,'request_file':str(path),'request_sha256':digest(path)}
    queued, _ = store.submit(task['owner'], task['session'], 'roadmodel-evaluation-board:'+task_id, payload)
    pointer = target.parent/'shared-queue.json'
    proof = {'hardware_job':queued['id'],'owner':task['owner'],'task':task_id,'plan_sha256':plan['plan_sha256']}
    if pointer.exists():
        if load_json(pointer) != proof:raise ValueError('硬件队列任务身份变化')
    else:save_json(pointer, proof)
    while True:
        current = store.job(task['owner'], queued['id'])
        if current['payload'] != payload:raise ValueError('共享硬件任务内容发生变化')
        if current['state'] == 'SUCCEEDED':
            result = current['result']['detail']
            if result.get('state') != 'succeeded' or result.get('plan_sha256') != plan['plan_sha256']:
                raise ValueError('共享硬件结果身份或完成收据不完整')
            return result
        if current['state'] not in ('QUEUED','RUNNING'):
            raise ValueError('共享硬件阶段'+current['state']+'，保留原收据，不自动重跑')
        detail={'stage':'等待共享台架队列' if current['state']=='QUEUED' else '共享台架正在执行',
                'hardware_job':queued['id'],'issues':[]}
        with rt.connect() as db:
            db.execute("update chains set state='RUNNING',detail=? where task=? and kind='compile_alignment'",
                       (json.dumps(detail),task_id))
        time.sleep(1)


def execute_queued(gateway_cfg, store, queued):
    # Called exclusively by the original gateway hardware worker.
    from bench_gateway.core import PreDispatchRejected
    entered = False
    try:
        current = store.job(queued['owner'], queued['id'])
        if current['state'] != 'RUNNING' or current['payload'] != queued['payload']:
            raise ValueError('共享硬件任务未被正式领取')
        payload = queued['payload']
        if payload.get('entry_runtime') != ENTRY:raise ValueError('Wrong entry runtime')
        path = request_path(gateway_cfg, payload['roadmodel_task'])
        if payload['request_file'] != str(path) or any(p.is_symlink() for p in (path,*path.parents)) or digest(path)!=payload['request_sha256']:
            raise ValueError('硬件请求文件身份不同')
        document = load_json(path)
        rt, cfg, task, workflow, job = bound_task(gateway_cfg, payload['roadmodel_task'], queued['owner'])
        expected = {'schema':ENTRY,'task':task['id'],'owner':task['owner'],'session':task['session'],
                    'event_id':task['event_id'],'inputs':task['inputs'],
                    'workflow_sha256':workflow['workflow_sha256'],'plan':job['plan'],
                    'target':str((rt.root/'runs'/task['id']/'effective/compile_alignment/execution').resolve())}
        if document != expected or queued['session'] != task['session']:
            raise ValueError('队列请求与原卡确认或冻结计划不同')
        if (Path(gateway_cfg['project_root'])/'runtime/service/drain.json').exists():
            return {'state':'RESOURCE_BUSY'}
        with store.connect() as db:
            db.execute("update jobs set result=?,updated=? where id=? and owner=? and state='RUNNING'",
                (json.dumps({'dispatch_started':True,'roadmodel_task':task['id']}),time.time(),queued['id'],queued['owner']))
        entered = True
        from .compile_scope import execute
        native = execute(job['plan'],document['target'],cfg['compile_profile'],bench_available=True)
        return {'detail':native,'project':'roadmodel','resource_phase_complete':True}
    except Exception:
        if entered:raise
        raise PreDispatchRejected('RoadModel板端派发前核验未通过',
            {'schema_version':1,'code':'ROADMODEL_PREDISPATCH_REJECTED','hardware_actions':'NOT_RUN','no_new_dispatch':True}) from None
