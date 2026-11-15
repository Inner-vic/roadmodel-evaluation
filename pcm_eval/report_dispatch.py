"""Start one reviewed, durable report coordinator per frozen execution.

This stage cannot dispatch inference. Unknown coordinator startup remains pending
for review and never creates another publisher automatically.
"""
from pathlib import Path
import os,subprocess,time
from .common import load_json,save_json,digest,utc_now

def start_report_pipeline(binding,*,spawn=None):
    spec=binding.get('report_pipeline')
    if not spec:return {'state':'NOT_BOUND'}
    config=Path(spec['config']);script=Path(spec['script']);receipt=Path(spec['dispatch_receipt'])
    for path,sha in ((config,spec['config_sha256']),(script,spec['script_sha256'])):
        if not path.is_absolute() or digest(path)!=sha:raise ValueError('Reviewed report coordinator changed')
    prepared=load_json(config)
    if prepared['report']['native_task_id']!=binding['native_task_id']:
        raise ValueError('Report pipeline belongs to another native task')
    expected=prepared['report']['frozen_requests'].get(binding['kind'])
    if expected!=binding['frozen_request']['sha256']:raise ValueError('Report pipeline has a different frozen input')
    if receipt.exists():
        old=load_json(receipt)
        if old.get('config_sha256')!=spec['config_sha256']:raise ValueError('Existing coordinator has different inputs')
        return old
    if not Path(spec['python']).is_absolute() or not Path(spec['cwd']).is_dir():raise ValueError('Prepared report executable and directory required')
    value={'state':'DISPATCHING','config_sha256':spec['config_sha256'],'started_at':utc_now()}
    claim=receipt.with_suffix('.claim.json')
    try:save_json(claim,{'pid':os.getpid(),'config_sha256':spec['config_sha256']})
    except FileExistsError:
        for _ in range(50):
            if receipt.exists():return load_json(receipt)
            time.sleep(.02)
        return {'state':'DISPATCH_ACK_PENDING','config_sha256':spec['config_sha256']}
    def atomic(value):
        tmp=receipt.with_suffix('.'+str(os.getpid())+'.tmp');save_json(tmp,value,overwrite=tmp.exists());os.replace(tmp,receipt)
    atomic(value)
    launcher=spawn or subprocess.Popen
    try:
        with receipt.with_suffix('.log').open('xb') as log:
            p=launcher([spec['python'],str(script),str(config)],cwd=spec['cwd'],env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value.update(state='STARTED',pid=p.pid);atomic(value)
    except Exception:
        value['state']='NEEDS_REVIEW';atomic(value);raise
    return value
