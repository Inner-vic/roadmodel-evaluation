"""Dispatch reviewed native controllers once, or adopt their existing execution.

Only an administrator-frozen binding supplies programs and environment. A crash
between the dispatch receipt and child acknowledgement is unknown, never retried.
"""
import json
import os
from pathlib import Path
import subprocess
from .common import digest,load_json,save_json,utc_now


def start_or_attach(binding,*,spawn=None):
    root=Path(binding['execution_root']).resolve()
    progress=root/('model/progress.json' if binding['kind']=='model_accuracy' else 'compile/progress.json')
    if progress.exists():return {'mode':'adopt_existing','new_dispatches':0}
    spec=binding.get('launch')
    if not spec:raise ValueError('No reviewed launch binding; inspect existing native execution')
    prepared=load_json(Path(spec['preparation_receipt']['path']))
    if digest(Path(spec['preparation_receipt']['path']))!=spec['preparation_receipt']['sha256']:
        raise ValueError('Preparation receipt changed')
    if (prepared.get('state')!='VERIFIED' or prepared.get('native_task_id')!=binding['native_task_id']
            or prepared.get('frozen_request_sha256')!=binding['frozen_request']['sha256']):
        raise ValueError('Native preparation does not match the frozen task')
    programs=spec['programs']
    expected_roles=({'model_controller','vision_controller'} if binding.get('vision_required',True) else {'model_controller'}) if binding['kind']=='model_accuracy' else {'compile_controller'}
    if {p['role'] for p in programs}!=expected_roles or len(programs)!=len(expected_roles):
        raise ValueError('Native controller roles are incomplete or duplicated')
    for program in programs:
        script=Path(program['script'])
        if not script.is_absolute() or digest(script)!=program['sha256']:
            raise ValueError('Reviewed native controller changed')
        if not Path(program['python']).is_absolute() or not Path(program['cwd']).is_dir():
            raise ValueError('Native executable and working directory must be prepared')
        if not isinstance(program.get('env',{}),dict) or any(not isinstance(v,str) for v in program.get('env',{}).values()):
            raise ValueError('Native environment must be frozen strings')
    root.mkdir(parents=True,exist_ok=True)
    path=root/('workstation-dispatch.json' if binding['kind']=='model_accuracy' else 'compile-dispatch.json')
    receipt={'state':'DISPATCHING','native_task_id':binding['native_task_id'],'started_at':utc_now(),
             'frozen_request_sha256':binding['frozen_request']['sha256'],'children':[]}
    # Exclusive creation arbitrates concurrent submissions. Never replace ambiguity.
    save_json(path,receipt)
    launcher=spawn or subprocess.Popen
    try:
        for program in programs:
            log=root/(program['role']+'-dispatch.log')
            with log.open('xb') as stream:
                child=launcher([program['python'],program['script']],cwd=program['cwd'],
                               env={**os.environ,**program.get('env',{})},stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            receipt[program['role']+'_pid']=child.pid
            receipt['children'].append({'role':program['role'],'pid':child.pid,'program_sha256':program['sha256']})
            save_json(path,receipt,overwrite=True)
        receipt['state']='STARTED';save_json(path,receipt,overwrite=True)
    except Exception:
        receipt['state']='NEEDS_REVIEW';save_json(path,receipt,overwrite=True)
        raise
    return {'mode':'new_native_dispatch','new_dispatches':len(programs),'receipt':str(path)}
