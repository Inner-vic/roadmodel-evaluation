"""Adopt the exact already-dispatched native evaluation; never replay it.

The binding is installed by the administrator, not accepted from the form. Its
frozen request, process identity and native progress are checked independently.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
from .common import load_json, digest, save_json
from .retained import checked_file

PHASE_LABELS={'PREPARING':'准备输入','STAGING_OCC_BATCH':'准备本批输入','RUNNING_OCC':'原生评测中',
 'RESUMING_RESULT_COLLECTION':'续收原执行结果','WAITING_GPU_MEMORY':'等待GPU显存','OCC_SUCCEEDED':'OCC评测完成',
 'WAITING_OCC_AND_DATA':'等待OCC完成','WAITING_PREREQUISITE_REVIEW':'前置任务待核验','RUNNING':'原生评测中',
 'WAITING_BENCH':'等待台架空闲','WAITING_GT':'等待GT准备','FINALIZING':'汇总原生结果','SUCCEEDED':'原生评测完成',
 'NEEDS_REVIEW':'待核验','NOT_REQUESTED':'本范围未启用'}
def phase(state):return PHASE_LABELS.get(state,'处理中')


def verify(root, record, kind, version, dataset_id):
    binding_path=checked_file(root,record)
    binding=load_json(binding_path)
    if (binding.get('schema')!='roadmodel.active-execution/v1' or binding.get('kind')!=kind
            or binding.get('version')!=version or binding.get('dataset_id')!=dataset_id):
        raise ValueError('Active execution binding identity differs from submitted materials')
    request_path=Path(binding['frozen_request']['path'])
    if not request_path.is_absolute() or digest(request_path)!=binding['frozen_request']['sha256']:
        raise ValueError('Frozen native request changed')
    request=load_json(request_path)
    if request['task_id']!=binding['native_task_id'] or request['model_version']!=version:
        raise ValueError('Native task version or identity differs')
    if request['model_delivery_url']!=binding['model_delivery_url'] or request['dataset_version']!='26-10-20-1':
        raise ValueError('Native task model or data identity differs')
    frames=request['frames']
    if len(frames)!=binding['expected_frames'] or len(set(map(tuple,frames)))!=len(frames):
        raise ValueError('Frozen native frame scope differs')
    return binding


def process_alive(pid, script):
    try:
        os.kill(int(pid),0)
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return script in command
    except (OSError,ValueError):return False


def poll(binding):
    kind=binding['kind'];root=Path(binding['execution_root'])
    progress_path=root/('model/progress.json' if kind=='model_accuracy' else 'compile/progress.json')
    if not progress_path.exists():
        dispatch=load_json(root/('workstation-dispatch.json' if kind=='model_accuracy' else 'compile-dispatch.json'))
        programs=binding.get('launch',{}).get('programs',[])
        alive=bool(programs) and all(process_alive(dispatch.get(p['role']+'_pid'),p['script']) for p in programs)
        return {'state':'RUNNING' if alive else 'NEEDS_REVIEW','mode':'active_native_execution','binding':binding,
                'native_task_id':binding['native_task_id'],'completed_frames':0,'expected_frames':binding['expected_frames'],
                'native_finished':False,'report_url':None,'repeated_completed_frames':0,
                'issues':['原生控制器已派发，正在启动预检。' if alive else '启动收据存在，但尚未确认预检进程；不会重复派发。'],
                'proof':{'dispatch_state':dispatch['state'],'native_controllers_alive':alive}}
    if kind=='model_accuracy':
        progress=load_json(root/'model/progress.json')
        vision_required=binding.get('vision_required',True)
        vision=load_json(root/'vision/progress.json') if vision_required else {'state':'NOT_REQUESTED','expected_frames':0}
        dispatch=load_json(root/'workstation-dispatch.json')
        completed=progress['completed_OCC_frames'];expected=binding['expected_frames']
        programs={p['role']:p['script'] for p in binding.get('launch',{}).get('programs',[])}
        alive=process_alive(dispatch['model_controller_pid'],programs.get('model_controller','formal_model_controller.py'))
        vision_alive=process_alive(dispatch['vision_controller_pid'],programs.get('vision_controller','formal_vision_controller.py')) if vision_required else False
        done=progress['state']=='OCC_SUCCEEDED' and (not vision_required or vision['state']=='SUCCEEDED')
        state='NEEDS_REVIEW' if progress['state']=='NEEDS_REVIEW' or vision['state']=='NEEDS_REVIEW' else 'RUNNING'
        if not done and not alive and progress['state']!='OCC_SUCCEEDED':state='NEEDS_REVIEW'
        if vision_required and progress['state']=='OCC_SUCCEEDED' and not vision_alive and vision['state']!='SUCCEEDED':state='NEEDS_REVIEW'
        lines=[f'OCC：{completed}/{expected} 帧；'+phase(progress['state'])]
        if vision_required:lines.append(f"视觉：{vision.get('completed_frames',0)}/{vision['expected_frames']} 帧；"+phase(vision['state']))
        proof={'model_progress':progress,'vision_progress':vision,'model_controller_alive':alive,'vision_controller_alive':vision_alive}
    else:
        progress=load_json(root/'compile/progress.json')
        completed=progress['completed_frames'];expected=binding['expected_frames']
        programs={p['role']:p['script'] for p in binding.get('launch',{}).get('programs',[])}
        alive=process_alive(progress['pid'],programs.get('compile_controller','formal_compile_remote.py'))
        done=progress['state']=='SUCCEEDED'
        state='NEEDS_REVIEW' if progress['state']=='NEEDS_REVIEW' or (not done and not alive) else 'RUNNING'
        lines=[f'板端与ONNX及GT：{completed}/{expected} 帧；'+phase(progress['state'])]
        proof={'compile_progress':progress,'compile_controller_alive':alive}
    if type(completed)!=int or not 0<=completed<=expected:
        raise ValueError('Invalid native completion count')
    if done:
        lines.append('原生评测已结束，结果收集、报告生成和飞书回读仍需完成。')
    # A report URL is accepted only from a new, scope-bound full delivery receipt.
    delivery_path=Path(binding['delivery_receipt'])
    report_url=None
    if done and delivery_path.exists():
        delivery=load_json(delivery_path)
        request_hash=delivery.get('frozen_requests',{}).get(kind,delivery.get('frozen_request_sha256'))
        if (delivery.get('state')=='VERIFIED' and delivery.get('native_task_id')==binding['native_task_id']
                and request_hash==binding['frozen_request']['sha256']
                and delivery.get('scope')==binding.get('report_scope','full') and delivery.get('five_tables_match') is True):
            report_url=delivery['report_url'];state='SUCCEEDED'
    if done and not report_url:
        journal=delivery_path.parent/'journal.json'
        failed=Path(binding['report_coordinator_failure']) if binding.get('report_coordinator_failure') else None
        if (journal.exists() and load_json(journal).get('state')=='NEEDS_REVIEW') or (failed and failed.exists()):
            state='NEEDS_REVIEW';lines.append('真实评测已结束，报告生成或回填待核验；不会重新评测。')
    return {'state':state,'mode':'active_native_execution','binding':binding,'native_task_id':binding['native_task_id'],
            'completed_frames':completed,'expected_frames':expected,'native_finished':done,
            'report_url':report_url,'issues':lines,'proof':proof,
            'inference_dispatch':'adopt_existing_exact_input_task','repeated_completed_frames':0}


def execute(job,target):
    binding=verify(job['material_root'],job['active_execution'],job['kind'],job['version'],job['dataset_id'])
    from .native_dispatch import start_or_attach
    dispatch=start_or_attach(binding)
    # Native workers write progress asynchronously. Preserve the dispatch identity
    # while the first preflight starts; a second submission must never launch it again.
    if dispatch['mode']=='new_native_dispatch':
        detail={'state':'RUNNING','mode':'active_native_execution','binding':binding,
                'native_task_id':binding['native_task_id'],'completed_frames':0,
                'expected_frames':binding['expected_frames'],'native_finished':False,
                'issues':['已派发原生评测控制器，正在执行输入和环境预检。'],
                'dispatch':dispatch,'repeated_completed_frames':0,'report_url':None}
    else:
        detail=poll(binding)
    from .report_dispatch import start_report_pipeline
    detail['report_coordinator']=start_report_pipeline(binding)
    target.mkdir(parents=True,exist_ok=False)
    save_json(target/'native-attachment.json',detail)
    return detail
