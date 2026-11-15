"""Native evaluation -> independent atomic reports -> same Feishu report.

Used by live first-dispatch validation and the existing full-run completion
coordinator. It never performs inference, changes a bench lock or restarts work.
"""
from copy import deepcopy
from pathlib import Path
import time
from .common import load_json,save_json,digest,utc_now
from .active_execution import verify,poll
from .model_aggregate import aggregate_occ
from .model_report import build_model_report,render_model_xml
from .compile_aggregate import aggregate_compile
from .business_layout import render_model_business,compile_business_sections
from .compile_report import render_sections
from .reference_report import organize,render
from .feishu_update import update_report


def generate_and_publish(manifest):
    root=Path(manifest['output']);root.mkdir(parents=True,exist_ok=True)
    verification=root/'publication/verification.json'
    if verification.exists():return load_json(verification)
    generation=root/'generation.json'
    if generation.exists():
        saved=load_json(generation)
        if saved['state']!='GENERATED':raise ValueError('Report generation previously interrupted; reconcile existing files')
        if saved['manifest_sha256']!=manifest['manifest_sha256']:raise ValueError('Generation manifest changed')
    else:
        save_json(generation,{'state':'GENERATING','manifest_sha256':manifest['manifest_sha256'],'started_at':utc_now()})
        baseline=load_json(manifest['baseline_model'])
        model_frames=load_json(manifest['model_request'])['frames']
        model=aggregate_occ(manifest['model_results'],model_frames,root/'model',run_id=manifest['native_task_id']+'-occ')
        binding=deepcopy(baseline['binding']);binding['dataset']['selected_frames']=model_frames;binding['dataset']['version']='26-10-20-1'
        if manifest.get('vision_result'):
            vision=load_json(manifest['vision_result'])
            expected_vision=manifest['vision_frames']
            if vision['provenance']['native_run']['selected_frames']!=expected_vision:raise ValueError('Full vision scope differs')
            heads=[vision]
        else:
            # Historical head results remain explicitly scoped; not called new/full inference.
            heads=[v for k,v in baseline.get('head_results',{}).items() if k=='vision']
        if baseline.get('head_results',{}).get('trajectory'):heads.append(baseline['head_results']['trajectory'])
        model_report=build_model_report(model,binding=binding,head_results=heads)
        save_json(root/'model/report.json',model_report)
        (root/'model/report.xml').write_text(render_model_business(model_report,render_model_xml(model_report,json_attachment='report.json')),encoding='utf8')
        compile_frames=load_json(manifest['compile_request'])['frames']
        compiler=aggregate_compile(manifest['compile_frames'],compile_frames,root/'compile',request=manifest['compile_execution_request'],
                 run_id=manifest['native_task_id']+'-compile',checkpoint=manifest.get('compile_checkpoint'))
        sections=compile_business_sections(compiler,render_sections(compiler))
        (root/'compile/report.xml').write_text('<title>编译精度单次报告</title>'+''.join(s['xml'] for s in sections),encoding='utf8')
        data=organize(model_report,compiler,load_json(manifest['reference']))
        data['evaluation_scope']=manifest['scope']
        data['atomic_results']={'model':model_report,'compile':compiler}
        data['generation'].update(native_task_id=manifest['native_task_id'],evaluation_scope=manifest['scope'],
              historical_vision_retained=not bool(manifest.get('vision_result')),
              new_model_inference_frames=manifest.get('new_model_frames',0),new_compile_inference_frames=manifest.get('new_compile_frames',0))
        save_json(root/'data.json',data);(root/'report.xml').write_text(render(data,'data.json'),encoding='utf8')
        save_json(generation,{'state':'GENERATED','manifest_sha256':manifest['manifest_sha256'],'five_tables':len(data['tables']),
                 'model_frames':len(model_frames),'compile_frames':len(compile_frames),'generated_at':utc_now()},overwrite=True)
    receipt=update_report(root/'report.xml',root/'data.json',root/'publication',document_id=manifest['document_id'],
           native_task_id=manifest['native_task_id'],frozen_requests=manifest['frozen_requests'],scope=manifest['scope'])
    return receipt


def wait_and_publish(config_file):
    config=load_json(config_file);output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    while True:
        try:states=[poll(b) for b in config['bindings']]
        except (FileNotFoundError,ValueError):
            # Native progress writes can briefly be incomplete; no new work is launched.
            time.sleep(2);continue
        if all(s['native_finished'] for s in states):break
        save_json(output/'coordinator.json',{'state':'WAITING_NATIVE_EXECUTION','updated_at':utc_now(),
                  'chains':[{'kind':s['binding']['kind'],'state':s['state'],'completed_frames':s['completed_frames']} for s in states]},overwrite=(output/'coordinator.json').exists())
        time.sleep(20)
    manifest=deepcopy(config['report'])
    if config['scope']=='full':
        task=Path(config['model_task_root']);r=load_json(task/'request.json');attempts=load_json(task/'model-attempts.json') if (task/'model-attempts.json').exists() else {}
        manifest['model_results']=[str(task/'model'/f"batch-{i:04d}-{attempts.get(str(i),'a02')}"/'normalized-result.json') for i,_ in enumerate(range(0,len(r['frames']),100))]
        manifest['compile_frames']=[str(Path(config['compile_task_root'])/'compile/frames'/f'frame_{i:06d}') for i in range(len(r['frames']))]
        manifest['compile_checkpoint']=str(Path(config['compile_task_root'])/'compile/checkpoint.npz')
        manifest['vision_result']=str(task/'vision/normalized-result.json');manifest['vision_frames']=r['vision']['frames']
    manifest['manifest_sha256']=digest(Path(config_file))
    result=generate_and_publish(manifest)
    if config.get('runtime_root'):
        from .runtime import Runtime
        Runtime(config['runtime_root'],catalog_file=config.get('runtime_catalog')).refresh_active()
    save_json(output/'coordinator.json',{'state':'VERIFIED','receipt':result,'finished_at':utc_now()},overwrite=(output/'coordinator.json').exists())
    return result
