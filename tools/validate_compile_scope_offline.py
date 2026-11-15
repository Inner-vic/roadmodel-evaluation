"""Import the actual native controller and ORT session, with all network denied."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys


def main(release,output):
    release=Path(release);sys.path[:0]=[str(release),str(release/'tools')]
    from pcm_eval.common import load_json
    from pcm_eval.compile_scope import prepare
    from pcm_eval.gt_runtime import validated_settings
    cfg=load_json(release/'configs/evaluation.json')
    assert cfg['bench_available'] is False
    old=Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-997e5dda6e8546a4a935/workflow.json')
    workflow=load_json(old);job=next(j for j in workflow['jobs'] if j['kind']=='compile_alignment')
    prepared=prepare(job['plan'],Path(output)/'native-scope',cfg['compile_profile'])
    assert prepared['expected_frames']==2 and prepared['original_GT_body_unchanged']
    def denied(*args,**kwargs):raise AssertionError('No network allowed in offline validator')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(prepared['environment']);code=Path(prepared['controller']).parent
    sys.path[:0]=[str(code)]
    spec=importlib.util.spec_from_file_location('offline_controller',prepared['controller'])
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    assert controller.TOTAL==2 and controller.FRAMES==job['plan']['frames']
    settings=validated_settings(code/'gt-runtime.json')
    engine=controller.Engine()
    try:
        result={'state':'OFFLINE_VERIFIED','frames':controller.FRAMES,'expected_frames':controller.TOTAL,
                'subnets':len(engine.graph['stages']),'edges':len(engine.graph['edges']),
                'external_inputs':len(engine.graph['external_inputs']),'public_outputs':len(engine.public),
                'all_subnet_outputs':sum(len(s['outputs']) for s in engine.graph['stages']),
                'same_native_GT_body':True,'GT_backend':settings['backend'],'GT_workers':settings['workers'],
                'ORT_session_initialized':True,'ORT_provider':engine.session.get_providers(),
                'native_material_binding_checked':True,'bench_contacted':False,
                'ONNX_inference_frames':0,'board_inference_frames':0,'dataset_copied':False}
        assert (result['subnets'],result['edges'],result['external_inputs'],result['public_outputs'],result['all_subnet_outputs'])==(5,32,11,18,49)
        print(json.dumps(result))
    finally:engine.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.release,args.output)
