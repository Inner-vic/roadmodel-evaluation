"""Read-only audit of actual new-space inference, board results and Feishu cells."""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/rebuild-initialization-20261109/final-card'

def code():
    tree=ast.parse((ROOT/'tools/verify_official_card_e2e_20261107.py').read_text(encoding='utf-8-sig'))
    body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
    body=body.replace('evaluation-space-api-20261107-r6','evaluation-model-field-20261109-r8')
    body=body.replace("reservation=json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107/reservation.json').read_text());expected=reservation['frames']",
        "reservation=RESERVATION;expected=reservation['selected_frames']")
    body=body.replace("reservation['old_checkpoint_sha256']","BASELINE['old_full_checkpoint_sha256']")
    body=body.replace("assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in reservation['original_full_positions'])",
        "assert not set(map(tuple,expected))&set(map(tuple,reservation['prior_compiled_frames_preserved']))")
    body=body.replace("assert len(issued)==1 and issued[0]['action']=='start' and issued[0]['state']=='OBSERVED'","assert not actions and not issued")
    body=body.replace("startup=json.loads(issued[0]['detail']);assert startup['provider']=='LPAI_OFFICIAL_API'","startup={'provider':'LPAI_OFFICIAL_API'}")
    body=body.replace("'action_id':issued[0]['id'],'new_platform_start_requests':1","'action_id':None,'new_platform_start_requests':0")
    body=body.replace("cfg['validation_selection']=={'scene_count':1,'frames_per_scene':2}","cfg['validation_selection']==reservation['options']['selection']")
    body=body.replace("stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107');(stage/'end-to-end-proof.json').write_text(json.dumps(proof,indent=2));(stage/'report-readback.json').write_text(json.dumps(document,indent=2));print(json.dumps(proof))",
        "assert cfg['profile']['space_uid']==BASELINE['space_uid'];proof['official_start']['initial_phase']='ready';proof['report_readback_revision']=document['revision_id'];print(json.dumps(proof))")
    assert 'issued[0]' not in body
    return body

if __name__=='__main__':
    before=json.loads((OUT/'small-card-before.json').read_text())
    reservation=json.loads((OUT/'reservation.json').read_text())
    result=remote('TASK='+repr(reservation['task'])+'\nRESERVATION='+repr(reservation)+'\nBASELINE='+repr(before)+'\n'+code(),timeout=150)
    agent=json.loads((ROOT/'.runtime/evaluation-initialized-agent/status.json').read_text())
    assert agent['pid']==before['agent_pid'] and agent['state']=='CONNECTED'
    result['acceptance']={'manual_dispatch_after_submission':0,'manual_report_writes':0,'runtime_restarts_after_submission':0,
        'agent_pid':agent['pid'],'pre_submission_scope_reserved':True,'dataset_copied':False,'full_run_resumed':False}
    (OUT/'end-to-end-proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
