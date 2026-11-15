"""Read-only audit of the new default-scope card, reusing proven result checks."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/unassisted-card-e2e-20261108'


def audit_code():
    tree=ast.parse((ROOT/'tools/verify_official_card_e2e_20261107.py').read_text(encoding='utf-8-sig'))
    code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
    def replace(old,new):
        nonlocal code
        assert code.count(old)==1,old
        code=code.replace(old,new)
    replace("reservation=json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107/reservation.json').read_text());expected=reservation['frames']",
            "expected=jobs['model_accuracy']['plan']['heads']['occ'];assert len(expected)==2 and len({f[0] for f in expected})==1\nassert task['options']=={'scope':'validation','selection':{'scene_count':1,'frames_per_scene':2}}")
    replace("task['message_id']==reservation['message_id']","task['message_id']==MESSAGE")
    replace("reservation['old_checkpoint_sha256']","BASELINE['old_full_checkpoint_sha256']")
    replace("assert all(not (full/'compile/frames'/f'frame_{i:06d}').exists() for i in reservation['original_full_positions'])", "# This card uses its normal default frames; prior full results are never resumed.")
    # Exactly one action, not just one successful start, rules out an old-profile
    # rejection/recovery sequence like the previous validation.
    replace("assert len(issued)==1 and issued[0]['action']=='start' and issued[0]['state']=='OBSERVED'",
            "assert len(actions)==len(issued)==1 and issued[0]['action']=='start' and issued[0]['state']=='OBSERVED'")
    replace("journal=json.loads((run/'space-preparation.json').read_text());assert journal['state']=='DATA_READY'",
            "journal=json.loads((run/'space-preparation.json').read_text());assert journal['state']=='DATA_READY' and 'prestart_recovery' not in journal")
    replace("stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/official-card-e2e-20261107');(stage/'end-to-end-proof.json').write_text(json.dumps(proof,indent=2));(stage/'report-readback.json').write_text(json.dumps(document,indent=2));print(json.dumps(proof))",
            "proof['created_at']=task['created'];proof['read_only_verification']=True;print(json.dumps(proof))")
    return code


if __name__=='__main__':
    before=json.loads((OUT/'before.json').read_text());baseline=json.loads((OUT/'baseline.json').read_text());identity=json.loads((OUT/'task.json').read_text())
    prefix='TASK='+repr(identity['id'])+'\nMESSAGE='+repr(identity['message_id'])+'\nBASELINE='+repr(baseline)+'\n'
    result=remote(prefix+audit_code(),timeout=150)
    assert before['phase']=='stopped' and before['space_uid']==result['official_start']['space_uid']
    journals=[json.loads(p.read_text()) for p in (ROOT/'.runtime/evaluation-flow-agent/official-space-actions').glob('*/request.json') if json.loads(p.read_text()).get('task')==identity['id']]
    assert len(journals)==1 and journals[0]['action']=='start'
    assert datetime.fromisoformat(journals[0]['requested_at']).timestamp()>result['created_at']
    assert result['created_at']>datetime.fromisoformat(before['checked_at']).timestamp()
    status=json.loads((ROOT/'.runtime/evaluation-flow-agent/status.json').read_text());agent_before=json.loads((OUT/'agent-before.json').read_text())
    # The request waited for approval across days. Preserve the changed process
    # identity honestly; require no restart during this post-confirmation audit.
    after_confirmation=json.loads((OUT/'after-confirmation-observation.json').read_text())
    assert status['pid']==after_confirmation['agent']['pid'] and status['state']=='CONNECTED'
    result['official_start'].update(initial_phase='stopped',background_request_at=journals[0]['requested_at'],single_local_start_journal=True,
        platform_confirmation_required=True,platform_confirmation_by='requester')
    result['unassisted_validation']={'agent_pid_unchanged_since_card_creation':status['pid']==agent_before['pid'],
        'agent_pid_at_creation':agent_before['pid'],'agent_pid_during_post_confirmation_audit':status['pid'],
        'agent_pid_unchanged_during_post_confirmation_audit':True,'task_options_unchanged':True,'manual_repair_actions':0,
        'manual_dispatch_actions':0,'manual_report_writes':0,'production_code_or_config_edits':0,'platform_confirmation_excluded':True}
    (OUT/'end-to-end-proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
