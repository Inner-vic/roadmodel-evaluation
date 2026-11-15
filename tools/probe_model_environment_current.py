"""Check an existing SSH alias without changing the target or exposing config."""
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

root=Path('evidence/environment/model-input-review-20261027');root.mkdir(parents=True,exist_ok=True)
cmd=[shutil.which('ssh.exe') or 'ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
     '-o','ConnectionAttempts=1','lpai','true']
result={'checked_at':datetime.now(timezone.utc).isoformat(),'ssh_alias':'lpai','remote_action':'true',
        'changed_remote_state':False,'new_evaluation_started':False}
try:
    proc=subprocess.run(cmd,capture_output=True,timeout=15)
    result.update(state='reachable' if proc.returncode==0 else 'unavailable',exit_code=proc.returncode)
    result['observation']='Read-only SSH probe succeeded' if proc.returncode==0 else 'Existing SSH alias did not establish a usable session'
except subprocess.TimeoutExpired:
    result.update(state='unavailable',observation='Read-only SSH probe timed out after 15 seconds')
(root/'ssh-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
