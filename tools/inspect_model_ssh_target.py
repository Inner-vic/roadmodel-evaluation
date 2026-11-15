import json
import shutil
import subprocess
from pathlib import Path
proc=subprocess.run([shutil.which('ssh.exe') or 'ssh','-G','lpai'],capture_output=True,timeout=10)
allowed={'hostname','port','user','identityfile','identitiesonly','hostkeyalias'}
fields={}
for line in proc.stdout.decode(errors='replace').splitlines():
 key,_,value=line.partition(' ')
 if key in allowed:fields.setdefault(key,[]).append(value)
root=Path('evidence/model-evaluation/model-reference-20261027')
(root/'existing-ssh-target.json').write_text(json.dumps(fields,indent=2)+'\n')
print(json.dumps(fields))
