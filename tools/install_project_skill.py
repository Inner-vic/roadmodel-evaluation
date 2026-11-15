"""Install only the validated project skill, refusing to overwrite another one."""
import json
import os
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
source=root/'skills/roadmodel-evaluation/SKILL.md'
skill_home=(Path(os.environ['CODEX_HOME']) if os.environ.get('CODEX_HOME') else Path.home()/'.codex')/'skills'
target=(skill_home/'roadmodel-evaluation').resolve()
if target.parent!=skill_home.resolve():raise ValueError('Unexpected skill installation target')
dest=target/'SKILL.md'
if dest.exists() and dest.read_bytes()!=source.read_bytes():raise ValueError('An existing different skill must be reviewed before update')
target.mkdir(parents=True,exist_ok=True)
if not dest.exists():shutil.copyfile(source,dest)
if dest.read_bytes()!=source.read_bytes():raise ValueError('Installed skill content differs')
receipt={'state':'installed','skill':str(dest),'project_dependency':str(root)}
out=root/'evidence/batch-upgrade-20261029/skill-installation.json'
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
