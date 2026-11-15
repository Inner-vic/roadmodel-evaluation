"""Install the reviewed project skills, preserving previous definitions as a workspace backup."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
home=Path(os.environ['CODEX_HOME']) if os.environ.get('CODEX_HOME') else Path.home()/'.codex'
target=(home/'skills').resolve()
names=['roadmodel-evaluation','roadmodel-model-accuracy','roadmodel-compile-alignment']
backup=ROOT/'evidence/compile-subnet-review-20261029'/'skill-backup'
out=[]
for name in names:
    source=ROOT/'skills'/name/'SKILL.md'
    dest=target/name/'SKILL.md'
    assert dest.resolve().is_relative_to(target)
    if dest.exists() and dest.read_bytes()!=source.read_bytes():
        prior=backup/name/'SKILL.md';prior.parent.mkdir(parents=True,exist_ok=True)
        if prior.exists():assert prior.read_bytes()==dest.read_bytes(),'Inspect previous update before repeating'
        else:shutil.copy2(dest,prior)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    assert dest.read_bytes()==source.read_bytes()
    out.append(str(dest))
receipt={'state':'installed','skills':out,'project_dependency':str(ROOT)}
path=ROOT/'evidence/compile-subnet-review-20261029/skill-installation.json'
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
