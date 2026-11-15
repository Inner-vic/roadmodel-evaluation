"""Locate existing relevant SSH key paths; never read or output key material."""
import json
from pathlib import Path
home=Path.home()/'.ssh'
names=[p.name for p in home.iterdir() if p.is_file() and ('bmad' in p.name.lower() or 'lpai' in p.name.lower())]
aliases=[];current={}
for line in (home/'config').read_text(encoding='utf-8').splitlines():
 stripped=line.strip()
 if not stripped or stripped.startswith('#'):continue
 key,_,value=stripped.partition(' ');key=key.lower();value=value.strip()
 if key=='host':
  if current:aliases.append(current)
  current={'host':value}
 elif key in ('hostname','port','identityfile','user'):
  current[key]=value
if current:aliases.append(current)
print(json.dumps({'existing_relevant_key_filenames':names,'relevant_aliases':[a for a in aliases if any(k in str(a).lower() for k in ['lpai','bmad'])]}))
