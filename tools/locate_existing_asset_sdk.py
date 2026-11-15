from pathlib import Path
import json,sys,importlib.util
p=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
c=next(x for x in json.loads(p.read_text(encoding='utf-8-sig'))['connections'] if x['id']=='lpai-sdk-closedloop-20261022')
print('settings_keys',list(c['settings']))
for k,v in c['settings'].items():
 if any(word in k.lower() for word in ['python','venv','sdk_path','module_path']):print(k,v)
print('python',sys.executable)
for base in ['D:/developer/Desktop/Project','D:/codex','C:/Users/developer/.local/share/uv/tools','C:/Users/developer/AppData/Local/Programs/Python']:
 p=Path(base)
 if p.exists():print('directories',base,[x.name for x in p.iterdir() if x.is_dir()][:70])
for p in Path('D:/developer/Desktop/Project').glob('**/pyvenv.cfg'):
 site=p.parent/'Lib/site-packages'
 if (site/'asset').exists():print('FOUND_SDK',site)
