import subprocess,json
from pathlib import Path
roots=['D:/developer/Desktop/Project/AutoOpt','D:/developer/Desktop/Project/Agent',
 'D:/codex/cache','C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab']
p=subprocess.run(['rg','--files','--hidden','--no-ignore','--no-messages','-g','__init__.py','-g','*asset*.whl',*roots],capture_output=True,text=True)
print(json.dumps([x for x in p.stdout.splitlines() if '/asset/' in x.replace('\\','/') or x.endswith('.whl')]))
