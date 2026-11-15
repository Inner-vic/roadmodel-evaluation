from pathlib import Path
from model_space_transport import run,put
root=Path('evidence/model-evaluation/remaining-heads-20261027')
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027'
for local,remote in [('tools/trajectory_head_adapter_remote.py','trajectory_head_adapter.py'),
 ('tools/head_preflight_remote.py','head_preflight.py'),('tools/run_remaining_heads_remote.py','run_remaining_heads.py')]:
 put(Path(local),base+'/'+remote)
for kind in ['vision','trajectory']:put(root/(kind+'-launch.json'),base+'/'+kind+'-01/launch.json')
code="import pathlib,subprocess,sys,json;b=pathlib.Path(sys.argv[1]);assert not (b/'workflow-state.json').exists(),'Workflow already started';log=(b/'workflow-console.log').open('w');p=subprocess.Popen([sys.executable,str(b/'run_remaining_heads.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid,'directory':str(b)}))"
print(run(['python3','-c',code,base]).decode())
