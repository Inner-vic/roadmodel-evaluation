from pathlib import Path
from model_space_transport import run,put
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027'
script=base+'/run_trajectory_diagnostic_01.py'
put(Path('tools/run_trajectory_diagnostic_remote.py'),script)
code="import pathlib,subprocess,sys,json;b=pathlib.Path(sys.argv[1]);assert not (b/'trajectory-diagnostic-01').exists();log=(b/'trajectory-diagnostic-01-console.log').open('x');p=subprocess.Popen([sys.executable,sys.argv[2]],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid,'script':sys.argv[2]}))"
print(run(['python3','-c',code,base,script]).decode())
