from pathlib import Path
from model_space_transport import run,put
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
put(Path('tools/run_native_model_remote.py'),base+'/run_native_model.py')
code="import pathlib,subprocess,sys,json;p=pathlib.Path(sys.argv[1]);assert not (p/'execution_receipt.json').exists(),'Run already started';log=(p/'eval-launch.log').open('w');proc=subprocess.Popen([sys.executable,str(p/'run_native_model.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':proc.pid,'directory':str(p)}))"
print(run(['python3','-c',code,base]).decode())
