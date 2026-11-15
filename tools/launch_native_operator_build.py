from pathlib import Path
from model_space_transport import run,put
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
print('script',put(Path('tools/build_native_model_ops_remote.py'),base+'/build_native_ops.py'))
code="import pathlib,subprocess,sys,json;p=pathlib.Path(sys.argv[1]);state=p/'operator-build-state.json';assert not state.exists(),'Build already started';log=(p/'operator-build-launch.log').open('w');proc=subprocess.Popen([sys.executable,str(p/'build_native_ops.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':proc.pid,'directory':str(p)}))"
print(run(['python3','-c',code,base]).decode())
