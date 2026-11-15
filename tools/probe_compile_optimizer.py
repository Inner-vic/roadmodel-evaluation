"""Read-only capability and source probe for the private fast controller."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import sys,pathlib,json,inspect,shlex
package=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020');sys.path.insert(0,str(package))
from board_infer_client import BoardInferClient,BoardInferConfig
c=BoardInferClient(BoardInferConfig(host='192.0.2.10',port=22,username='root'));c.connect()
try:
 code='import sys,json,shutil;print(json.dumps({"python":sys.version,"gcc":shutil.which("gcc"),"g++":shutil.which("g++")}));import numpy;print("numpy="+numpy.__version__)'
 rc,out,err=c._exec('python3 -c '+shlex.quote(code),timeout=15,check=False)
 print(json.dumps({'board':{'exit_code':rc,'out':out,'err':err}}))
finally:c.close()
import onnx_localagg_backend as agg
print(inspect.getsource(agg.localagg_from_gaussian_bins));print(inspect.getsource(agg._import_localagg))
from run_localagg_cpu import run_cpu_localagg
print(inspect.getsource(run_cpu_localagg))
from eval_metrics_bridge import MetricsAccumulator
print(inspect.getsource(MetricsAccumulator.summary))
task=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
root=task/'compile/frames/frame_000000';graph=json.loads((root/'chain-contract.json').read_text())
print(json.dumps({'stage0_keys':list(graph['stages'][0]),'future':[{ 'index':i,'exists':(task/'compile/frames'/f'frame_{i:06d}').exists()} for i in (8333,8334)]}))
'''
if __name__=='__main__':print(call(CODE))
