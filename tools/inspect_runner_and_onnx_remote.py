import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_subnet_remote import remote

code=r'''
import pathlib,json,sys
p=pathlib.Path('/home/lixiang/bench-eval/current/automation/runner_state.py');lines=p.read_text().splitlines()
out={'runner_api':'\n'.join(f'{i+1}: {line}' for i,line in enumerate(lines) if line.startswith(('def ','class '))),
 'runner_claim_source':'\n'.join(f'{i+1}: {lines[i]}' for i in list(range(0,70))+list(range(190,287))),
 'onnx':{}}
sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/venv/lib/python3.12/site-packages')
import onnx
m=onnx.load('/mnt/data/developer/pcm-roadmodel-eval/model-materials/v3.2.2-subnets/v322.onnx',load_external_data=False)
def port(x):return {'name':x.name,'shape':[d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim],
 'dtype':onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type)}
out['onnx']={'inputs':[port(x) for x in m.graph.input],'outputs':[port(x) for x in m.graph.output],
 'custom_ops':sorted({(x.domain,x.op_type) for x in m.graph.node if x.domain})}
print(json.dumps(out))
'''
if __name__=='__main__':
    # Workstation venv is authoritative; avoid relying on system site-packages.
    code=code.replace("sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/venv/lib/python3.12/site-packages')",'')
    from pcm_eval.planning import ssh_command
    import subprocess
    p=subprocess.run(ssh_command('ubuntu-workstation',['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python','-']),input=code.encode(),capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
    value=json.loads(p.stdout);Path('evidence/compile-subnet-review-20261029/runner-and-onnx.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False,indent=2))
