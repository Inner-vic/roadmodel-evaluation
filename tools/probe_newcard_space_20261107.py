"""Read-only SSH, mount and GPU readiness proof for the observed space."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
code='''import pathlib,json,subprocess,os
root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
index=root/'data_link_26-10-20-1_train859_eval135_v7.json'
mounts=[line for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4]==str(root)]
assert len(mounts)==1 and 'ro' in mounts[0].split()[5].split(','),'Input dataset is not a unique read-only mount'
assert index.is_file(),'Input index unavailable'
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
assert gpu.returncode==0,gpu.stderr
rows=[tuple(map(int,line.split(','))) for line in gpu.stdout.splitlines() if line.strip()]
assert rows and max(x[1] for x in rows)>=6144,'No visible GPU has sufficient free memory'
data=json.loads(index.read_text())
print(json.dumps({'ssh':'CONNECTED','mount':'READ_ONLY','root':str(root),'index':str(index),'index_bytes':index.stat().st_size,'index_keys':list(data),'gpu_free_mib':rows,'dataset_copied':False}))
'''
print(json.dumps(remote('''import pathlib,json,sys,subprocess,shlex,time
profile=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4/configs/evaluation.json').read_text())['profile']
code=json.load(sys.stdin)['code']
cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=accept-new','-p','30331','-i',profile['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',code])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
assert p.returncode==0,p.stderr[-1800:]
proof=json.loads(p.stdout);proof['checked_at']=time.time()
pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107/space-readiness-probe.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
''',json.dumps({'code':code}).encode(),timeout=60)))
