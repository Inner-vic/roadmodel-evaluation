"""Record this turn's actual LPAI UI observation after an SSH mount/GPU probe."""
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT/'evidence/core-e2e-20261107/reservation.json').read_text())['task']
PROBE = '''import pathlib,json,subprocess,hashlib
root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
rows=[r.split() for r in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if r.split()[4]==str(root)]
assert len(rows)==1 and 'ro' in rows[0][5].split(',')
p=root/'data_link_26-10-20-1_train859_eval135_v7.json';sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha=='e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c'
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert gpu.returncode==0
gpus=[tuple(map(int,line.split(','))) for line in gpu.stdout.splitlines() if line.strip()]
assert gpus and max(x[1] for x in gpus)>=6144
print(json.dumps({'ssh':'CONNECTED','mount':'READ_ONLY','index_sha256':sha,'gpu_free_mib':gpus,'dataset_copied':False}))
'''

CODE = '''import pathlib,json,sys,subprocess,shlex,time
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-model-core-20261107-r5');sys.path.insert(0,str(release))
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');cfg=json.loads(r.config_file.read_text());profile=cfg['profile']
payload=json.load(sys.stdin)
args=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=accept-new','-p','30331','-i',profile['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',payload['code']])]
p=subprocess.run(args,capture_output=True,text=True,timeout=50);assert p.returncode==0,p.stderr[-1000:]
probe=json.loads(p.stdout);probe['checked_at']=time.time()
observation={'source':'LPAI_UI','namespace':'vc-chassis','space_name':'roadmodel-eval-v322-20261106','observed_at':time.time(),'exists':True,
 'space':{'metadata':{'name':'roadmodel-eval-v322-20261106','namespace':'vc-chassis','uid':'lpai-ui:vc-chassis:roadmodel-eval-v322-20261106'},
 'spec':{'clusterName':'cnhb01-algorithm-bidu-p-03','dataVols':[{'name':'smartchassisdata-v3','type':'dataset','version':'26-10-20-1','mountPath':'/mnt/datasets/smartchassisdata-v3/26-10-20-1','mode':'ReadOnly','envKey':'LPAI_INPUT_DATASET_0','envValue':'/mnt/datasets/smartchassisdata-v3/26-10-20-1','fsType':'ofs'}]},
 'status':{'phase':'ready'},'verification':{'source':'LPAI_UI','identity_kind':'page_binding'},
 'runtime_access':{'host':'ssh-d.lpai.lixiang.com','port':30331,'credential_name':'bmad-roadmodel-20261009'}}}
completed=[]
with r.connect() as db:
 queued=[dict(x) for x in db.execute("select id,action,task from space_actions where state='QUEUED' order by updated")]
assert queued and all(a['task']==TASK and a['action']=='observe' for a in queued),'Different UI action must be reviewed'
for row in queued:
 a=r.claim_space_action(cfg['agent_id'],cfg['owner']);assert a['id']==row['id']
 r.complete_space_action(cfg['agent_id'],a['id'],{'state':'OBSERVED','real_browser_call_completed':True,'observation':observation,'readiness_probe':probe})
 completed.append(a['id'])
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/core-e2e-20261107')
(stage/'running-ui-observation.json').write_text(json.dumps(observation,indent=2));(stage/'space-readiness-probe.json').write_text(json.dumps(probe,indent=2))
print(json.dumps({'state':'OBSERVED','actions':completed,'probe':probe,'observation':observation}))
'''

if __name__ == '__main__':
    result = remote('TASK='+repr(TASK)+'\n'+CODE, json.dumps({'code': PROBE}).encode(), timeout=70)
    (ROOT/'evidence/core-e2e-20261107/space-readiness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'observation'}))
