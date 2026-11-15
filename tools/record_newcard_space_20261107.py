"""Record only a fresh LPAI UI observation made by Codex in this test."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote

print(json.dumps(remote('''import sys,json,time,pathlib
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
from pcm_eval.evaluation import EvaluationRuntime
r=EvaluationRuntime('/mnt/data/bench-eval/state/roadmodel/evaluation');c=json.loads(r.config_file.read_text())
probe=json.loads(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107/space-readiness-probe.json').read_text())
assert time.time()-probe['checked_at']<180 and probe['ssh']=='CONNECTED' and probe['mount']=='READ_ONLY' and max(x[1] for x in probe['gpu_free_mib'])>=6144
observation={'source':'LPAI_UI','namespace':'vc-chassis','space_name':'roadmodel-eval-v322-20261106','observed_at':time.time(),'exists':True,
 'space':{'metadata':{'name':'roadmodel-eval-v322-20261106','namespace':'vc-chassis','uid':'lpai-ui:vc-chassis:roadmodel-eval-v322-20261106'},
 'spec':{'clusterName':'cnhb01-algorithm-bidu-p-03','dataVols':[{'name':'smartchassisdata-v3','type':'dataset','version':'26-10-20-1','mountPath':'/mnt/datasets/smartchassisdata-v3/26-10-20-1','mode':'ReadOnly','envKey':'LPAI_INPUT_DATASET_0','envValue':'/mnt/datasets/smartchassisdata-v3/26-10-20-1','fsType':'ofs'}]},
 'status':{'phase':'ready'},'verification':{'source':'LPAI_UI','identity_kind':'page_binding'},
 'runtime_access':{'host':'ssh-d.lpai.lixiang.com','port':30331,'credential_name':'bmad-roadmodel-20261009'}}}
completed=[]
while True:
 a=r.claim_space_action(c['agent_id'],c['owner'])
 if not a:break
 assert a['task']=='rm-aeedd04d020548c6aa58' and a['action']=='observe','Different action needs fresh review'
 r.complete_space_action(c['agent_id'],a['id'],{'state':'OBSERVED','real_browser_call_completed':True,'observation':observation,'readiness_probe':probe})
 completed.append(a['id'])
assert completed,'Expected queued UI observation'
pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/new-card-dual-20261107/running-ui-observation.json').write_text(json.dumps(observation,indent=2))
print(json.dumps({'state':'OBSERVED','actions':completed,'dataset':'26-10-20-1','phase':'ready'}))
''')))
