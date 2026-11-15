"""Read current native progress and recent completion times; never dispatch work."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE=r'''import pathlib,json,sys,os,statistics,datetime,shutil,subprocess
from zoneinfo import ZoneInfo
m=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01')
c=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
now=datetime.datetime.now(datetime.timezone.utc)
def load(p):return json.loads(p.read_text())
def stamp(p):return datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()
mp=load(m/'model/progress.json');cp=load(c/'compile/progress.json')
out={'checked_at':now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),'model_progress':mp,'compile_progress':cp,
 'vision_data':load(m/'vision-data-state.json'),'report_coordinator':load(m/'full-report/coordinator.json'),
 'report_failure':load(m/'full-report-coordinator.failure.json') if (m/'full-report-coordinator.failure.json').exists() else None,
 'disk_free_GiB':round(shutil.disk_usage(c).free/1024**3,1)}
ends=[]
for i in range(max(0,cp['completed_frames']-31),cp['completed_frames']):
 p=c/'compile/frames'/f'frame_{i:06d}'/'gt_frame_receipt.json'
 if p.exists():ends.append({'index':i,'time':p.stat().st_mtime})
if len(ends)>1:
 seconds=(ends[-1]['time']-ends[0]['time'])/(len(ends)-1)
 out['compile_recent']={'completed_intervals':len(ends)-1,'seconds_per_frame':seconds,'remaining_hours':(8337-cp['completed_frames'])*seconds/3600,
   'window_first':datetime.datetime.fromtimestamp(ends[0]['time'],datetime.timezone.utc).isoformat(),'window_last':datetime.datetime.fromtimestamp(ends[-1]['time'],datetime.timezone.utc).isoformat()}
complete=[]
for p in sorted((m/'model').glob('batch-*/normalized-result.json')):
 data=load(p);complete.append({'batch':p.parent.name,'frames':data['coverage']['expected_frames'],'time':p.stat().st_mtime})
recent=sorted([x for x in complete if x['frames']==100],key=lambda x:x['time'])[-11:]
if len(recent)>1:
 seconds=(recent[-1]['time']-recent[0]['time'])/(100*(len(recent)-1))
 out['model_recent']={'batch_intervals':len(recent)-1,'seconds_per_frame':seconds,'remaining_OCC_hours':max(0,8337-sum(x['frames'] for x in complete))*seconds/3600,
   'completed_including_reserved_tail':sum(x['frames'] for x in complete),'last_batches':recent}
for label,root in [('model',m),('compile',c)]:
 logs=[p for p in root.glob('*.log') if p.is_file()]
 out[label+'_controller_logs']={p.name:p.read_text(errors='replace')[-1600:] for p in logs}
active=c/'compile/frames'/f"frame_{cp['completed_frames']:06d}"
out['compile_current_artifacts']=[{'name':p.name,'modified':stamp(p)} for p in active.glob('*') if p.is_file()]
samples=[c/'compile/frames'/f'frame_{i:06d}' for i in range(max(0,cp['completed_frames']-3),cp['completed_frames'])]
usage=subprocess.run(['du','-sk',*[str(p) for p in samples]],capture_output=True,text=True,timeout=20)
out['recent_frame_disk_KiB']=usage.stdout.strip()
for p in [m/'full-report-coordinator.dispatch.json',m/'full-report-coordinator.dispatch.log',m/'full-report-coordinator.log']:
 if p.exists():out[p.name]=load(p) if p.suffix=='.json' else p.read_text(errors='replace')[-1800:]
# Read only the currently selected GPU and current batch; no content enumeration.
os.environ['ROADMODEL_IDENTITY']='/mnt/data/bench-eval/service/gateway/runtime/roadmodel/model-space.key'
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/resume-20261030/tools')
from model_space_transport import run
from formal_gpu import selected_gpu
gpu=selected_gpu(m)
space=r"""import pathlib,json,subprocess,sys
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029')
batch=sys.argv[1];p=base/'occ'/batch
gpu=subprocess.run(['nvidia-smi','--id='+sys.argv[2],'--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps({'GPU':gpu.stdout.strip(),'batch_directory':str(p),'execution_receipt':json.loads((p/'execution_receipt.json').read_text()) if (p/'execution_receipt.json').exists() else None,'console_tail':(p/'eval-console.log').read_text(errors='replace')[-1500:] if (p/'eval-console.log').exists() else None}))
"""
attempts=load(m/'model-attempts.json');batch=f"batch-{mp.get('batch',0):04d}-{attempts.get(str(mp.get('batch',0)),'a02')}"
try:out['model_space']=json.loads(run(['python3','-c',space,batch,gpu],timeout=40))
except Exception as e:out['model_space_read_error']=str(e)[-500:]
print(json.dumps(out,ensure_ascii=False))
'''

if __name__=='__main__':print(call(CODE,timeout=100))
