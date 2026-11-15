"""Verify real checkpoint accumulation and steady throughput after handoff."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,hashlib,numpy as np,datetime,os,statistics
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01');d=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-fast-20261030-r3')
source=json.loads((d/'restart-dispatch.json').read_text());start=source['preserved_prefix'];root=full/'compile';frozen=json.loads((full/'request.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(d/'before-restart-checkpoint.npz',allow_pickle=False) as cp:expected={k:cp[k].copy() for k in cp.files if k!='completed'}
with np.load(root/'checkpoint.npz',allow_pickle=False) as cp:cursor=int(cp['completed']);actual={k:cp[k].copy() for k in cp.files if k!='completed'}
times=[];board=[];GT=[];prep=[]
for i in range(start,cursor):
 f=root/'frames'/f'frame_{i:06d}';receipt=json.loads((f/'execution_receipt.json').read_text());ex=json.loads((f/'execution.json').read_text());proof=json.loads((f/'single-counts-receipt.json').read_text());public=json.loads((f/'public-outputs.json').read_text());allout=json.loads((f/'all-subnet-outputs.json').read_text())
 assert receipt['state']=='succeeded' and receipt['selected_frames']==[frozen['frames'][i]] and receipt['release']['state']=='RELEASED'
 assert len(public)==18 and len(allout)==49 and all(x['finite'] for x in allout.values())
 for p in public:
  meta=allout[p['tensor']];blob=f/('sub'+str(meta['stage']))/p['filename'];assert blob.stat().st_size==p['nbytes'] and sha(blob)==meta['sha256']
 assert proof['execution_sha256']==sha(f/'execution_receipt.json') and proof['GT_sha256']==sha(f/'gt_frame_receipt.json') and proof['counts_sha256']==sha(f/'single-counts.npz')
 with np.load(f/'single-counts.npz',allow_pickle=False) as cp:
  for key in expected:expected[key]+=cp[key]
 times.append((i,(f/'gt_frame_receipt.json').stat().st_mtime));board.append(ex['optimized_board_seconds_before_diagnostic']);GT.append(proof['GT_seconds']);prep.append(json.loads((f/'prepare.json').read_text())['preparation_seconds'])
assert all(np.array_equal(expected[k],actual[k]) for k in expected)
recent=times[-11:];assert len(recent)>=5,'Need more real committed frames before claiming steady throughput'
seconds=(recent[-1][1]-recent[0][1])/(len(recent)-1)
model=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01');report=json.loads((model/'full-report-coordinator.dispatch.json').read_text());os.kill(report['pid'],0)
p=json.loads((root/'progress.json').read_text());assert p['pid']==source['new_pid'] and p['state']=='RUNNING';os.kill(p['pid'],0)
value={'state':'VERIFIED','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_prefix':start,'current_prefix':cursor,
 'new_committed_frames':cursor-start,'counts_exactly_equal_old_plus_new':True,'all_new_frames49_checked18_downloaded':True,
 'sampled_steady_intervals':len(recent)-1,'seconds_per_frame':seconds,'speedup_vs_prior45_seconds':45.139577950200724/seconds,
 'estimated_remaining_hours_at_sample_speed':(8337-cursor)*seconds/3600,'recent_window':[datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat() for _,t in (recent[0],recent[-1])],
 'average_board_seconds':statistics.mean(board[-10:]),'average_GT_seconds':statistics.mean(GT[-10:]),'average_preparation_seconds':statistics.mean(prep[-10:]),
 'model_OCC_progress':json.loads((model/'model/progress.json').read_text()),'model_vision_progress':json.loads((model/'vision/progress.json').read_text()),
 'report_coordinator_alive':True,'report_coordinator_state':json.loads((model/'full-report/coordinator.json').read_text())['state'],
 'report_coordinator_failure_file_present':(model/'full-report-coordinator.failure.json').exists(),'native_repeated_completed_frames':0,
 'program_release':'compile-fast-20261030-r3','controller_pid':p['pid']}
assert not value['report_coordinator_failure_file_present'];out=d/('verification-'+str(cursor)+'.json');assert not out.exists();out.write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
if __name__=='__main__':print(call(CODE))
