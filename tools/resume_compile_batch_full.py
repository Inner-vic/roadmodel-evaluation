"""Explicitly resume the sealed compile candidate, preserving all completed work."""
from connect_active_entry import call


CODE = r'''import pathlib,json,hashlib,subprocess,sys,os,shutil,datetime
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r2')
deploy=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-batch-20261030-r2')
audit=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-resume-20261102-r1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
assert not audit.exists(),'Inspect prior resume receipt; never dispatch twice'
pointer=load(full/'compile-next-controller.json');ready=load(pathlib.Path(pointer['candidate_receipt']))
assert sha(pathlib.Path(pointer['candidate_receipt']))==pointer['candidate_receipt_sha256']
assert ready['state']=='VERIFIED_INACTIVE_CANDIDATE' and ready['release']==str(release)
assert sha(release/'formal_compile_remote.py')==ready['controller_sha256']
assert sha(deploy/'staging.json')==ready['source_receipt_sha256']
for n,h in load(deploy/'staging.json')['source'].items():assert sha(release/n)==h,'Sealed candidate changed'
for proof in ready['proofs'].values():assert sha(pathlib.Path(proof['path']))==proof['sha256']
p=load(full/'compile/progress.json');assert p['state']=='STOPPED_AT_BOUNDARY' and p['completed_frames']==2151
assert not (full/'compile/controller.lock').exists()
proc=pathlib.Path('/proc')/str(p['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert sha(full/'compile/checkpoint.npz')==ready['checkpoint_sha256']
assert sha(full/'request.json')=='78880091a77e74d5b42ca0b26597c7e33fead188495022e0a22cc82fbadb93b9'
assert (full/'STOP_AFTER_FRAME').is_file()
req=load(full/'request.json');assert req['model_version']=='v3.2.2' and len(req['frames'])==8337
cache={}
for slot in ready['retained_slots']:
 root=full/'compile/frames'/('frame_%06d'%slot['index'])
 names=['execution_receipt.json','single-counts.npz','single-counts-receipt.json'] if slot['state']=='NATIVE_AND_GT_RETAINED' else ['prepare.json']
 for n in names:cache[str(root/n)]=sha(root/n)
 if slot['state']=='NATIVE_AND_GT_RETAINED':
  receipt=load(root/'execution_receipt.json');counts=load(root/'single-counts-receipt.json')
  assert receipt['state']=='succeeded' and receipt['release']['state']=='RELEASED'
  assert receipt['selected_frames']==[req['frames'][slot['index']]]
  assert counts['counts_sha256']==sha(root/'single-counts.npz') and counts['execution_sha256']==sha(root/'execution_receipt.json')
 else:assert load(root/'prepare.json')['state']=='READY'
audit.mkdir();shutil.copy2(full/'compile/checkpoint.npz',audit/'before-checkpoint.npz');shutil.copy2(full/'compile/progress.json',audit/'before-progress.json')
(audit/'retained-cache-hashes.json').write_text(json.dumps(cache,indent=2))
value={'state':'DISPATCHING','model_version':'v3.2.2','resumed_from_prefix':2151,'expected_frames':8337,
       'release':str(release),'controller_sha256':ready['controller_sha256'],
       'retained_native_and_GT_slots':sum(x['state']=='NATIVE_AND_GT_RETAINED' for x in ready['retained_slots']),
       'retained_reference_only_slots':sum(x['state']=='REFERENCE_READY_NATIVE_NOT_DISPATCHED' for x in ready['retained_slots']),
       'same_original_metrics_and_input_contract':True,'next_930_dual_image_and_expanded_ROI_not_applied':True,
       'user_resume_authorized':True,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(audit/'dispatch.json').write_text(json.dumps(value,indent=2))
# Preserve the former pause evidence; the user explicitly revoked that pause.
(full/'STOP_AFTER_FRAME').replace(audit/'STOP_AFTER_FRAME.before-resume')
try:
 with (audit/'controller.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,str(release/'formal_compile_remote.py')],cwd=release,
    env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
except Exception:
 (audit/'STOP_AFTER_FRAME.before-resume').replace(full/'STOP_AFTER_FRAME')
 value['state']='FAILED_BEFORE_SPAWN';(audit/'dispatch.json').write_text(json.dumps(value,indent=2));raise
value.update(state='STARTED',pid=child.pid)
(audit/'dispatch.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''


if __name__=='__main__':print(call(CODE,timeout=90))
