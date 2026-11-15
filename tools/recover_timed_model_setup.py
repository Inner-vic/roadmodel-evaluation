"""Resume only a proven pre-inference setup failure, preserving the first clock."""
from prepare_timed_model_rerun import upload_bundle,TASK,RELEASE
from connect_active_entry import call


def main():
    archive,sha=upload_bundle()
    code='''import pathlib,json,hashlib,subprocess,shlex,shutil,tarfile,io,sys,os,datetime
task=pathlib.Path(%r);oldrel=pathlib.Path(%r);release=pathlib.Path(str(oldrel)+'-setup-fix2')
cfg=json.loads((task/'run-config.json').read_text());summary=json.loads((task/'timing-summary.json').read_text())
assert summary['state']=='NEEDS_REVIEW' and not list((task/'model').glob('batch-*'))
assert not release.exists() and not (task/'dispatch-resume2.json').exists()
original=json.loads((task/'dispatch-resume1.json').read_text());proc=pathlib.Path('/proc')/str(original['pid'])/'cmdline'
assert not proc.exists() or str(task).encode() not in proc.read_bytes(),'Original controller still active'
for n,h in cfg['compile_pause_hashes'].items():assert hashlib.sha256((pathlib.Path(cfg['compile_root'])/n).read_bytes()).hexdigest()==h
probe='import pathlib,json;root=pathlib.Path('+repr(cfg['space_root'])+');r=list(root.rglob("execution_receipt.json"));assert not r,"Native work already dispatched; reconcile instead";print("ZERO_NATIVE_RECEIPTS")'
args=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','31017','-i',cfg['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',probe])]
p=subprocess.run(args,capture_output=True,text=True,timeout=35);assert p.returncode==0,p.stderr;assert p.stdout.strip()=='ZERO_NATIVE_RECEIPTS'
raw=pathlib.Path(%r).read_bytes();assert hashlib.sha256(raw).hexdigest()==%r
shutil.copytree(oldrel,release)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();p=release/member.name;assert p.resolve().is_relative_to(release);p.write_bytes(tar.extractfile(member).read())
history=task/'timing-history';history.mkdir(exist_ok=True);(history/'setup-etag-rule-error.json').write_text(json.dumps(summary,indent=2))
cfg['resume_setup_only']=True;cfgpath=task/'run-config-resume2.json';cfgpath.write_text(json.dumps(cfg,indent=2))
env={**os.environ,'PYTHONUNBUFFERED':'1','ROADMODEL_TASK_ROOT':str(task),'ROADMODEL_STAGE_LOCAL':'1','ROADMODEL_IDENTITY':cfg['identity_file']}
log=(task/'controller-resume2.log').open('xb');p=subprocess.Popen([sys.executable,str(release/'tools/timed_model_rerun.py'),str(cfgpath)],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'STARTED','task_id':cfg['task_id'],'pid':p.pid,'release':str(release),'resumed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_started_at':summary['started_at'],'new_inference_before_recovery':0,'clock_reset':False,'compile_paused':True}
(task/'dispatch-resume2.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
''' % (TASK,RELEASE,archive,sha)
    print(call(code,timeout=80))


if __name__=='__main__':main()
