"""Read-only status for the fresh model timing run, including actual native work."""
from connect_active_entry import call
from prepare_timed_model_rerun import TASK

code = '''import pathlib,json,hashlib,os,subprocess,shlex,datetime
r=pathlib.Path(%r);cfg=json.loads((r/'run-config.json').read_text())
print('UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for n in ['dispatch.json','dispatch-resume1.json','dispatch-resume2.json','timing-start.json','progress.json','model/progress.json','vision/progress.json','timing-summary.json']:
 p=r/n
 if p.exists():
  try:
   item=json.loads(p.read_text());label=n
   if n=='timing-summary.json' and (r/'progress.json').exists() and json.loads((r/'progress.json').read_text()).get('state')=='RUNNING' and item.get('state')=='NEEDS_REVIEW':label='PRIOR_SETUP_FAILURE_SNAPSHOT'
   print(label,json.dumps(item))
  except json.JSONDecodeError:print(n,'being_written')
dispatch=next(r/n for n in ['dispatch-resume2.json','dispatch-resume1.json','dispatch.json'] if (r/n).exists())
d=json.loads(dispatch.read_text()) if dispatch.exists() else {}
cmd=pathlib.Path('/proc')/str(d.get('pid',-1))/'cmdline'
print('CONTROLLER_ALIVE',cmd.exists() and str(r).encode() in cmd.read_bytes())
if (r/'timing-events.jsonl').exists():print('LATEST_TIMING',(r/'timing-events.jsonl').read_text().splitlines()[-8:])
log=next(r/n for n in ['controller-resume2.log','controller-resume1.log','controller.log'] if (r/n).exists())
if log.exists():print('LOG_TAIL',log.read_text(errors='replace')[-2400:])
print('COMPILE_PAUSE_UNCHANGED',all(hashlib.sha256((pathlib.Path(cfg['compile_root'])/n).read_bytes()).hexdigest()==h for n,h in cfg['compile_pause_hashes'].items()))
probe="import pathlib,json,re,subprocess;root=pathlib.Path(%r);b=root/'occ/batch-0000-a02';p=b/'execution_receipt.json';d=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({k:d.get(k) for k in ['state','run_id','started_at','expected_frames','finished_at','exit_code']}));l=b/'eval-console.log';s=l.read_text(errors='replace')[-16000:] if l.exists() else '';print('NATIVE_PROGRESS',re.findall(r'(\\\\d+)/100',s)[-5:]);print('NATIVE_LOG_TAIL',s[-1200:]);print('NATIVE_TIMING',(b/'native-timing.jsonl').read_text()[-1800:] if (b/'native-timing.jsonl').exists() else 'not_started')"
argv=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','31017','-i',cfg['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',probe])]
p=subprocess.run(argv,capture_output=True,text=True,timeout=35);print('GPU_SPACE',p.returncode,p.stdout,p.stderr[-500:])
''' % (TASK, '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/model-timing-v322-20261102-r1')

if __name__ == '__main__':
    print(call(code, timeout=70))
