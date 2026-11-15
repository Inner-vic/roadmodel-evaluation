"""Compare two READ-only SFTP methods on an already produced output.

No inference, remote write, lock change, or result overwrite is performed.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,sys,time,hashlib,datetime,shlex
method=__METHOD__
task=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
from board_infer_client import BoardInferClient,BoardInferConfig
client=BoardInferClient(BoardInferConfig(host='192.0.2.10',port=22,username='root'));client.connect()
pairs=[];meta=None;deadline=time.monotonic()+30
try:
 while time.monotonic()<deadline and not pairs:
  try:
   progress=json.loads((task/'compile/progress.json').read_text());i=progress['completed_frames'];root=task/'compile/frames'/f'frame_{i:06d}'
   ex=json.loads((root/'execution.json').read_text())
   if ex['state']!='RUNNING':time.sleep(.25);continue
   candidates=[x for s in ex['stages'] if s['state']=='SUCCEEDED' for x in s['outputs'] if 1024**2<=x['nbytes']<=8*1024**2]
   for x in sorted(candidates,key=lambda v:v['nbytes'],reverse=True)[:2]:
    p=x['remote_file'];assert p.startswith('/userdata/benchEval/models/'+ex['run_id']+'/sub') and ex['run_id'].startswith('pcm-v322-subnets-full-')
    a=client._sftp.file(p,'rb');b=client._sftp.file(p,'rb');a.settimeout(10);b.settimeout(10)
    assert a.stat().st_size==b.stat().st_size==x['nbytes'];pairs.append((x,a,b))
   if pairs:meta={'frame_index':i,'run_id':ex['run_id']}
  except (FileNotFoundError,KeyError,ValueError):pass
  if not pairs:time.sleep(.25)
 assert pairs,'No stable completed output was available within read-only probe window'
 rows=[]
 for x,a,b in pairs:
  try:
   started=time.monotonic();old=a.read();old_time=time.monotonic()-started
   started=time.monotonic()
   if method=='prefetch':
    b.prefetch(file_size=x['nbytes'],max_concurrent_requests=16);new=b.read()
   else:
    stdin,stdout,stderr=client._ssh.exec_command('cat -- '+shlex.quote(x['remote_file']),timeout=10)
    stdin.close();new=stdout.read(x['nbytes']+1);err=stderr.read();assert stdout.channel.recv_exit_status()==0 and not err
    stdout.close();stderr.close()
   new_time=time.monotonic()-started
   assert len(old)==len(new)==x['nbytes'] and hashlib.sha256(old).hexdigest()==hashlib.sha256(new).hexdigest()==x['sha256']
   rows.append({'tensor':x['tensor'],'bytes':x['nbytes'],'serial_read_seconds':old_time,'candidate_read_seconds':new_time,
                'speedup':old_time/new_time,'sha256':x['sha256'],'identical_bytes':True})
  finally:a.close();b.close()
 value={'state':'VERIFIED','method':method,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':meta,'samples':rows,
 'native_inference_frames':0,'GT_recomputations':0,'board_writes':0,'running_controller_modified':False,'scope':'read-only transport probe; not a measured full-frame speedup'}
 out=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/compile-timing-audit')/(method+'-read-probe-'+str(meta['frame_index'])+'.json');assert not out.exists();out.write_text(json.dumps(value,indent=2));print(json.dumps(value))
finally:
 for _,a,b in pairs:a.close();b.close()
 client.close()
'''
if __name__=='__main__':
    method=sys.argv[1] if len(sys.argv)>1 else 'prefetch'
    assert method in ('prefetch','stream')
    print(call(CODE.replace('__METHOD__',repr(method)),timeout=90))
