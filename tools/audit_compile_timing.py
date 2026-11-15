"""Read only recorded compile timings and implementation; never rerun frames."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,datetime,statistics,re,hashlib
task=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
progress=json.loads((task/'compile/progress.json').read_text());n=progress['completed_frames'];rows=[]
for i in range(max(1,n-31),n):
 root=task/'compile/frames'/f'frame_{i:06d}';prev=task/'compile/frames'/f'frame_{i-1:06d}'
 def load(name):return json.loads((root/name).read_text())
 def mt(name):return (root/name).stat().st_mtime
 ex=load('execution.json');rc=load('execution_receipt.json')
 started=datetime.datetime.fromisoformat(rc['started_at']).timestamp()
 stages=[]
 for s in ex['stages']:
  k=s['stage'];files=[root/('sub'+str(k))/x['filename'] for x in s['outputs']]
  finished=max(p.stat().st_mtime for p in files);returned=(root/f'sub{k}.stdout.log').stat().st_mtime
  fetch=max(0,finished-returned)
  log=(root/f'sub{k}.stdout.log').read_text(errors='replace');us=re.findall(r'Infer time:\s*(\d+)\s*us',log)
  stages.append({'stage':k,'seconds':s['seconds'],'outputs':len(s['outputs']),'bytes':sum(x['nbytes'] for x in s['outputs']),
    'post_command_collection_seconds_approx':fetch,'command_seconds_approx':max(0,s['seconds']-fetch),'kernel_seconds':sum(map(int,us))/1e6})
 end=mt('gt_frame_receipt.json');previous=(prev/'gt_frame_receipt.json').stat().st_mtime
 rows.append({'index':i,'total':end-previous,'export_and_prepare':mt('prepare.json')-previous,
   'admission':started-mt('prepare.json'),'board_transfer_chain_compare_release':mt('execution_receipt.json')-started,
   'GT':end-mt('execution_receipt.json'),'five_stages':sum(s['seconds'] for s in stages),'stages':stages})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'progress':progress,'frames_sampled':len(rows),
 'averages':{k:statistics.mean(r[k] for r in rows) for k in rows[0] if k not in ('index','stages')},'recent_frames':rows[-4:]}
out['stage_averages']=[{'stage':k,**{f:statistics.mean(r['stages'][k][f] for r in rows) for f in rows[0]['stages'][k] if f!='stage'}} for k in range(5)]
root=task/'compile/frames'/f'frame_{n-1:06d}'
out['last_frame_timeline']={p.name:p.stat().st_mtime for p in root.iterdir() if p.is_file()}
driver=(task/'code/formal_subnet_chain.py').read_text();a=driver.index('def prepare(');b=driver.index('\ndef write_compatible_results',a)
out['formal_prepare_source']=driver[a:b]
package=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
ort=(package/'onnxruntime_infer.py').read_text();a=ort.index('def run_inference(');b=ort.index('\ndef main(',a);out['ort_run_source']=ort[a:b]
client=(package/'board_infer_client.py').read_text()
for name in ('_sftp_read_bytes','_sftp_write_bytes','_exec'):
 start=client.find('    def '+name+'(');out[name]=client[start:client.find('\n    def ',start+5)]
evidence=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/compile-timing-audit');evidence.mkdir(exist_ok=True)
data={k:v for k,v in out.items() if not k.endswith('_source') and k not in ('_exec','_sftp_read_bytes','_sftp_write_bytes')}
data['driver_sha256']=hashlib.sha256(driver.encode()).hexdigest();target=evidence/f'frames-{n-31:06d}-{n-1:06d}.json';assert not target.exists();target.write_text(json.dumps(data,ensure_ascii=False,indent=2));out['evidence']=str(target)
print(json.dumps(out,ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE,timeout=90))
