"""Inspect the task's logged real commit, not its cache directory identifier."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/camera-GT-mapping-20261107'
CODE=r'''import pathlib,subprocess,json,sys,os,re
from urllib.parse import urlsplit
commit=json.load(sys.stdin)['commit'];repo='/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69'
result={'actual_task_commit':commit,'working_tree_changed':False,'source_executed':False}
p=subprocess.run(['git','-C',repo,'remote','get-url','origin'],capture_output=True,text=True,timeout=10)
if p.returncode==0:
 value=p.stdout.strip();url=urlsplit(value) if '://' in value else None
 host=url.hostname if url else value.split('@')[-1].split(':')[0];path=url.path if url else value.split(':',1)[-1]
 result['repository']={'host':host,'path':path}
 if host and (host.endswith('.lixiang.com') or host.endswith('.chehejia.com')):
  try:
   p=subprocess.run(['git','-C',repo,'fetch','--no-tags','origin',commit],capture_output=True,text=True,timeout=40,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
   result['fetch_exit_code']=p.returncode
   result['fetch_error_categories']=[name for name,pattern in [('authentication','authentication|access denied|could not read Username'),('DNS','Could not resolve'),('connection','timed out|Connection refused'),('missing_object','not our ref|unadvertised|couldn.t find remote ref'),('permission','Permission denied')] if re.search(pattern,p.stderr,re.I)]
  except subprocess.TimeoutExpired:result['fetch_error_categories']=['timeout']
exists=subprocess.run(['git','-C',repo,'cat-file','-e',commit+'^{commit}'],capture_output=True,timeout=10).returncode==0
result['commit_available']=exists;result['source_files']=[]
if exists:
 p=subprocess.run(['git','-C',repo,'ls-tree','-r','--name-only',commit],capture_output=True,text=True,timeout=10)
 names=[n for n in p.stdout.splitlines() if n.endswith('.py') and any(w in n.lower() for w in ('reorg','timestamp','align','camera','v16','minidataset'))]
 result['candidate_paths']=names[:100]
 # Source only, no execution. Hash each exact file and return bounded relevant files.
 for name in names:
  p=subprocess.run(['git','-C',repo,'show',commit+':'+name],capture_output=True,text=True,timeout=10)
  if any(t in p.stdout for t in ('generation_timestamp_us','refined_transform_camera','timestamp_offset_us')) and len(p.stdout)<300000:
   import hashlib
   result['source_files'].append({'path':name,'sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'content':p.stdout})
  if len(result['source_files'])>=6:break
print(json.dumps(result))
'''
e=json.loads((OUT/'repository-revision-evidence.json').read_text());commits={h for p in e['preparation_logs'] for r in p.get('revision_evidence',[]) for h in r.get('announced_HEAD',[])}
assert len(commits)==1;commit=next(iter(commits))
v=json.loads(run(['python3','-c',CODE],json.dumps({'commit':commit}).encode(),timeout=100))
for f in v.pop('source_files'):
    path=OUT/'source'/f['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f['content'],encoding='utf-8')
    v.setdefault('collected_source_files',[]).append({k:f[k] for k in ('path','sha256')})
(OUT/'actual-source-access.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
