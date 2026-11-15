"""Read the precise original task code; do not checkout or execute it."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
CODE=r'''import pathlib,subprocess,os,json
repo='/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69';commit='69d7286ff6f92590325d4dbe79c35cd4d6e1d4e5'
result={'source_commit':commit,'source_executed':False,'checkout_performed':False}
remote=subprocess.run(['git','-C',repo,'remote','get-url','origin'],capture_output=True,text=True,timeout=10)
result['configured_origin_available']=remote.returncode==0
if remote.returncode==0:
 from urllib.parse import urlsplit
 url=remote.stdout.strip();host=urlsplit(url).hostname if '://' in url else url.split('@')[-1].split(':')[0]
 allowed=host and (host.endswith('.lixiang.com') or host.endswith('.chehejia.com'))
 result['internal_origin']=bool(allowed)
 if allowed:
  try:
   p=subprocess.run(['git','-C',repo,'fetch','--no-tags','origin',commit],capture_output=True,timeout=45,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
   result['fetch_exit_code']=p.returncode
  except subprocess.TimeoutExpired:result['fetch_state']='TIMED_OUT'
has=subprocess.run(['git','-C',repo,'cat-file','-e',commit+'^{commit}'],capture_output=True,timeout=10).returncode==0
result['commit_available']=has
if has:
 p=subprocess.run(['git','-C',repo,'ls-tree','-r','--name-only',commit],capture_output=True,text=True,timeout=10)
 names=[n for n in p.stdout.splitlines() if n.endswith('.py') and any(w in n.lower() for w in ('minidataset','datalink','v16','dataset_export'))]
 result['candidate_files']=names[:50];hits=[]
 for name in names[:20]:
  p=subprocess.run(['git','-C',repo,'show',commit+':'+name],capture_output=True,text=True,timeout=10)
  for i,line in enumerate(p.stdout.splitlines()):
   if any(word in line for word in ('img_label','vision_eval','minidataset')) and not any(word in line.upper() for word in ('PASSWORD','TOKEN','SECRET','API_KEY')):
    hits.append({'file':name,'line':i+1,'code':line.strip()[:220]})
 result['label_export_hints']=hits[:80]
print(json.dumps(result))
'''
v=json.loads(run(['python3','-c',CODE],timeout=90))
out=Path(__file__).resolve().parents[1]/'evidence/vision-frame-binding-20261107/original-export-commit.json';out.write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
