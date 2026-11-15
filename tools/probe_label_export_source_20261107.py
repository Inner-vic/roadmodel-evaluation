import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
code=r'''import pathlib,json,subprocess
repos=[pathlib.Path('/lpai/bmad-r3-a100-20261015'),pathlib.Path('/lpai/bmad-roadmodel-source'),pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')]
commit='69d7286ff6f92590325d4dbe79c35cd4d6e1d4e5';rows=[]
for repo in repos:
 if not (repo/'.git').exists():continue
 exists=subprocess.run(['git','-C',str(repo),'cat-file','-e',commit+'^{commit}'],capture_output=True,timeout=15).returncode==0
 row={'repo':str(repo),'has_original_data_task_commit':exists}
 if exists:
  p=subprocess.run(['git','-C',str(repo),'ls-tree','-r','--name-only',commit],capture_output=True,text=True,timeout=15)
  names=[s for s in p.stdout.splitlines() if s.endswith('.py') and any(t in s.lower() for t in ('minidataset','datalink','v161','v16'))];row['candidate_files']=names[:35];row['label_handling']=[]
  for name in names[:25]:
   p=subprocess.run(['git','-C',str(repo),'show',commit+':'+name],capture_output=True,text=True,timeout=15)
   for i,line in enumerate(p.stdout.splitlines()):
    if any(s in line for s in ('img_label','vision_eval','minidataset','timestamp_offset_us')):
     row['label_handling'].append({'file':name,'line':i+1,'code':line.strip()[:200]})
  row['label_handling']=row['label_handling'][:90]
 rows.append(row)
print(json.dumps({'repositories':rows,'source_commit':commit,'new_inference_frames':0,'git_fetch_or_checkout_performed':False}))
'''
v=json.loads(run(['python3','-c',code],timeout=75));p=Path(__file__).resolve().parents[1]/'evidence/vision-frame-binding-20261107/label-export-source.json';p.write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
