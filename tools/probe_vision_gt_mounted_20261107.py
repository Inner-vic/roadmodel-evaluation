"""Inspect a few mounted scenes and labels only. Never import or run the model."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1]
CODE=r'''import pathlib,json,hashlib,time,collections,os
started=time.monotonic();root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
mounts=[r.split() for r in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if r.split()[4]==str(root)]
assert len(mounts)==1 and 'ro' in mounts[0][5].split(',')
index=root/'data_link_26-10-20-1_train859_eval135_v7.json';raw=index.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c'
d=json.loads(raw);top=list(root.iterdir());assert len(top)<2000
root_json=[]
for p in top:
 if p.is_file() and p.suffix=='.json' and p!=index:
  root_json.append({'path':p.name,'bytes':p.stat().st_size})
selected=[]
for split,positions in [('eval',[0,38,67,134]),('joint_train',[0,858])]:
 scenes=list(d[split].items())
 for pos in positions:
  scene,info=scenes[pos];selected.append((split,scene,info))
rows=[];json_reads=[]
def inspect_json(p):
 size=p.stat().st_size
 if size>512*1024:return {'bytes':size,'state':'NOT_READ_SIZE_LIMIT'}
 value=json.loads(p.read_text());txt=json.dumps(value)
 result={'bytes':size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'top_keys':list(value)[:12] if isinstance(value,dict) else None,
 'classification_key_hits':[k for k in ('label','rainfall_intensity','basic_road_surface','surface_attachment','img_class','img_label_file','vision_eval') if k in txt]}
 if result['classification_key_hits']:result['sample']=str(value)[:1800]
 return result
for split,scene,info in selected:
 scene_root=root/scene;scene_children=list(scene_root.iterdir());frames=info['frames']
 picked=[frames[0],frames[-1]] if len(frames)>1 else frames
 row={'split':split,'scene':scene,'scene_children':[p.name for p in scene_children],
 'scene_json':{},'sampled_frames':[]}
 for base in (scene_root,scene_root/'dataset',scene_root/'labeled'):
  if not base.is_dir():continue
  children=list(base.iterdir());assert len(children)<2000
  for p in children:
   if p.is_file() and p.suffix=='.json':row['scene_json'][str(p.relative_to(root))]=inspect_json(p)
 for frame in picked:
  fid=frame['frame_index'];entry={'frame_index':fid,'index_fields':frame,'directories':{}}
  for pattern in ('dataset','labeled','original'):
   path=scene_root/pattern/str(fid)
   if not path.is_dir():continue
   files=list(path.iterdir());assert len(files)<200
   entry['directories'][pattern]=[p.name for p in files]
   for p in files:
    if p.is_file() and p.suffix=='.json':
     value=inspect_json(p);json_reads.append({'path':str(p.relative_to(root)),**value})
  row['sampled_frames'].append(entry)
 rows.append(row)
proof={'dataset_version':'26-10-20-1','mount':'READ_ONLY','index_sha256':hashlib.sha256(raw).hexdigest(),
 'index_counts':{k:{'scenes':len(v),'frames':sum(len(x.get('frames',[])) for x in v.values())} for k,v in d.items()},
 'top_directory_count':sum(p.is_dir() for p in top),'root_files':[p.name for p in top if p.is_file()],
 'vision_or_label_named_top_dirs':[p.name for p in top if any(s in p.name.lower() for s in ('vision','label','class'))],
 'root_other_json':root_json,'scenes_inspected':rows,'frame_json_read':json_reads,
 'scope':'6 scenes, first/last frame metadata only','inference_frames':0,'dataset_copied':False,'elapsed_seconds':time.monotonic()-started}
print(json.dumps(proof))
'''

def main():
    result=remote('''import sys,pathlib,json,subprocess,shlex,time
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4');cfg=json.loads((release/'configs/evaluation.json').read_text());profile=cfg['profile']
payload=json.load(sys.stdin)
cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=accept-new','-p','30331','-i',profile['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',payload['code']])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=140);assert p.returncode==0,p.stderr[-1600:]
proof=json.loads(p.stdout);proof['checked_at']=time.time()
target=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/vision-gt-audit-20261107');target.mkdir(exist_ok=True)
(target/'mounted-evidence.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
''',json.dumps({'code':CODE}).encode(),timeout=155)
    target=ROOT/'evidence/vision-gt-audit-20261107/mounted-evidence.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'index_counts':result['index_counts'],'root_files':result['root_files'],'vision_or_label_named_top_dirs':result['vision_or_label_named_top_dirs'],
        'scenes':[{k:v for k,v in row.items() if k!='scene_json'} for row in result['scenes_inspected']],
        'scene_json_summary':[{name:{'classification_key_hits':v.get('classification_key_hits'),'bytes':v['bytes']} for name,v in row['scene_json'].items()} for row in result['scenes_inspected']],
        'frame_json_read':result['frame_json_read'],'elapsed_seconds':result['elapsed_seconds']}))

if __name__=='__main__':main()
