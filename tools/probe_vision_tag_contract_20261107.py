"""Inspect label scope and the exact previous two frames; no model execution."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1]
CODE=r'''import json,pathlib,hashlib,ast
root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
index=root/'data_link_26-10-20-1_train859_eval135_v7.json';d=json.loads(index.read_text())
scenes=list(d['eval']);rows=[]
for offset in (0,38,67,134):
 scene=scenes[offset];path=root/scene/'tags.json';raw=path.read_bytes();tag=json.loads(raw)
 stats=tag.get('label_stats',{});hist=stats.get('label_counts',{});vlm=tag.get('vlm_tags_list',{})
 differences=[];uniform={}
 for task,counts in hist.items():
  positive=[(int(k),v) for k,v in counts.items() if v]
  if len(positive)==1:
   uniform[task]=positive[0][0]
   if task in vlm and vlm[task]!=positive[0][0]:differences.append({'task':task,'scene_vlm':vlm[task],'frame_histogram_only_class':positive[0][0],'frame_count':positive[0][1]})
  elif len(positive)>1:differences.append({'task':task,'scene_vlm':vlm.get(task),'nonuniform_frame_counts':dict(positive)})
 row={'scene':scene,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'gt_version':tag.get('gt_version'),
  'vlm_tags_list':vlm,'label_stats':{k:stats.get(k) for k in ('scene','vla_supplier_labeled','label_counts')},
  'scene_label_vs_frame_histogram_differences':differences,'vision_frames_in_index':len(d['vision_eval'][scene]['frames'])}
 if offset==38:
  row['last_two_frame_check']=[]
  for fid in (29,30):
   folder=root/scene/'dataset'/str(fid);files=list(folder.iterdir())
   row['last_two_frame_check'].append({'frame_index':fid,'image_120_exists':(folder/(str(fid)+'_f_ca_120.jpg')).is_file(),
     'listed_as_supplier_labeled':fid in stats.get('vla_supplier_labeled',[]),'frame_label_files':[p.name for p in files if p.suffix=='.json'],
     'index_img_label_file':next(f for f in d['eval'][scene]['frames'] if f['frame_index']==fid).get('img_label_file')})
 rows.append(row)
source=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/source/dataset/dataset.py');text=source.read_text();tree=ast.parse(text)
methods={}
for node in tree.body:
 if isinstance(node,ast.ClassDef) and node.name in ('SmartChassisDatasetV6','SmartChassisDatasetV7'):
  for method in node.body:
   if isinstance(method,ast.FunctionDef) and method.name in ('_load_img_label','_init_vision_data','_build_vision_index','_pair_frame_file_names'):
    methods[node.name+'.'+method.name]=ast.get_source_segment(text,method)
print(json.dumps({'rows':rows,'actual_native_source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'methods':methods},
 'inference_frames':0,'dataset_copied':False,'whole_dataset_evaluation':False}))
'''

result=remote('''import pathlib,sys,json,subprocess,shlex,time
c=json.loads(pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4/configs/evaluation.json').read_text())
code=json.load(sys.stdin)['code'];cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=12','-p','30331','-i',c['profile']['identity_file'],'root@ssh-d.lpai.lixiang.com',shlex.join(['python3','-c',code])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=70);assert p.returncode==0,p.stderr[-1200:]
value=json.loads(p.stdout);value['checked_at']=time.time()
pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/vision-gt-audit-20261107/tag-contract.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
''',json.dumps({'code':CODE}).encode(),timeout=85)
(ROOT/'evidence/vision-gt-audit-20261107/tag-contract.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'scenes':[{'scene':r['scene'],'vlm_task_count':len(r['vlm_tags_list']),'frame_stats_task_count':len(r['label_stats']['label_counts']),
 'differences':r['scene_label_vs_frame_histogram_differences'],'last_two_frame_check':r.get('last_two_frame_check')} for r in result['rows']],
 'source_sha256':result['actual_native_source']['sha256'],'native_methods':list(result['actual_native_source']['methods']),'inference_frames':0}))
