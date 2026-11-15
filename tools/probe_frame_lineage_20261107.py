import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-frame-binding-20261107'
CODE=r'''import pathlib,json,sys,hashlib,numpy as np,ast
from PIL import Image
p=json.load(sys.stdin);root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1');scene=p['selected'][0]['scene'];base=root/scene/'dataset'
rows=[];candidate=p['selected'][0];old=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029/vision-input')/candidate['original_image']['path'];oldarr=np.asarray(Image.open(old).convert('RGB').resize((480,270)))
for fid in range(15,25):
 path=base/str(fid)/(str(fid)+'_f_ca_120.jpg')
 row={'frame_index':fid,'exists':path.is_file()}
 if path.is_file():
  arr=np.asarray(Image.open(path).convert('RGB').resize((480,270)));row.update(md5=hashlib.md5(path.read_bytes()).hexdigest(),difference_to_old18=float(np.abs(oldarr.astype(np.int16)-arr.astype(np.int16)).mean()))
 rows.append(row)
metadata={}
for name in ('all_transform_matrices.json','refined_cam0_matrices.json','refined_transform_camera.json'):
 obj=json.loads((base/name).read_text());metadata[name]={}
 for fid in (17,18,19,23):
  item=obj.get('frame_'+str(fid),{});selected={k:v for k,v in item.items() if k not in ('transformation_matrix','lidar_to_image_transform','interp_alpha') and not isinstance(v,list)}
  metadata[name][str(fid)]=selected
oldroot=pathlib.Path('/mnt/datasets/smartchassisdata-v3/1-0-10');old_listing=[x.name for x in oldroot.iterdir()];assert len(old_listing)<1000
source=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/source')
hits=[]
for name in ('tools/dataset_process/1-DATASET_MANAGEMENT/generate_datalink_v4.py','tools/dataset_process/Occ&dataLink/smartChassisDatasetProcess_v3.py','tools/tools_eval/vision-dataset-iterator/step5_generate_datalink.py'):
 path=source/name
 if not path.is_file():continue
 lines=path.read_text(errors='replace').splitlines()
 for i,line in enumerate(lines):
  if any(word in line for word in ('frame_index =','original_name','img_label','image_timestamp','timestamp_offset','vision_eval')):
   hits.append({'file':name,'line':i+1,'code':line.strip()[:220]})
print(json.dumps({'nearby_images':rows,'metadata':metadata,'other_mount_root_listing':old_listing,'source_hints':hits[:70],'new_inference_frames':0}))
'''
v=json.loads(run(['python3','-c',CODE],(OUT/'candidate-binding.json').read_bytes(),timeout=100))
(OUT/'frame-lineage.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(v))
