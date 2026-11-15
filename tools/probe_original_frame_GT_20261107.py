"""Read two historical frame label files and submitted images; never score them."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-frame-binding-20261107'
CODE=r'''import pathlib,json,sys,hashlib,numpy as np
from PIL import Image
p=json.load(sys.stdin);root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
info=[r.split() for r in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if r.split()[4]==str(root)]
assert len(info)==1 and 'ro' in info[0][5].split(',')
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
caches=[base/'formal-20261029/vision-input',base/'heads-20261027/input',base/'coverage-20261029/input',pathlib.Path('/mnt/datasets/smartchassisdata-v3/1-0-10')]
rows=[]
for row in p['selected']:
 source=root/row['image']['path'];assert source.is_file();raw=source.read_bytes()
 current=Image.open(source).convert('RGB');cur=np.asarray(current)
 item={'scene':row['scene'],'frame':row['frame'],'submitted_image':{'path':str(source),'bytes':len(raw),'md5':hashlib.md5(raw).hexdigest(),'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(cur.shape),'pixel_sha256':hashlib.sha256(cur.tobytes()).hexdigest()},'historical_candidates':[]}
 for cache in caches:
  label=cache/row['label']['path'];image=cache/row['original_image']['path']
  candidate={'root':str(cache),'label_exists':label.is_file(),'image_exists':image.is_file()}
  if label.is_file():
   body=label.read_bytes();obj=json.loads(body);candidate['label']={'path':str(label),'sha256':hashlib.sha256(body).hexdigest(),'md5':hashlib.md5(body).hexdigest(),'content':obj}
   assert hashlib.md5(body).hexdigest()==row['label']['etag'].lower(),'Historic label differs from verified object inventory'
  if image.is_file():
   body=image.read_bytes();arr=np.asarray(Image.open(image).convert('RGB'));candidate['image']={'path':str(image),'shape':list(arr.shape),'md5':hashlib.md5(body).hexdigest(),'sha256':hashlib.sha256(body).hexdigest(),'pixel_sha256':hashlib.sha256(arr.tobytes()).hexdigest()}
   assert hashlib.md5(body).hexdigest()==row['original_image']['etag'].lower(),'Historic image differs from verified object inventory'
   if arr.shape==cur.shape:
    diff=np.abs(arr.astype(np.int16)-cur.astype(np.int16));candidate['pixel_comparison']={'equal':bool(np.array_equal(arr,cur)),'mean_absolute_difference':float(diff.mean()),'max_absolute_difference':int(diff.max())}
  item['historical_candidates'].append(candidate)
 rows.append(item)
metadata=[]
scene=p['selected'][0]['scene']
for name in ('all_transform_matrices.json','refined_cam0_matrices.json','refined_transform_camera.json'):
 path=root/scene/'dataset'/name
 if not path.is_file():continue
 obj=json.loads(path.read_text());metadata.append({'file':name,'type':type(obj).__name__,'keys':list(obj)[:10] if isinstance(obj,dict) else None,
 'first_value_structure':{k:type(v).__name__ for k,v in next(iter(obj.values())).items()} if isinstance(obj,dict) and isinstance(next(iter(obj.values())),dict) else None,
 'image_name_strings':[]})
 def walk(x):
  if len(metadata[-1]['image_name_strings'])>=6:return
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,str) and ('f_ca_120' in v or v.endswith('.jpg')):metadata[-1]['image_name_strings'].append({'key':k,'value':v})
    elif isinstance(v,(dict,list)):walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(obj)
print(json.dumps({'rows':rows,'metadata_structure':metadata,'evaluation_dataset_version':'26-10-20-1','new_inference_frames':0}))
'''
value=json.loads(run(['python3','-c',CODE],(OUT/'candidate-binding.json').read_bytes(),timeout=100))
(OUT/'original-GT-probe.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(value))
