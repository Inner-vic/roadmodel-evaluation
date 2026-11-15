"""Compare two real frames with a re-encoding-only control. No model inference."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_space_transport import run
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/camera-GT-mapping-20261107'
CODE=r'''import pathlib,json,sys,hashlib,io,numpy as np
from PIL import Image,JpegImagePlugin,ImageDraw
import cv2
payload=json.load(sys.stdin);root=pathlib.Path('/mnt/datasets/smartchassisdata-v3/26-10-20-1')
original=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029/vision-input')
rows=[];tiles=[]
def stats(a,b):
 diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
 return {'mean_absolute_error':float(diff.mean()),'max_absolute_error':int(diff.max()),'equal_fraction':float((diff==0).mean())}
for c in payload['selected']:
 current=root/c['image']['path'];old=original/c['original_image']['path']
 with Image.open(current) as im:
  q=im.quantization;sampling=JpegImagePlugin.get_sampling(im);now=im.convert('RGB');current_format={'mode':im.mode,'size':list(im.size),'subsampling':sampling,'progressive':bool(im.info.get('progressive')),'qtables':q,'exif_orientation':im.getexif().get(274)}
 with Image.open(old) as im:
  previous=im.convert('RGB');old_format={'mode':im.mode,'size':list(im.size),'subsampling':JpegImagePlugin.get_sampling(im),'progressive':bool(im.info.get('progressive')),'qtables':im.quantization,'exif_orientation':im.getexif().get(274)}
 a,b=np.asarray(now),np.asarray(previous)
 control=io.BytesIO();previous.save(control,format='JPEG',qtables=q,subsampling=sampling,optimize=False)
 reencoded=Image.open(io.BytesIO(control.getvalue())).convert('RGB');r=np.asarray(reencoded)
 control_self=io.BytesIO();now.save(control_self,format='JPEG',qtables=q,subsampling=sampling,optimize=False)
 self_arr=np.asarray(Image.open(io.BytesIO(control_self.getvalue())).convert('RGB'))
 ag=cv2.cvtColor(cv2.resize(a,(960,540)),cv2.COLOR_RGB2GRAY);bg=cv2.cvtColor(cv2.resize(b,(960,540)),cv2.COLOR_RGB2GRAY)
 orb=cv2.ORB_create(nfeatures=1800);ka,da=orb.detectAndCompute(ag,None);kb,db=orb.detectAndCompute(bg,None)
 good=[]
 if da is not None and db is not None:
  good=[m for m,n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db,da,k=2) if m.distance<.7*n.distance]
 motion={}
 if len(good)>=8:
  src=np.float32([kb[m.queryIdx].pt for m in good]);dst=np.float32([ka[m.trainIdx].pt for m in good])
  H,mask=cv2.findHomography(src,dst,cv2.RANSAC,2.0)
  if mask is not None:
   keep=mask.ravel().astype(bool);delta=dst[keep]-src[keep]
   motion={'feature_pairs':len(good),'inliers':int(keep.sum()),'median_dx_at_960':float(np.median(delta[:,0])),'median_dy_at_960':float(np.median(delta[:,1])),
     'median_displacement_at_960':float(np.median(np.linalg.norm(delta,axis=1))),'homography':H.tolist()}
 comparison={'scene':c['scene'],'frame_index':c['frame']['frame_index'],'current_JPEG':current_format,'old_JPEG':old_format,
  'same_quantization_tables':q==old_format['qtables'],'raw_pair_difference':stats(a,b),'reencode_only_control_error':stats(b,r),
  'current_self_reencode_error':stats(a,self_arr),'reencoded_old_vs_current':stats(a,r),'feature_motion':motion}
 rows.append(comparison)
 canvas=Image.new('RGB',(1536,400),'white');draw=ImageDraw.Draw(canvas)
 for i,(label,image) in enumerate((('Original annotated image',previous),('Current submitted image',now),('Original re-encoded with current JPEG tables',reencoded))):
  draw.text((i*512+4,5),'Frame '+str(c['frame']['frame_index'])+' - '+label,fill='black')
  canvas.paste(image.resize((512,288)),(i*512,30))
 tiles.append(canvas)
target=pathlib.Path('/lpai/roadmodel-evaluation/camera-GT-mapping-20261107');target.mkdir(exist_ok=True,parents=True)
sheet=Image.new('RGB',(1536,400*len(tiles)),'white')
for i,t in enumerate(tiles):sheet.paste(t,(0,i*400))
sheet.save(target/'JPEG-comparison.png')
proof={'scope':'same two candidate frames only','rows':rows,'new_inference_frames':0,'evaluation_dataset_version':'26-10-20-1'}
(target/'JPEG-comparison.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
payload=(ROOT/'evidence/vision-frame-binding-20261107/candidate-binding.json').read_bytes()
v=json.loads(run(['python3','-c',CODE],payload,timeout=100));(OUT/'JPEG-comparison.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
shot=run(['python3','-c',"import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/lpai/roadmodel-evaluation/camera-GT-mapping-20261107/JPEG-comparison.png').read_bytes())"],timeout=30)
(OUT/'JPEG-comparison.png').write_bytes(shot)
print(json.dumps({'rows':[{k:r[k] for k in ('scene','frame_index','same_quantization_tables','raw_pair_difference','reencode_only_control_error','current_self_reencode_error','reencoded_old_vs_current','feature_motion')} for r in v['rows']],'new_inference_frames':0}))
