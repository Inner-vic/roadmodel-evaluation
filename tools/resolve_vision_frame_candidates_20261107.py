"""Match existing image object metadata, without selecting another image dataset."""
import json,re
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/vision-frame-binding-20261107';OUT.mkdir(exist_ok=True)
read=lambda path:json.loads((ROOT/path).read_text(encoding='utf-8'))
current=read('evidence/requested-dataset/data_link_26-10-20-1_train859_eval135_v7.json')
original=read('evidence/model-evaluation/remaining-heads-20261027/asset-text/DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json')
files=read('evidence/formal-v322-20261029-01/vision-files.json')['files']
cache={x['path']:x for x in read('evidence/requested-dataset/eval135-file-inventory.json')['files']}
old_files={}
for f in files:
    parts=f['path'].split('/')
    if len(parts)>=5 and parts[-3]=='dataset':old_files.setdefault((parts[-4],int(parts[-2])),{})[f['role']]=f
matches=[];different=[];overlap=set()
for scene,info in current['eval'].items():
    for frame in info['frames']:
        fid=frame['frame_index'];pair=old_files.get((scene,fid))
        if not pair:continue
        overlap.add(scene)
        image=cache[scene+'/dataset/'+str(fid)+'/'+str(fid)+'_f_ca_120.jpg'];old=pair['image']
        match={'scene':scene,'frame':frame,'image':image,'original_image':old,'label':pair['label'],
               'source_scene':str(PurePosixPath(old['path']).parent.parent.parent).split('/',1)[1]}
        if image['etag'].lower()!=old['etag'].lower() or image['size']!=old['size']:
            match['requires_pixel_and_timestamp_audit']=True;different.append(match);continue
        matches.append(match)
chosen=(matches or different)[:2]
assert len(chosen)==2
proof={'dataset_version':'26-10-20-1','original_training_task':'8fc0b602-344c-4f73-8fe7-f7b6f77b60aa',
 'original_label_source_version':'26-08-14-0','original_source_role':'candidate GT provenance only; evaluation images stay in submitted dataset',
 'same_scene_overlap':len(overlap),'exact_image_object_metadata_matches':len(matches),'different_image_pairs':len(different),
 'selected':chosen,'selection_limit':2,'verification_state':'METADATA_MATCH_ONLY_REQUIRES_ACTUAL_BYTES_AND_LABEL_SCHEMA',
 'no_scene_tags_used_as_GT':True,'new_inference_frames':0}
(OUT/'candidate-binding.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))
