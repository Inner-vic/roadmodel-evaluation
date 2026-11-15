"""Bind original per-image annotations to submitted images with exact identity.

Scene tags, class histograms, frame-number equality and nearby timestamps are
never evidence of an image's classification label. All inputs are local files
inside approved roots; this module performs no downloads or inference.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

from .common import digest,load_json,save_json,json_digest


class BindingRejected(ValueError):
    def __init__(self, code, detail):
        self.code,self.detail=code,detail
        super().__init__(code)


def _inside(path, roots):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise BindingRejected('SYMLINK_NOT_ALLOWED',{'path':str(path)})
    target=path.resolve()
    if not any(target.is_relative_to(Path(root).resolve()) for root in roots):
        raise BindingRejected('UNAPPROVED_SOURCE_ROOT',{'path':str(target)})
    if not target.is_file():raise BindingRejected('FILE_MISSING',{'path':str(target)})
    return target


def _rgb_identity(path):
    from PIL import Image
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        return {'size':list(rgb.size),'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}


def prepare(*, dataset_root,index_path,index_sha256,candidates,approved_GT_roots,required_tasks,output):
    """Publish a derived vision index only after every candidate is verified.

    candidates are frozen mappings to original label JSON/image/hash and task
    provenance. Image files always remain in dataset_root. The submitted index
    is never rewritten; only the JSON image key is renamed after exact RGB
    identity is proven. Label values are preserved verbatim.
    """
    root=Path(dataset_root).resolve();index=_inside(index_path,[root])
    if digest(index)!=index_sha256:raise BindingRejected('INDEX_CHANGED',{})
    if not candidates or len(candidates)>20:raise BindingRejected('INVALID_VALIDATION_SCOPE',{})
    data=load_json(index);accepted=[];identities=set()
    for candidate in candidates:
        scene,fid=candidate['scene'],candidate['frame_index'];identity=(scene,str(fid))
        if identity in identities:raise BindingRejected('DUPLICATE_FRAME',{'scene':scene,'frame_index':fid})
        identities.add(identity)
        frames=data.get('eval',{}).get(scene,{}).get('frames',[])
        matches=[f for f in frames if str(f['frame_index'])==str(fid)]
        if len(matches)!=1:raise BindingRejected('NOT_IN_SUBMITTED_VALIDATION',{'scene':scene,'frame_index':fid})
        image=_inside(candidate['image'],[root]);source_image=_inside(candidate['source_image'],approved_GT_roots)
        label=_inside(candidate['source_label'],approved_GT_roots)
        expected_image=(root/scene/'dataset'/str(fid)/(str(fid)+'_f_ca_120.jpg')).resolve()
        if image!=expected_image:raise BindingRejected('SUBMITTED_IMAGE_PATH_MISMATCH',{'scene':scene,'frame_index':fid})
        for name,path in (('image',image),('source_image',source_image),('source_label',label)):
            if digest(path)!=candidate[name+'_sha256']:raise BindingRejected('SOURCE_CHANGED',{'role':name})
        source=candidate.get('origin',{})
        if not source.get('task_uid') or not source.get('dataset_ref'):
            raise BindingRejected('LABEL_PROVENANCE_MISSING',{})
        current_rgb,original_rgb=_rgb_identity(image),_rgb_identity(source_image)
        if current_rgb!=original_rgb:
            raise BindingRejected('SOURCE_IMAGE_DIFFERS',{'scene':scene,'frame_index':fid,
                'submitted':current_rgb,'original':original_rgb,'origin':source})
        annotations=load_json(label);entry=annotations.get(source_image.name)
        if not isinstance(entry,dict) or not isinstance(entry.get('label'),dict) or not entry['label']:
            raise BindingRejected('NO_EXACT_IMAGE_LABEL',{'image_key':source_image.name})
        values=entry['label']
        if any(type(v) is not int or v<0 for v in values.values()):
            raise BindingRejected('INVALID_CLASSIFICATION_LABELS',{})
        missing=sorted(set(required_tasks)-set(values))
        if missing:raise BindingRejected('REQUIRED_TASK_LABELS_MISSING',{'tasks':missing})
        accepted.append({'candidate':candidate,'frame':deepcopy(matches[0]),'image':image,'label_values':deepcopy(values),'pixel_identity':current_rgb})
    # Validation finishes before any derived index or annotation is created.
    out=Path(output).resolve()
    if out.exists():raise BindingRejected('NEW_BINDING_DIRECTORY_REQUIRED',{})
    if out.is_relative_to(root):raise BindingRejected('INPUT_DATASET_MUST_REMAIN_UNCHANGED',{})
    out.mkdir(parents=True)
    subset={k:{} for k in data};subset.setdefault('vision_eval',{});proofs=[]
    for i,row in enumerate(accepted):
        candidate=row['candidate'];scene=candidate['scene'];fid=candidate['frame_index']
        label=save_json(out/'labels'/(str(i)+'.json'),{row['image'].name:{'label':row['label_values']}})
        frame=row['frame'];frame.update(img_file=row['image'].name,img_label_file=str(label))
        subset['vision_eval'].setdefault(scene,{'scene_name':scene,'frames':[]})['frames'].append(frame)
        proofs.append({**candidate,'pixel_identity':row['pixel_identity'],'adapted_label':str(label),'adapted_label_sha256':digest(label),'label_values_unchanged':True})
    link=save_json(out/'vision-index.json',subset)
    receipt={'schema':'roadmodel.vision-GT-binding/v1','state':'VERIFIED','index_source_sha256':index_sha256,
        'derived_index':str(link),'derived_index_sha256':digest(link),'dataset_root':str(root),'frames':proofs,
        'scene_tags_used':False,'new_inference_frames':0}
    receipt['binding_sha256']=json_digest(receipt);save_json(out/'binding.json',receipt)
    return receipt


def verify(dataset,binding):
    """Recheck the frozen binding before planning or launching its native head."""
    path=Path(binding['path'])
    if digest(path)!=binding['sha256']:raise BindingRejected('BINDING_CHANGED',{})
    receipt=load_json(path)
    if binding.get('manifest') is not None and receipt!=binding['manifest']:
        raise BindingRejected('BINDING_MANIFEST_CHANGED',{})
    if (receipt.get('state')!='VERIFIED' or receipt.get('schema')!='roadmodel.vision-GT-binding/v1'
            or receipt['binding_sha256']!=json_digest({k:v for k,v in receipt.items() if k!='binding_sha256'})
            or receipt['scene_tags_used'] is not False):raise BindingRejected('INVALID_BINDING',{})
    root=Path(dataset['mount_receipt']['mount_path']).resolve()
    if Path(receipt['dataset_root']).resolve()!=root or receipt['index_source_sha256']!=dataset['index_sha256']:
        raise BindingRejected('DIFFERENT_SUBMITTED_DATASET',{})
    index=_inside(dataset['index_path'],[root])
    if digest(index)!=dataset['index_sha256']:raise BindingRejected('INDEX_CHANGED',{})
    derived=_inside(receipt['derived_index'],[path.parent])
    if digest(derived)!=receipt['derived_index_sha256']:raise BindingRejected('DERIVED_INDEX_CHANGED',{})
    data=load_json(derived);frames=[];original=load_json(index)
    for row in receipt['frames']:
        scene,fid=row['scene'],row['frame_index']
        if not any(str(f['frame_index'])==str(fid) for f in original.get('eval',{}).get(scene,{}).get('frames',[])):
            raise BindingRejected('NOT_IN_SUBMITTED_VALIDATION',{})
        for name in ('image','source_image','source_label'):
            if digest(Path(row[name]))!=row[name+'_sha256']:raise BindingRejected('SOURCE_CHANGED',{'role':name})
        image=_inside(row['image'],[root]);expected=(root/scene/'dataset'/str(fid)/(str(fid)+'_f_ca_120.jpg')).resolve()
        if image!=expected or _rgb_identity(image)!=row['pixel_identity'] or _rgb_identity(Path(row['source_image']))!=row['pixel_identity']:
            raise BindingRejected('SOURCE_IMAGE_DIFFERS',{})
        adapted=_inside(row['adapted_label'],[path.parent])
        if digest(adapted)!=row['adapted_label_sha256']:raise BindingRejected('ADAPTED_LABEL_CHANGED',{})
        source=load_json(row['source_label']).get(Path(row['source_image']).name,{})
        if load_json(adapted)!={image.name:{'label':source.get('label')}}:
            raise BindingRejected('LABEL_VALUES_CHANGED',{})
        found=[f for f in data['vision_eval'].get(scene,{}).get('frames',[]) if str(f['frame_index'])==str(fid)]
        if len(found)!=1 or found[0].get('img_label_file')!=str(adapted) or found[0].get('img_file')!=image.name:
            raise BindingRejected('DERIVED_FRAME_BINDING_CHANGED',{})
        frames.append([scene,fid])
    declared=[[s,f['frame_index']] for s,v in data['vision_eval'].items() for f in v['frames']]
    if declared!=frames or len(set(map(tuple,frames)))!=len(frames):raise BindingRejected('DERIVED_FRAME_SCOPE_CHANGED',{})
    return data,frames


def attach(dataset,binding):
    _,frames=verify(dataset,binding)
    value=deepcopy(dataset);value['frames_by_head']['vision']=frames
    value['vision_gt_binding']={**deepcopy(binding),'manifest':load_json(binding['path'])}
    if value.get('frame_manifest_sha256'):
        value['original_frame_manifest_sha256']=value['frame_manifest_sha256']
        value['frame_manifest_sha256']=json_digest(value['frames_by_head'])
    return value


def declared_frames(dataset,binding):
    """Controller-side metadata validation; LPAI file paths are not local paths.

    The mount-side attach() returns this frozen manifest. Native run()/launch
    must still call verify() in LPAI before inference.
    """
    receipt=binding.get('manifest',{})
    if (receipt.get('state')!='VERIFIED' or receipt.get('schema')!='roadmodel.vision-GT-binding/v1'
            or receipt.get('binding_sha256')!=json_digest({k:v for k,v in receipt.items() if k!='binding_sha256'})
            or receipt.get('scene_tags_used') is not False
            or receipt.get('index_source_sha256')!=dataset['index_sha256']
            or receipt.get('dataset_root')!=dataset['mount_receipt']['mount_path']):
        raise BindingRejected('INVALID_FROZEN_BINDING_MANIFEST',{})
    frames=[[r['scene'],r['frame_index']] for r in receipt['frames']]
    if not frames or len(frames)!=len(set((s,str(f)) for s,f in frames)):
        raise BindingRejected('INVALID_BOUND_FRAME_MANIFEST',{})
    return frames
