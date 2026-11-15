"""Prepare bounded, explicitly approved visual asset reads; no images/GT on PC."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcm_eval.dataset_mount import dataset_reference
from pcm_eval.lpai_native_model import select_frames
from head_data_asset import client_for_reference, read_object


def prepare_request(ref, selection, policy, task):
    parsed = dataset_reference(ref)
    if (not policy.get('enabled') or ref != policy['source_ref'] or not selection
            or parsed['version'] != '26-08-14-0' or parsed['namespace'] != 'vc-chassis'):
        raise ValueError('视觉资产读取未获本次数据及小场景授权')
    client, storage = client_for_reference()  # Existing standard SDK identity, no token copying.
    raw, _ = read_object(client, storage, policy['index_relative'], limit=64*1024*1024)
    if hashlib.sha256(raw).hexdigest() != policy['index_sha256']:
        raise ValueError('视觉数据源索引与已核验版本不同')
    index = json.loads(raw)
    frames = [[s,f['frame_index']] for s,info in index['vision_eval'].items() for f in info['frames']]
    selected = select_frames(frames, selection)
    if not selected or len(selected) > policy['max_frames']:
        raise ValueError('本次视觉资产读取超过两帧授权或索引没有可选帧')
    prefix = storage['path'].rstrip('/')+'/'
    files, roots = [], set()
    for scene, fid in selected:
        if PurePosixPath(scene).is_absolute() or '..' in PurePosixPath(scene).parts:
            raise ValueError('Invalid visual scene path')
        legacy = scene.startswith('dataset-')
        root = 'DataReorganized-VisionHead' if legacy else 'DataReorganized-VisionHead-standard'
        roots.add(root)
        for role, style in (('image','original' if legacy else 'dataset'),('label','labeled' if legacy else 'dataset')):
            folder = f'{root}/{scene}/{style}/{fid}/'
            listing = client.list_objects_v2(Bucket=storage['bucket'],Prefix=prefix+folder,MaxKeys=100)
            if listing.get('IsTruncated'):
                raise ValueError('视觉帧目录候选过多')
            rows = [x for x in listing.get('Contents',[]) if
                    (PurePosixPath(x['Key']).name.startswith(str(fid)+'_f_ca_120') and x['Key'].endswith('.jpg')
                     if role=='image' else PurePosixPath(x['Key']).name.startswith('img_label') and x['Key'].endswith('.json'))]
            if len(rows)!=1:
                raise ValueError('视觉图片或逐帧GT不唯一，停止资产读取')
            row=rows[0];relative=row['Key'][len(prefix):]
            url=client.generate_presigned_url('get_object',Params={'Bucket':storage['bucket'],'Key':row['Key']},ExpiresIn=600)
            if url.startswith('http://'):url='https://'+url[7:]
            files.append({'path':relative,'role':role,'scene':scene,'frame_index':fid,
                          'bytes':row['Size'],'etag':row['ETag'].strip('"'),'url':url})
    if sum(f['bytes'] for f in files)>policy['max_bytes']:
        raise ValueError('视觉图片和GT超过本次文件预算')
    subset={key:{} for key in index}
    for scene,fid in selected:
        values=[f for f in index['vision_eval'][scene]['frames'] if f['frame_index']==fid]
        if len(values)!=1:raise ValueError('视觉索引帧不唯一')
        subset['vision_eval'].setdefault(scene,{'scene_name':scene,'frames':[]})['frames'].append(values[0])
    return {'source_ref':ref,'version':parsed['version'],'selection':selection,'selected_frames':selected,
            'source_index_sha256':policy['index_sha256'],'index':subset,'files':files,'roots':sorted(roots),
            'destination':'/lpai/roadmodel-evaluation/runs/'+task+'-vision-asset',
            'max_frames':policy['max_frames'],'max_bytes':policy['max_bytes']}


if __name__ == '__main__':
    import sys
    payload=json.load(sys.stdin)
    # stdout is a private subprocess pipe consumed by the parent Agent. The
    # request contains short-lived object URLs and must never be logged.
    print(json.dumps(prepare_request(payload['ref'],payload['selection'],payload['policy'],payload['task'])))
