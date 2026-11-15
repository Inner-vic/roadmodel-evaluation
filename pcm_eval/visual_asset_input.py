"""LPAI-only receiver for bounded visual files. No credentials or URLs persisted."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import urlopen


def stage(request):
    selected = request['selected_frames']
    if not selected or len(selected)>2 or len(selected)>request['max_frames']:
        raise ValueError('Only the explicitly approved two-frame validation is supported')
    files=request['files']
    if (len(files)!=2*len(selected) or len({f['path'] for f in files})!=len(files)
            or sum(f['bytes'] for f in files)>min(request['max_bytes'],32*1024*1024)):
        raise ValueError('Visual asset budget or file identity differs')
    root=Path(request['destination']).resolve()
    root.relative_to('/lpai/roadmodel-evaluation/runs')
    root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for entry in files:
        relative=PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or [entry['scene'],entry['frame_index']] not in selected:
            raise ValueError('Visual asset path is outside the frozen scope')
        target=(root/relative).resolve();target.relative_to(root)
        if target.exists():
            raw=target.read_bytes()
        else:
            url=urlsplit(entry['url'])
            if (url.scheme!='https' or not url.hostname or not url.hostname.endswith('.aliyuncs.com')
                    or url.username or url.password or url.port not in (None,443)):
                raise ValueError('Visual object URL is not an approved HTTPS asset endpoint')
            try:
                with urlopen(entry['url'],timeout=45) as response:
                    if response.geturl()!=entry['url']:raise ValueError('Object redirects are not accepted')
                    raw=response.read(entry['bytes']+1)
            except Exception:
                raise ValueError('Visual object read failed; no URL or authorization is logged') from None
            if len(raw)!=entry['bytes'] or hashlib.md5(raw).hexdigest()!=entry['etag'].lower():
                raise ValueError('Visual source object size or ETag differs')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        if len(raw)!=entry['bytes'] or hashlib.md5(raw).hexdigest()!=entry['etag'].lower():
            raise ValueError('Retained visual object differs from this source version')
        manifest.append({k:v for k,v in entry.items() if k!='url'} | {'sha256':hashlib.sha256(raw).hexdigest()})
    index=root/'data_link.json';payload=json.dumps(request['index'],ensure_ascii=False,sort_keys=True).encode()
    if index.exists() and index.read_bytes()!=payload:raise ValueError('Existing visual index differs')
    if not index.exists():index.write_bytes(payload)
    receipt={'state':'ASSET_FILES_VERIFIED','mount_path':str(root),'execution_location':'lpai',
             'workstation_data_used':False,'dataset_mounted':False,'source_ref':request['source_ref'],
             'source_index_sha256':request['source_index_sha256'],'selected_frames':selected,
             'files':manifest,'index_sha256':hashlib.sha256(payload).hexdigest()}
    signature=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    receipt['receipt_sha256']=signature
    (root/'asset-receipt.json').write_text(json.dumps(receipt,indent=2))
    return {'id':signature,'source_ref':request['source_ref'],'version':request['version'],
            'index_path':str(index),'index_sha256':receipt['index_sha256'],
            'source_index_sha256':request['source_index_sha256'],'mount_receipt':receipt,
            'data_roots':[str(root/p) for p in request['roots']],
            'frames_by_head':{'occ':[],'vision':selected},'asset_selection':request['selection']}
