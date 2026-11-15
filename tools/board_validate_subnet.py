"""Private board helper: preserve all intermediate checks without downloading them."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def checked_root(value):
    root=Path(value)
    if not root.is_absolute() or '..' in root.parts or root.parent!=Path('/userdata/benchEval/models'):
        raise ValueError('Isolated model root required')
    if not root.name.startswith('pcm-v322-subnets-full-'):
        raise ValueError('Unregistered frame namespace')
    return root


def validate_stage(root,stage):
    root=checked_root(root);directory=root/('sub'+str(stage['stage']));result=[]
    for port in stage['outputs']:
        aliases={port['filename'],Path(port['declared_data_path']).name}
        aliases|={'Frame_0_'+x for x in list(aliases)}
        candidates=[p for p in directory.rglob('*.bin') if p.name in aliases and 'export' not in p.relative_to(directory).parts
                    and not p.is_symlink() and p.stat().st_size==port['nbytes']]
        if not candidates:raise ValueError('Missing fresh output '+port['tensor'])
        hashes=[]
        for p in candidates:
            body=p.read_bytes();hashes.append(hashlib.sha256(body).hexdigest())
            if not np.isfinite(np.frombuffer(body,dtype=port['dtype'])).all():raise ValueError('Non-finite output '+port['tensor'])
        if len(set(hashes))!=1:raise ValueError('Ambiguous fresh output '+port['tensor'])
        target=directory/'export'/port['filename'];target.parent.mkdir(exist_ok=True)
        if target.exists():raise ValueError('Export already exists')
        os.link(candidates[0],target)
        result.append({**port,'stage':stage['stage'],'remote_file':str(target),'sha256':hashes[0],
                       'finite':True,'validation_backend':'board-numpy','fresh_candidate_count':len(candidates)})
    return result


def initialize_frame(root,cache):
    root=checked_root(root);cache=Path(cache)
    if cache.parent!=root.parent or not cache.name.startswith('pcm-v322-fast-cache-'):raise ValueError('Wrong cache namespace')
    root.mkdir(exist_ok=False);(root/'inputs').mkdir()
    for i in range(5):
        dest=root/('sub'+str(i));dest.mkdir();(dest/'output').mkdir();(dest/'export').mkdir()
        for source in (cache/('sub'+str(i))).iterdir():
            if source.name not in ('output','export','data','infer_output','model_info.json'):
                (dest/source.name).symlink_to(source,target_is_directory=source.is_dir())


if __name__=='__main__':
    mode,root=sys.argv[1:3]
    if mode=='initialize':initialize_frame(root,sys.argv[3]);print('{}')
    elif mode=='validate':
        stage=json.loads(Path(sys.argv[3]).read_text());print(json.dumps(validate_stage(root,stage)))
    else:raise ValueError('Unsupported private helper mode')
