"""Stream one frozen model batch from workstation cache; adapt GT into new bytes."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

BASE = Path('/mnt/data/developer/pcm-roadmodel-eval')
TASK = BASE / 'formal-runs/pcm-formal-v322-20261029-01'
DATA = BASE / 'datasets/smartchassisdata-v3/26-10-20-1'
request = json.loads((Path(sys.argv[3]) if len(sys.argv) > 3 else TASK / 'request.json').read_text())
start, count = map(int, sys.argv[1:3])
selected = request['frames'][start:start + count]
assert 0 <= start < 8337 and 1 <= count <= 100 and len(selected) == count
data = json.loads((DATA / 'data_link_26-10-20-1_train859_eval135_v7.json').read_text())
subset = {k: {} for k in data}
files = []
adaptations = []
added = set()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
    def add(name, body, **extra):
        if name in added: return
        added.add(name)
        info = tarfile.TarInfo(name); info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
        files.append({'path': name, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body), **extra})
    for scene, fr in selected:
        item = dict(next(f for f in data['eval'][scene]['frames'] if f['frame_index'] == fr))
        folder = DATA / scene / 'dataset' / str(fr)
        names = list(folder.iterdir())
        lidar = next(p for p in names if p.name.startswith('lidar_original') and p.suffix == '.npy')
        image = next(p for p in names if p.name.startswith(str(fr) + '_f_ca_120') and p.suffix == '.jpg')
        for path in [lidar, image, DATA / scene / 'dataset/sensor_param.json']:
            add(path.relative_to(DATA).as_posix(), path.read_bytes(), source='verified_workstation_cache')
        raw_path = folder / item['occ_file']
        body = raw_path.read_bytes()
        a = np.load(io.BytesIO(body), allow_pickle=False)
        assert a.ndim == 2 and a.shape[1] == 5 and np.isfinite(a).all() and np.equal(a, np.rint(a)).all()
        assert ((a[:, :3] >= 0) & (a[:, :3] < [800, 48, 200])).all()
        adapted = a.copy(); adapted[:, 2] -= 50
        keep = ((adapted[:, :3] >= 0) & (adapted[:, :3] < [400, 48, 100])).all(1)
        adapted = adapted[keep]
        stream = io.BytesIO(); np.save(stream, adapted, allow_pickle=False)
        name = 'adapted_' + raw_path.name
        item['occ_file'] = name
        add(scene + '/dataset/' + str(fr) + '/' + name, stream.getvalue(), source='ROI_aligned_GT', original_sha256=hashlib.sha256(body).hexdigest())
        adaptations.append({'scene': scene, 'frame_index': fr, 'source_rows': len(a), 'kept_rows': len(adapted), 'index_shift': [0, 0, -50]})
        subset['eval'].setdefault(scene, dict(data['eval'][scene], frames=[]))['frames'].append(item)
    subset['joint_eval'] = subset['eval']
    add('batch-data-link.json', json.dumps(subset).encode())
    add('data-binding.json', json.dumps({'selected_frames': selected, 'files': files, 'GT_adaptations': adaptations,
             'source_data_link_sha256': request['data_link_sha256'], 'new_cloud_downloads': 0}).encode())
