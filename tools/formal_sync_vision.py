"""Prepare the exact independent full vision validation set directly in its space."""
import concurrent.futures
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from pcm_eval.common import load_json, save_json, utc_now
from model_space_transport import run, put
from head_data_asset import client_for_reference

TASK = ROOT / 'evidence/formal-v322-20261029-01'
SPACE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029'
DEST = SPACE + '/vision-input'


def state(phase, **fields):
    d = {'state': phase, 'updated_at': utc_now(), **fields}
    save_json(TASK / 'vision-data-state.json', d, overwrite=True)
    code = 'import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(".tmp");q.write_bytes(sys.stdin.buffer.read());q.replace(p)'
    run(['python3', '-c', code, SPACE + '/vision-data-state.json'], json.dumps(d).encode())


def main():
    index = ROOT / 'evidence/model-evaluation/remaining-heads-20261027/asset-text/DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json'
    data = load_json(index)
    client, storage = client_for_reference()
    prefix = storage['path'].rstrip('/') + '/'
    inventory = TASK / 'vision-files.json'
    if inventory.exists(): rows = load_json(inventory)['files']
    else:
        state('INDEXING', expected_frames=11543)
        def listing(relative):
            files = []
            for page in client.get_paginator('list_objects_v2').paginate(Bucket=storage['bucket'], Prefix=prefix + relative):
                files.extend(page.get('Contents', []))
                if len(files) > 20000: raise ValueError('Unexpected scene inventory size')
            return files
        def inspect(pair):
            scene, sc = pair
            legacy = scene.startswith('dataset-')
            root = 'DataReorganized-VisionHead' if legacy else 'DataReorganized-VisionHead-standard'
            folder = root + '/' + scene + '/'
            labels = listing(folder + ('labeled/' if legacy else 'dataset/'))
            images = listing(folder + 'original/') if legacy else labels
            out = []
            for fr in sc['frames']:
                part = '/' + str(fr['frame_index']) + '/'
                for objects, kind in [(labels, 'label'), (images, 'image')]:
                    candidates = [r for r in objects if part in r['Key'][len(prefix + folder):] and
                                  (Path(r['Key']).name.startswith('img_label') and r['Key'].endswith('.json') if kind == 'label' else
                                   Path(r['Key']).name.startswith(str(fr['frame_index']) + '_f_ca_120') and r['Key'].endswith('.jpg'))]
                    assert len(candidates) == 1, (scene, fr['frame_index'], kind, len(candidates))
                    r = candidates[0]
                    out.append({'path': r['Key'][len(prefix):], 'size': r['Size'], 'etag': r['ETag'].strip('"'), 'role': kind})
            return out
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows = [row for group in pool.map(inspect, data['vision_eval'].items()) for row in group]
        assert len(rows) == 11543 * 2 and len({r['path'] for r in rows}) == len(rows)
        assert sum(r['size'] for r in rows) < 35 * 1024 ** 3, 'Preserve space staging reserve'
        save_json(inventory, {'files': rows, 'expected_frames': 11543, 'bytes': sum(r['size'] for r in rows)})
    put(index, DEST + '/full-data-link.json')
    checker = (ROOT / 'tools/check_required_dataset.py').read_text().replace("'/mnt/data/developer/pcm-roadmodel-eval'", repr(SPACE))
    worker = (ROOT / 'tools/workstation_download_worker.py').read_text().replace("'/home/lixiang/developer/pcm-roadmodel-eval'", repr(SPACE))
    for name, body in [('vision-checker.py', checker), ('vision-downloader.py', worker)]:
        path = TASK / name; path.write_text(body, encoding='utf-8'); put(path, SPACE + '/code/' + name)
    # Seed already evaluated inputs from both existing space caches, without redownloading.
    code = """import pathlib,json,sys,hashlib
d=json.load(sys.stdin);root=pathlib.Path(d['root']);sources=[pathlib.Path(p) for p in d['sources']];n=0
for r in d['files']:
 p=root/r['path']
 if p.exists():continue
 for source in sources:
  q=source/r['path']
  if q.is_file() and q.stat().st_size==r['size'] and hashlib.md5(q.read_bytes()).hexdigest()==r['etag'].lower():
   p.parent.mkdir(parents=True,exist_ok=True);p.hardlink_to(q);n+=1;break
print(json.dumps({'seeded_files':n}))
"""
    parent = SPACE.rsplit('/formal-20261029', 1)[0]
    print(run(['python3', '-c', code], json.dumps({'root': DEST, 'files': rows, 'sources': [parent + '/heads-20261027/input', parent + '/coverage-20261029/input']}).encode(), timeout=180).decode(), flush=True)
    pre = json.loads(run(['python3', SPACE + '/code/vision-checker.py'], json.dumps({'root': DEST, 'files': rows}).encode(), timeout=180))
    assert not pre['conflicts']
    verified = set(pre['verified_paths']); pending = [r for r in rows if r['path'] not in verified]
    done = len(verified)
    state('DOWNLOADING', completed_files=done, total_files=len(rows), expected_frames=11543, reused_files=done)
    for start in range(0, len(pending), 120):
        files = []
        for r in pending[start:start + 120]:
            u = client.generate_presigned_url('get_object', Params={'Bucket': storage['bucket'], 'Key': prefix + r['path']}, ExpiresIn=3600)
            files.append(dict(r, url='https://' + u[7:] if u.startswith('http://') else u))
        result = json.loads(run(['python3', SPACE + '/code/vision-downloader.py'], json.dumps({'root': DEST, 'files': files, 'workers': 16}).encode(), timeout=600))
        assert result['state'] == 'complete'
        done += len(files)
        state('DOWNLOADING', completed_files=done, total_files=len(rows), expected_frames=11543, reused_files=len(verified))
        print(json.dumps({'vision_files': done, 'total': len(rows)}), flush=True)
    state('READY', completed_files=done, total_files=len(rows), expected_frames=11543, verified_files=done)


if __name__ == '__main__':
    try: main()
    except Exception as e:
        state('NEEDS_REVIEW', error_type=type(e).__name__, error=str(e)[:400]); raise
