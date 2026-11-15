"""Select at most 16 vision frames by small GT metadata, never bulk image data."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json
from head_data_asset import client_for_reference, read_object

OUT = Path('evidence/coverage-supplement-20261029')
SOURCE = Path('evidence/model-evaluation/remaining-heads-20261027/asset-text/DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json')


def main():
    spec = load_json('configs/report_template.json')['model_accuracy']['vision_major_classes']
    desired = {(r['task'], int(r['class_id'])) for r in spec}
    data = load_json(SOURCE)
    scenes = list(data['vision_eval'].items())
    # Stratified over the source ordering; metadata only, one candidate per scene.
    candidates = scenes[::4] + [x for i, x in enumerate(scenes) if i % 4]
    client, storage = client_for_reference()
    prefix = storage['path'].rstrip('/') + '/'
    cache = OUT / 'vision-labels'
    cache.mkdir(parents=True, exist_ok=True)

    def inspect(pair):
        scene, info = pair
        fr = info['frames'][0]
        index = fr['frame_index']
        root = 'DataReorganized-VisionHead' if scene.startswith('dataset-') else 'DataReorganized-VisionHead-standard'
        folder = root + '/' + scene + ('/labeled/' if scene.startswith('dataset-') else '/dataset/') + str(index) + '/'
        key = hashlib.sha256(folder.encode()).hexdigest()[:20]
        dest = cache / (key + '.json')
        if dest.exists():
            return load_json(dest)
        listing = client.list_objects_v2(Bucket=storage['bucket'], Prefix=prefix + folder, MaxKeys=100)
        files = [x['Key'][len(prefix):] for x in listing.get('Contents', [])]
        labels = [x for x in files if Path(x).name.startswith('img_label') and x.endswith('.json')]
        if len(labels) != 1:
            return {'scene': scene, 'frame': fr, 'state': 'missing_label', 'folder': folder}
        raw, meta = read_object(client, storage, labels[0], limit=32768)
        obj = json.loads(raw)
        if len(obj) != 1:
            raise ValueError('Ambiguous classification GT')
        image, payload = next(iter(obj.items()))
        row = {'scene': scene, 'frame': fr, 'root': root, 'label_path': labels[0],
               'image_path': root + '/' + scene + ('/original/' if scene.startswith('dataset-') else '/dataset/') + str(index) + '/' + image,
               'labels': payload['label'], 'label_sha256': hashlib.sha256(raw).hexdigest(),
               'label_bytes': len(raw), 'state': 'available'}
        save_json(dest, row)
        (cache / (key + '.raw.json')).write_bytes(raw)
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(candidates), 80):
            rows.extend(pool.map(inspect, candidates[start:start + 80]))
            covered = {(k, v) for r in rows if r['state'] == 'available' for k, v in r['labels'].items()} & desired
            print(json.dumps({'metadata_scenes_checked': len(rows), 'reference_classes_found': len(covered), 'desired': len(desired)}), flush=True)
            if covered == desired:
                break
    remaining = set(desired)
    selected = []
    available = [r for r in rows if r['state'] == 'available']
    while remaining and available and len(selected) < 16:
        best = max(available, key=lambda r: len(set(r['labels'].items()) & remaining))
        gain = set(best['labels'].items()) & remaining
        if not gain:
            break
        selected.append(best)
        available.remove(best)
        remaining -= gain
    subset = {k: {} for k in data}
    for row in selected:
        scene = row['scene']
        subset['vision_eval'][scene] = dict(data['vision_eval'][scene], frames=[row['frame']])
    assert 1 <= len(selected) <= 16
    save_json(OUT / 'vision-subset.json', subset)
    save_json(OUT / 'vision-selection.json', {'dataset_version': '26-08-14-0', 'selection': 'class coverage controls, not population estimates',
               'source_path': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'metadata_scenes_checked': len(rows), 'selected': selected, 'uncovered': sorted(remaining),
               'max_frames': 16, 'new_image_bytes': 0, 'full_evaluation': False})
    print(json.dumps({'selected_frames': len(selected), 'covered_reference_classes': len(desired) - len(remaining),
                      'uncovered': sorted(remaining), 'selection': [[r['scene'], r['frame']['frame_index']] for r in selected]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
