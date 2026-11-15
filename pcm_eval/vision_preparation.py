"""Prepare exact per-image GT from the submitted mount, without another dataset.

Only frozen evaluation frames are inspected. Scene tags, neighbouring frames and
historic caches are never searched or broadcast into per-image labels.
"""
from copy import deepcopy
from pathlib import Path

from .common import digest, json_digest, load_json, save_json
from .vision_gt_binding import BindingRejected, _inside


def _write(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise BindingRejected('PREPARATION_CHANGED', {'path': str(path)})
    else:
        save_json(path, value)


def _labels(path, image_key, roots):
    path = _inside(path, roots)
    if path.stat().st_size > 1024 * 1024:
        raise BindingRejected('LABEL_FILE_TOO_LARGE', {})
    obj = load_json(path)
    entry = obj.get(image_key) if isinstance(obj, dict) else None
    values = entry.get('label') if isinstance(entry, dict) else None
    if not isinstance(values, dict) or not values:
        raise BindingRejected('NO_EXACT_IMAGE_LABEL', {'image_key': image_key})
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise BindingRejected('INVALID_CLASSIFICATION_LABELS', {})
    return path, values


def prepare(dataset, selected, output):
    """All selected frames must have unambiguous GT; never shrink coverage.

    Explicit vision_eval declarations take precedence. If that split is empty,
    inspect only the frozen OCC evaluation frames for labels delivered beside
    their exact image. An optional historical binding must pass pixel identity.
    """
    root = Path(dataset['mount_receipt']['mount_path']).resolve()
    index = _inside(dataset['index_path'], [root])
    if digest(index) != dataset['index_sha256']:
        raise BindingRejected('INDEX_CHANGED', {})
    original = load_json(index)
    bound = dataset.get('vision_gt_binding')
    if bound:
        from .vision_gt_binding import verify
        data, _ = verify(dataset, bound)
    else:
        data = original
    has_vision = any(v.get('frames') for v in data.get('vision_eval', {}).values())
    split = 'vision_eval' if has_vision else 'eval'
    if not selected or len(selected) != len(set((s, str(f)) for s, f in selected)):
        raise BindingRejected('INVALID_VISION_SCOPE', {})
    out = Path(output).resolve()
    if out.is_relative_to(root):
        raise BindingRejected('INPUT_DATASET_MUST_REMAIN_UNCHANGED', {})
    label_roots = [root] + ([Path(bound['path']).parent] if bound else [])
    accepted, issues = [], []
    for scene, fid in selected:
        try:
            if Path(scene).is_absolute() or '..' in Path(scene).parts or str(fid) in ('', '.', '..') or Path(str(fid)).name != str(fid):
                raise BindingRejected('INVALID_FRAME_PATH', {})
            matches = [f for f in data.get(split, {}).get(scene, {}).get('frames', [])
                       if str(f['frame_index']) == str(fid)]
            if len(matches) != 1:
                raise BindingRejected('NOT_IN_SUBMITTED_VALIDATION', {})
            frame = deepcopy(matches[0])
            roots = [Path(p).resolve() for p in dataset.get('data_roots', [str(root)])]
            for data_root in roots:
                data_root.relative_to(root)
            folders = [r/scene/style/str(fid) for r in roots for style in ('dataset','original')]
            if frame.get('img_file'):
                if Path(frame['img_file']).name != frame['img_file']:
                    raise BindingRejected('SUBMITTED_IMAGE_PATH_MISMATCH', {})
                images = [p/frame['img_file'] for p in folders if (p/frame['img_file']).is_file()]
            else:
                images = [p for folder in folders for p in folder.glob(str(fid) + '_f_ca_120*.jpg')]
            images = list(dict.fromkeys(images))
            if len(images) != 1:
                raise BindingRejected('IMAGE_MISSING_OR_AMBIGUOUS', {})
            image = _inside(images[0], [root])
            folder = image.parent
            if image.parent != folder.resolve():
                raise BindingRejected('SUBMITTED_IMAGE_PATH_MISMATCH', {})
            directories = [folder, folder/'labels', folder/'label',
                           *[r/scene/'labeled'/str(fid) for r in roots]]
            explicit = frame.get('img_label_file')
            if explicit:
                label_path = Path(explicit)
                candidates = ([label_path] if label_path.is_absolute() else
                              [d/label_path for d in directories if (d/label_path).is_file()])
                if len(candidates) != 1:
                    raise BindingRejected('LABEL_MISSING_OR_AMBIGUOUS', {})
                label, values = _labels(candidates[0], image.name, label_roots)
            else:
                candidates = list(dict.fromkeys(p for d in directories if d.is_dir()
                                               for p in d.glob('img_label*.json')))
                if len(candidates) > 100:
                    raise BindingRejected('LABEL_CANDIDATES_AMBIGUOUS', {})
                matching = []
                for candidate in candidates:
                    try:
                        matching.append(_labels(candidate, image.name, [root]))
                    except BindingRejected as error:
                        if error.code != 'NO_EXACT_IMAGE_LABEL':
                            raise
                if not matching:
                    raise BindingRejected('PER_IMAGE_GT_NOT_FOUND', {'image_key': image.name})
                if len(matching) != 1:
                    raise BindingRejected('LABEL_MISSING_OR_AMBIGUOUS', {})
                label, values = matching[0]
            frame.update(img_file=image.name, img_label_file=str(label))
            accepted.append({'scene': scene, 'frame_index': fid, 'frame': frame,
                             'image': str(image), 'image_sha256': digest(image),
                             'label': str(label), 'label_sha256': digest(label),
                             'image_key': image.name, 'tasks': sorted(values)})
        except (BindingRejected, OSError, ValueError, TypeError, KeyError) as error:
            issues.append({'scene': scene, 'frame_index': fid,
                           'code': error.code if isinstance(error, BindingRejected) else 'INVALID_FRAME_GT',
                           'detail': error.detail if isinstance(error, BindingRejected) else {}})
    receipt = {'schema': 'roadmodel.vision-preparation/v1',
               'state': 'BLOCKED' if issues else 'READY',
               'dataset_index_sha256': dataset['index_sha256'], 'dataset_root': str(root),
               'selected_frames': selected, 'source_split': split, 'frames': accepted,
               'issues': issues, 'scene_tags_used': False, 'inference_frames': 0}
    if not issues:
        subset = {key: {} for key in original}
        subset.setdefault('vision_eval', {})
        for row in accepted:
            subset['vision_eval'].setdefault(row['scene'], {'scene_name': row['scene'], 'frames': []})['frames'].append(row['frame'])
        link = out/'data-link.json'
        _write(link, subset)
        receipt.update(index_path=str(link), index_sha256=digest(link))
    receipt['preparation_sha256'] = json_digest(receipt)
    _write(out/'preparation.json', receipt)
    return receipt


def verify(receipt, required_tasks=()):
    """Run immediately before native evaluation, using actual configured tasks."""
    if (receipt.get('state') != 'READY' or receipt.get('schema') != 'roadmodel.vision-preparation/v1'
            or receipt.get('scene_tags_used') is not False
            or receipt.get('preparation_sha256') != json_digest({k:v for k,v in receipt.items() if k != 'preparation_sha256'})):
        raise BindingRejected('INVALID_VISION_PREPARATION', {})
    if digest(Path(receipt['index_path'])) != receipt['index_sha256']:
        raise BindingRejected('DERIVED_INDEX_CHANGED', {})
    for row in receipt['frames']:
        if digest(Path(row['image'])) != row['image_sha256'] or digest(Path(row['label'])) != row['label_sha256']:
            raise BindingRejected('SOURCE_CHANGED', {'scene': row['scene'], 'frame_index': row['frame_index']})
        values = load_json(row['label'])[row['image_key']]['label']
        missing = sorted(set(required_tasks) - set(values))
        if missing:
            raise BindingRejected('REQUIRED_TASK_LABELS_MISSING', {'tasks': missing})
    return receipt
