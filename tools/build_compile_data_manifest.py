"""Build a file allowlist for the supplied NumPy compilation-alignment pipeline.

This preserves all eval frames. It does not silently reduce scene coverage.
The profile is the supplied one-camera, cold-start reference path, not a claim
about future multi-camera/temporal models or the official Torch dataloader.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/requested-dataset'
LINK = EVIDENCE / 'data_link_26-10-20-1_train859_eval135_v7.json'
INVENTORY = EVIDENCE / 'eval135-file-inventory.json'


def build(profile):
    raw = INVENTORY.read_bytes()
    source = json.loads(raw)
    rows = {r['path']: r for r in source['files']}
    link = json.loads(LINK.read_text(encoding='utf-8'))
    if link.get('joint_eval') != link.get('eval'):
        raise ValueError('The runtime combines eval and joint_eval; review their different scopes first')
    by_parent = collections.defaultdict(list)
    for name in rows:
        by_parent[str(PurePosixPath(name).parent)].append(name)
    required, frames = {}, set()

    def include(name, role):
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or name not in rows:
            raise ValueError('Missing or unsafe required input: ' + name)
        required[name] = dict(rows[name], role=role)

    include(LINK.name, 'data_link')
    for scene, info in link['eval'].items():
        if not info.get('sensor_param'):
            include(scene + '/dataset/sensor_param.json', 'sensor_param')
        for frame in info['frames']:
            frame_id = str(frame['frame_index'])
            key = (scene, frame_id)
            if key in frames:
                raise ValueError('Duplicate scene/frame requires explicit runtime deduplication review')
            frames.add(key)
            parent = scene + '/dataset/' + frame_id
            candidates = by_parent[parent]
            for field, role, predicate in (
                ('img_file', 'camera_120', lambda n: n.startswith(frame_id + '_f_ca_120') and n.endswith('.jpg')),
                ('pcd_file', 'raw_lidar', lambda n: n.startswith('lidar_original') and n.endswith('.npy')),
            ):
                if frame.get(field):
                    name = parent + '/' + PurePosixPath(frame[field]).name
                else:
                    matches = [n for n in candidates if predicate(PurePosixPath(n).name)]
                    if len(matches) != 1:
                        raise ValueError('Input pairing is ambiguous or missing: ' + parent + '/' + role)
                    name = matches[0]
                include(name, role)
            if profile == 'with_occ':
                include(parent + '/' + PurePosixPath(frame['occ_file']).name, 'occ_gt')
    ordered = [required[r['path']] for r in source['files'] if r['path'] in required]
    roles = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
    for row in ordered:
        roles[row['role']]['files'] += 1
        roles[row['role']]['bytes'] += row['size']
    available_path = EVIDENCE / 'workstation-reuse-plan.json'
    available = json.loads(available_path.read_text())['available'] if available_path.exists() else {}
    have = [r for r in ordered if r['path'] in available and available[r['path']]['etag'] == r['etag']]
    total = sum(r['size'] for r in ordered)
    return {'schema': 'pcm.compile-data-manifest/v1', 'profile': profile,
            'source_inventory_sha256': hashlib.sha256(raw).hexdigest(),
            'data_link_sha256': hashlib.sha256(LINK.read_bytes()).hexdigest(),
            'source_uri': source['source_uri'], 'scenes': len(link['eval']), 'frames': len(frames),
            'contract': {'pipeline': 'supplied NumPy reference', 'camera': 'single f_ca_120',
                         'temporal': False, 'trajectory': 'generated fixed lines',
                         'gt_based_scene_sampling': False, 'frame_fraction': 1.0,
                         'ground_truth_metrics': profile == 'with_occ'},
            'file_count': len(ordered), 'total_bytes': total, 'by_role': dict(roles),
            'excluded_files': len(rows) - len(ordered),
            'excluded_bytes': sum(r['size'] for r in source['files']) - total,
            'local_reuse_snapshot': {'verified_or_materialized_files': len(have),
                                     'verified_or_materialized_bytes': sum(r['size'] for r in have),
                                     'remaining_files': len(ordered) - len(have),
                                     'remaining_bytes': total - sum(r['size'] for r in have),
                                     'note': 'Reuse audit snapshot; stat/receipt preflight required before transfer'},
            'files': ordered}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['tensor_only', 'with_occ'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
