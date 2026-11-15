"""Read-only discovery of existing dataset scenes on mounted workstation disks.

Only audit reports are written, under this task's own directory. Source datasets
are never modified. Matching scene names are candidates, not verified reuse.
"""
import collections
import json
import os
import re
import time
from pathlib import Path

ROOT = Path('/mnt/data/developer/pcm-roadmodel-eval')
OUT = ROOT / 'dataset-audit'
SEARCH = ['/mnt/data', '/mnt/T7', '/media/lixiang', '/home/lixiang']
SKIP = {'.git', 'node_modules', '__pycache__', 'miniconda3', 'anaconda3',
        'miniforge3', 'venv', '.venv', 'build', 'install', 'lost+found',
        'System Volume Information', 'thinclient_drives'}


def main():
    OUT.mkdir(exist_ok=True)
    inventory = json.loads((OUT / 'expected-inventory.json').read_text())
    scenes = {row['path'].split('/')[0] for row in inventory['files']
              if '/dataset/' in row['path']}
    seen, candidates, denied, links, indices, interesting, limited = set(), [], [], [], [], [], []
    archive_candidates, loose_payload_candidates = [], []
    expected_name_size = {(Path(r['path']).name, r['size']) for r in inventory['files']}
    count = 0
    started = time.time()
    stack = [(Path(s), 0) for s in SEARCH]
    while stack:
        path, depth = stack.pop()
        try:
            st = path.stat()
            identity = (st.st_dev, st.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            count += 1
            if path.name in scenes:
                candidates.append({'scene': path.name, 'path': str(path),
                                   'resolved': str(path.resolve())})
                continue
            # Avoid descending into every frame of unrelated data collections.
            if re.match(r'^20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_', path.name):
                continue
            if re.search(r'v161|evaldataset|smartchassis|datareorganized', path.name, re.I):
                interesting.append(str(path))
            if depth >= 10:
                limited.append(str(path))
                continue
            with os.scandir(str(path)) as entries:
                for entry in entries:
                    if entry.name.startswith('.') or entry.name in SKIP:
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((Path(entry.path), depth + 1))
                        elif entry.is_symlink():
                            target = Path(entry.path).resolve()
                            links.append({'path': entry.path, 'target': str(target)})
                            if entry.is_dir() and any(str(target).startswith(s + '/') for s in SEARCH):
                                stack.append((Path(entry.path), depth + 1))
                        elif entry.name.startswith('data_link') and entry.name.endswith('.json'):
                            indices.append({'path': entry.path, 'size': entry.stat().st_size})
                        elif entry.is_file():
                            n = entry.stat().st_size
                            if (entry.name, n) in expected_name_size:
                                loose_payload_candidates.append({'path': entry.path, 'size': n})
                            if n >= 100 * 1024 * 1024 and entry.name.lower().endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.7z', '.tar.zst')):
                                archive_candidates.append({'path': entry.path, 'size': n})
                    except (PermissionError, OSError):
                        denied.append(entry.path)
        except (PermissionError, OSError):
            denied.append(str(path))
        if count % 5000 == 0:
            print(json.dumps({'stage': 'discovery', 'directories': count,
                              'candidate_scenes': len(candidates)}), flush=True)
    grouped = collections.Counter(str(Path(c['path']).parent) for c in candidates)
    record = {'schema': 'pcm.workstation-dataset-discovery/v1', 'searched_roots': SEARCH,
              'directories_visited': count, 'elapsed_seconds': round(time.time() - started, 2),
              'candidate_roots': dict(grouped), 'candidate_scene_paths': candidates,
              'data_link_files': indices, 'dataset_named_directories': interesting,
              'symlinks': links, 'permission_or_io_errors': denied,
              'archive_candidates': archive_candidates, 'loose_payload_candidates': loose_payload_candidates,
              'max_depth_reached': limited, 'pruned_directory_names': sorted(SKIP),
              'note': 'Scene-name matches only. File-level size and content verification is required before reuse.'}
    (OUT / 'discovery.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ('directories_visited', 'elapsed_seconds',
                                             'candidate_roots', 'data_link_files',
                                             'dataset_named_directories', 'permission_or_io_errors')},
                     ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
