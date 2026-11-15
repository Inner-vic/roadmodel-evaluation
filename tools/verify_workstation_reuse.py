"""Verify discovered candidates against the frozen cloud object inventory.

Existing task transfers use their MD5/SHA256 receipts plus unchanged size/mtime.
Other files are hashed and compared with the inventory's single-part MD5 ETag.
This command never changes source datasets and never downloads objects.
"""
import collections
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/data/developer/pcm-roadmodel-eval')
OUT = ROOT / 'dataset-audit'
DATA = ROOT / 'datasets/smartchassisdata-v3/26-10-20-1'


def hash_file(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def main():
    raw = (OUT / 'expected-inventory.json').read_bytes()
    inventory = json.loads(raw)
    discovery = json.loads((OUT / 'discovery.json').read_text())
    receipts = {}
    with (DATA / '.transfer-receipts.jsonl').open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if record.get('state') == 'complete':
                    receipts[record['path']] = record
            except (ValueError, KeyError):
                pass
    expected = inventory['files']
    by_name_size = collections.defaultdict(list)
    for row in expected:
        by_name_size[(Path(row['path']).name, row['size'])].append(row)
    available, content_sources, conflicts = {}, {}, []
    own_files = 0
    for row in expected:
        path = DATA / row['path']
        if not path.is_file():
            continue
        st = path.stat()
        previous = receipts.get(row['path'], {})
        unchanged = (previous.get('etag') == row['etag'] and st.st_size == row['size']
                     and previous.get('mtime_ns') == st.st_mtime_ns
                     and re.fullmatch('[a-f0-9]{64}', previous.get('sha256', '')))
        sha = previous.get('sha256') if unchanged else None
        method = 'unchanged_verified_transfer_receipt'
        if not unchanged:
            if st.st_size == row['size'] and re.fullmatch('[a-fA-F0-9]{32}', row['etag']):
                sha, md5 = hash_file(path)
                unchanged = md5 == row['etag'].lower()
                method = 'fresh_md5_matches_cloud_etag'
        if not unchanged:
            conflicts.append({'path': row['path'], 'reason': 'Existing target does not match its verified source'})
            continue
        source = {'source': str(path.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                  'device': st.st_dev, 'inode': st.st_ino, 'sha256': sha,
                  'etag': row['etag'], 'verification': method}
        available[row['path']] = dict(source, mode='existing_target')
        content_sources.setdefault((row['etag'].lower(), row['size']), source)
        own_files += 1
    external_reports, hashed = [], {}
    # Scene discovery also finds prediction products and raw replay recordings.
    # Only exact size + content matches may become reuse sources.
    external_scenes = [c for c in discovery['candidate_scene_paths']
                       if DATA.resolve() not in Path(c['resolved']).parents]
    loose = discovery.get('loose_payload_candidates', [])
    for candidate in external_scenes:
        base = Path(candidate['path'])
        files = [Path(parent) / name for parent, _, names in os.walk(str(base)) for name in names]
        report = {'path': str(base), 'scene': candidate['scene'], 'files_examined': len(files),
                  'bytes': 0, 'name_size_candidates': 0, 'content_matches': 0, 'mismatches': 0,
                  'extensions': dict(collections.Counter(p.suffix for p in files))}
        for path in files:
            try:
                st = path.stat()
                report['bytes'] += st.st_size
                rows = by_name_size.get((path.name, st.st_size), [])
                if not rows:
                    continue
                report['name_size_candidates'] += 1
                identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                if identity not in hashed:
                    hashed[identity] = hash_file(path)
                sha, md5 = hashed[identity]
                matching = [row for row in rows if row['etag'].lower() == md5]
                if not matching:
                    report['mismatches'] += 1
                    continue
                report['content_matches'] += 1
                for row in matching:
                    content_sources.setdefault((row['etag'].lower(), row['size']), {
                        'source': str(path.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                        'device': st.st_dev, 'inode': st.st_ino, 'sha256': sha,
                        'etag': row['etag'], 'verification': 'fresh_md5_matches_cloud_etag'})
            except (PermissionError, OSError):
                report.setdefault('unreadable', []).append(str(path))
        external_reports.append(report)
    for item in loose:
        path = Path(item['path'])
        st = path.stat()
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if identity not in hashed:
            hashed[identity] = hash_file(path)
        sha, md5 = hashed[identity]
        for row in by_name_size.get((path.name, st.st_size), []):
            if row['etag'].lower() == md5:
                content_sources.setdefault((row['etag'].lower(), row['size']), {
                    'source': str(path.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                    'device': st.st_dev, 'inode': st.st_ino, 'sha256': sha,
                    'etag': row['etag'], 'verification': 'fresh_md5_matches_cloud_etag'})
    missing = []
    scenes = collections.defaultdict(lambda: {'expected_files': 0, 'existing_files': 0,
                                              'locally_reusable_files': 0, 'missing_bytes': 0})
    for row in expected:
        scene = row['path'].split('/')[0]
        counts = scenes[scene]
        counts['expected_files'] += 1
        if row['path'] not in available:
            source = content_sources.get((row['etag'].lower(), row['size']))
            if source:
                available[row['path']] = dict(source, mode='reuse_local_content')
        source = available.get(row['path'])
        if source:
            counts['existing_files' if source['mode'] == 'existing_target' else 'locally_reusable_files'] += 1
        else:
            missing.append(row)
            counts['missing_bytes'] += row['size']
    reused = {k: v for k, v in available.items() if v['mode'] == 'reuse_local_content'}
    summary = {'schema': 'pcm.workstation-reuse-audit/v1',
               'audited_at': datetime.now(timezone.utc).isoformat(),
               'inventory_sha256': hashlib.sha256(raw).hexdigest(),
               'expected_files': len(expected), 'expected_bytes': sum(r['size'] for r in expected),
               'existing_verified_files': own_files,
               'existing_verified_bytes': sum(v['size'] for v in available.values() if v['mode'] == 'existing_target'),
               'additional_locally_reusable_files': len(reused),
               'additional_locally_reusable_bytes': sum(v['size'] for v in reused.values()),
               'missing_files': len(missing), 'missing_bytes': sum(r['size'] for r in missing),
               'external_candidates': external_reports, 'target_conflicts': conflicts,
               'per_scene': dict(scenes), 'search_roots': discovery['searched_roots'],
               'scope_limits': {'permission_or_io_errors': discovery['permission_or_io_errors'],
                                'max_depth': 10, 'pruned_directory_names': discovery['pruned_directory_names']},
               'archive_candidates': discovery.get('archive_candidates', []),
               'download_state': 'paused_for_reuse_audit'}
    (OUT / 'reuse-plan.json').write_text(json.dumps({'schema': 'pcm.local-reuse-plan/v1',
        'inventory_sha256': summary['inventory_sha256'], 'available': available,
        'missing': [r['path'] for r in missing], 'conflicts': conflicts}, indent=2))
    (OUT / 'reuse-audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_scene'}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
