"""Read only candidate pose bytes from existing 7z archives, never entire datasets."""
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path('/mnt/data/developer/pcm-roadmodel-eval/dataset-audit')


def main():
    archives = json.loads((OUT / 'archive-audit.json').read_text())['archives']
    inventory = json.loads((OUT / 'expected-inventory.json').read_text())['files']
    plan = json.loads((OUT / 'reuse-plan.json').read_text())
    wanted = {r['etag'].lower(): r for r in inventory if r['path'].split('/')[-1].startswith('T_frame_')}
    cache = OUT / 'archive-pose-cache'
    cache.mkdir(exist_ok=True)
    results, verified_sources = [], {}
    for index, archive in enumerate(archives):
        candidates = [c for c in archive['name_size_candidates']
                      if Path(c['member']).name.startswith('T_frame_') and c['size'] == 256]
        if not candidates or not archive['path'].lower().endswith('.7z'):
            continue
        if any('\n' in c['member'] or '\r' in c['member'] for c in candidates):
            raise ValueError('Unsafe member name')
        names = OUT / ('pose-members-' + str(index) + '.txt')
        names.write_text('\n'.join(c['member'] for c in candidates) + '\n')
        # All selected NPY files are 256 bytes. Match by content digest, not
        # concatenation order or the old dataset's scene/frame naming.
        proc = subprocess.run(['7z', 'x', '-so', '-y', '-i@' + str(names), '--', archive['path']],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode or len(proc.stdout) != 256 * len(candidates):
            raise ValueError('Archive extraction count/length mismatch')
        matches = set()
        for offset in range(0, len(proc.stdout), 256):
            raw = proc.stdout[offset:offset + 256]
            md5 = hashlib.md5(raw).hexdigest()
            if md5 not in wanted:
                continue
            matches.add(md5)
            path = cache / (md5 + '.npy')
            if not path.exists():
                path.write_bytes(raw)
            st = path.stat()
            verified_sources[md5] = {'source': str(path), 'size': 256, 'mtime_ns': st.st_mtime_ns,
                'device': st.st_dev, 'inode': st.st_ino, 'sha256': hashlib.sha256(raw).hexdigest(),
                'etag': md5, 'verification': 'archive_bytes_md5_matches_cloud_etag',
                'source_archive': archive['path']}
        results.append({'path': archive['path'], 'candidate_members': len(candidates),
                        'bytes_checked': len(proc.stdout), 'matching_content_digests': len(matches)})
        print(json.dumps(results[-1]), flush=True)
    added = 0
    for row in inventory:
        if row['path'] not in plan['available'] and row['etag'].lower() in verified_sources:
            plan['available'][row['path']] = dict(verified_sources[row['etag'].lower()], mode='reuse_local_content')
            added += 1
    plan['missing'] = [r['path'] for r in inventory if r['path'] not in plan['available']]
    plan['held_archive_candidates'] = []
    plan['ready_for_download'] = False  # Local links and a download preflight are still required.
    plan['download_state'] = 'paused_after_local_inventory'
    (OUT / 'reuse-plan.json').write_text(json.dumps(plan, indent=2))
    result = {'archives': results, 'additional_reusable_targets': added,
              'all_name_size_pose_candidates_content_checked': True}
    (OUT / 'archive-pose-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'additional_reusable_targets': added}), flush=True)


if __name__ == '__main__':
    main()
