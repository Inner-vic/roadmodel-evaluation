"""Workstation-only validation of an explicit dataset file allowlist."""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def main():
    request = json.load(sys.stdin)
    root = Path(request['root']).resolve()
    allowed = Path('/mnt/data/developer/pcm-roadmodel-eval').resolve()
    if allowed not in root.parents:
        raise ValueError('Unexpected dataset root')
    receipt_file = root / '.transfer-receipts.jsonl'
    receipts = {}
    if receipt_file.exists():
        for line in receipt_file.open():
            try:
                row = json.loads(line)
                if row.get('state') == 'complete':
                    receipts[row['path']] = row
            except (ValueError, KeyError):
                pass

    def check(item):
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['path']:
            raise ValueError('Unsafe dataset path')
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError('Dataset path escaped the expected root')
        if not path.exists():
            return {'path': item['path'], 'state': 'missing'}
        if not path.is_file() or path.stat().st_size != item['size']:
            return {'path': item['path'], 'state': 'conflict', 'reason': 'type_or_size_mismatch'}
        before = path.stat()
        old = receipts.get(item['path'], {})
        unchanged = (old.get('etag') == item['etag'] and old.get('size') == item['size']
                     and old.get('mtime_ns') == before.st_mtime_ns
                     and re.fullmatch('[a-f0-9]{64}', old.get('sha256', '')))
        if unchanged and not request.get('rehash', False):
            return dict(old, state='complete', verification='unchanged_receipt', refreshed=False)
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
                md5.update(block)
        after = path.stat()
        valid = (md5.hexdigest() == item['etag'].lower() if re.fullmatch('[a-fA-F0-9]{32}', item['etag'])
                 else sha.hexdigest() == old.get('sha256'))
        if not valid or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return {'path': item['path'], 'state': 'conflict', 'reason': 'content_mismatch_or_changed'}
        return {'path': item['path'], 'size': item['size'], 'etag': item['etag'],
                'sha256': sha.hexdigest(), 'mtime_ns': after.st_mtime_ns, 'state': 'complete',
                'verification': 'fresh_content_hash', 'refreshed': not unchanged, 'reused': True}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, request['files']))
    complete = [r for r in rows if r['state'] == 'complete']
    refresh = [r for r in complete if r.get('refreshed')]
    if refresh:
        with receipt_file.open('a') as stream:
            for row in refresh:
                stream.write(json.dumps(row) + '\n')
    result = {'schema': 'pcm.required-dataset-validation/v1',
              'state': 'complete' if len(complete) == len(rows) else 'incomplete',
              'rehash': bool(request.get('rehash')), 'expected_files': len(rows),
              'verified_files': len(complete), 'verified_bytes': sum(r['size'] for r in complete),
              'verified_paths': [r['path'] for r in complete],
              'missing': [r['path'] for r in rows if r['state'] == 'missing'],
              'conflicts': [r for r in rows if r['state'] == 'conflict']}
    print(json.dumps(result), flush=True)
    return 0 if not result['conflicts'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
