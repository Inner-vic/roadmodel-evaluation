"""Materialize verified identical content using hard links in this task only."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/data/developer/pcm-roadmodel-eval').resolve()
OUT = ROOT / 'dataset-audit'
DATA = ROOT / 'datasets/smartchassisdata-v3/26-10-20-1'


def main():
    if not (DATA / '.stop-after-batch').exists():
        raise ValueError('Stop the downloader before preparing local reuse')
    raw = (OUT / 'expected-inventory.json').read_bytes()
    inventory = json.loads(raw)['files']
    expected = {r['path']: r for r in inventory}
    plan = json.loads((OUT / 'reuse-plan.json').read_text())
    if hashlib.sha256(raw).hexdigest() != plan['inventory_sha256'] or plan.get('held_archive_candidates'):
        raise ValueError('Inventory changed or content candidates remain unverified')
    linked = []
    with (DATA / '.transfer-receipts.jsonl').open('a') as receipts:
        for relative, info in plan['available'].items():
            if info['mode'] != 'reuse_local_content':
                continue
            source = Path(info['source']).resolve()
            target = DATA / relative
            # All currently verified reuse sources are already in this task.
            parent = target.parent.resolve()
            if ROOT not in source.parents or not (parent == DATA.resolve() or DATA.resolve() in parent.parents):
                raise ValueError('Source or destination escaped this task')
            before = source.stat()
            data = source.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            row = expected[relative]
            if (len(data) != row['size'] or sha != info['sha256']
                    or hashlib.md5(data).hexdigest() != row['etag'].lower()
                    or before.st_mtime_ns != source.stat().st_mtime_ns):
                raise ValueError('Local reuse source no longer matches the expected content')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not os.path.samefile(str(source), str(target)):
                    raise ValueError('An unexpected target exists; do not overwrite it')
            else:
                os.link(str(source), str(target))
            stat = target.stat()
            record = {'path': relative, 'size': stat.st_size, 'etag': row['etag'], 'sha256': sha,
                      'mtime_ns': stat.st_mtime_ns, 'state': 'complete', 'reused': True,
                      'reuse_method': 'hardlink', 'reuse_source': str(source)}
            receipts.write(json.dumps(record) + '\n')
            receipts.flush()
            linked.append(record)
    result = {'schema': 'pcm.local-reuse-receipt/v1', 'linked_files': len(linked),
              'logical_bytes': sum(r['size'] for r in linked), 'network_bytes': 0,
              'method': 'hardlink', 'source_files_modified': False, 'files': linked}
    (OUT / 'local-reuse-receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
