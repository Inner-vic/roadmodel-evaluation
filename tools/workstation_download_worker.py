"""Workstation-only byte transfer. Signed URLs are received on stdin, never saved."""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import sys
import threading
import time
import requests
import fcntl


def main():
    request = json.load(sys.stdin)
    root = pathlib.Path(request["root"]).resolve()
    allowed = pathlib.Path('/home/lixiang/developer/pcm-roadmodel-eval').resolve()
    if allowed not in root.parents:
        raise ValueError('Unexpected download root')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.download-worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 73
    if (root / '.stop-after-batch').exists():
        print(json.dumps({'state':'stopped_for_worker_update'}),flush=True)
        return 75
    receipts = root / '.transfer-receipts.jsonl'
    completed = {}
    sessions = threading.local()
    if receipts.exists():
        for line in receipts.read_text().splitlines():
            try:
                entry = json.loads(line)
                if entry.get('state') == 'complete':
                    completed[entry['path']] = entry
            except ValueError:
                pass

    def download(item):
        relative = pathlib.PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
            raise ValueError('Unsafe relative download path')
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError('Download path escapes root')
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = completed.get(item['path'], {})
        if target.is_file() and previous.get('etag') == item['etag'] and target.stat().st_size == item['size'] and previous.get('mtime_ns') == target.stat().st_mtime_ns:
            return dict(previous, reused=True)
        if target.exists():
            raise ValueError('An unverified target file already exists')
        partial = target.with_name(target.name + '.pcm-partial')
        sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
        try:
            if not hasattr(sessions, 'value'):
                sessions.value = requests.Session()
            with sessions.value.get(item['url'], stream=True, timeout=(15, 60)) as response, partial.open('wb') as stream:
                response.raise_for_status()
                etag = response.headers.get('ETag', '').strip('"').lower()
                if etag and etag != item['etag'].lower():
                    raise ValueError('Source ETag changed')
                for block in response.iter_content(1024 * 1024):
                    stream.write(block); sha.update(block); md5.update(block); size += len(block)
            if size != item['size']:
                raise ValueError('Downloaded length differs from object inventory')
            if re.fullmatch('[a-fA-F0-9]{32}', item['etag']) and md5.hexdigest() != item['etag'].lower():
                raise ValueError('Downloaded MD5 differs from object ETag')
            os.replace(str(partial), str(target))
            return {'path': item['path'], 'size': size, 'etag': item['etag'], 'sha256': sha.hexdigest(),
                    'mtime_ns': target.stat().st_mtime_ns, 'state': 'complete', 'reused': False}
        except Exception as error:
            # Error strings can include signed URLs. Persist only the type/status.
            return {'path': item['path'], 'state': 'failed', 'error_type': type(error).__name__,
                    'http_status': getattr(error, 'code', None)}

    def download_with_retry(item):
        result = None
        for attempt in range(3):
            result = download(item)
            if result.get('state') == 'complete':
                return result
            time.sleep(2 ** attempt)
        return result

    results = []
    workers = int(request.get('workers', 32))
    if not 1 <= workers <= 64:
        raise ValueError('Unexpected worker count')
    with receipts.open('a', encoding='utf-8') as log, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_with_retry, item) for item in request['files']]
        for future in concurrent.futures.as_completed(futures):
            item = future.result(); results.append(item)
            log.write(json.dumps(item, ensure_ascii=False) + '\n'); log.flush()
    failed = [x for x in results if x['state'] != 'complete']
    print(json.dumps({'state': 'failed' if failed else 'complete', 'root': str(root),
                      'files': len(results), 'bytes': sum(x.get('size', 0) for x in results),
                      'downloaded_files': sum(x.get('state')=='complete' and not x.get('reused',False) for x in results),
                      'downloaded_bytes': sum(x.get('size',0) for x in results if x.get('state')=='complete' and not x.get('reused',False)),
                      'reused_files': sum(x.get('reused', False) for x in results), 'failures': failed}, ensure_ascii=True), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
