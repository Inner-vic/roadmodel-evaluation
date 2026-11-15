"""Measure one missing allowlisted object via PC RAM -> workstation, no PC file copy."""
import contextlib
import hashlib
import json
from pathlib import Path
import time

from sync_requested_eval import BASE, LOCAL, SOURCE, REMOTE_ROOT, Quiet, ssh


def main():
    manifest = json.loads((LOCAL / 'compile-tensor-required-files.json').read_text())
    candidates = [r for r in reversed(manifest['files']) if r.get('role') == 'raw_lidar' and r['size'] > 4000000]
    check = 'import pathlib,sys; print(int((pathlib.Path(sys.argv[1])/sys.argv[2]).exists()))'
    item = next(r for r in candidates if ssh(['python3', '-c', check, REMOTE_ROOT, r['path']]).strip() == b'0')
    config_root = Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data')
    connection = next(c for c in json.loads((config_root / 'integrations.local.json').read_text(encoding='utf-8-sig'))['connections'] if c['id'] == 'lpai-sdk-closedloop-20261022')
    with contextlib.redirect_stdout(Quiet()), contextlib.redirect_stderr(Quiet()):
        import asset
        resource = asset.resource('datasets/smartchassisdata-v3/versions/26-10-20-1', config=asset.config(env='prod', bearer_file=connection['settings']['asset_bearer_file']))
        storage = resource._get_s3_path()
        client = resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
        url = client.generate_presigned_url('get_object', Params={'Bucket':storage['bucket'], 'Key':SOURCE + item['path']}, ExpiresIn=900)
    url = 'https://' + url.split('://', 1)[1]
    import requests
    start = time.monotonic()
    response = requests.get(url, timeout=(15, 90))
    response.raise_for_status()
    data = response.content
    fetch_seconds = time.monotonic() - start
    if len(data) != item['size'] or hashlib.md5(data).hexdigest() != item['etag'].lower():
        raise ValueError('Source size/MD5 mismatch')
    writer = '''import pathlib,sys,json,hashlib,os
root=pathlib.Path(sys.argv[1]).resolve();p=(root/sys.argv[2]).resolve();p.relative_to(root)
raw=sys.stdin.buffer.read()
if len(raw)!=int(sys.argv[3]) or hashlib.md5(raw).hexdigest()!=sys.argv[4]:raise ValueError('Content mismatch')
if p.exists():raise ValueError('Target already exists; preserve it')
p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_name(p.name+'.pc-transfer');temporary.write_bytes(raw);os.replace(str(temporary),str(p))
receipt={'path':sys.argv[2],'size':len(raw),'etag':sys.argv[4],'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':p.stat().st_mtime_ns,'state':'complete','reused':False,'route':'pc_memory_stream'}
with (root/'.transfer-receipts.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\\n')
print(json.dumps({'bytes':len(raw),'sha256':receipt['sha256']}))'''
    upload_start = time.monotonic()
    result = json.loads(ssh(['python3', '-c', writer, REMOTE_ROOT, item['path'], str(item['size']), item['etag']], data))
    result.update(path=item['path'], fetch_seconds=fetch_seconds, upload_seconds=time.monotonic()-upload_start,
                  total_seconds=time.monotonic()-start)
    (LOCAL / 'pc-stream-probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type':type(error).__name__}), flush=True)
        raise SystemExit(2)
