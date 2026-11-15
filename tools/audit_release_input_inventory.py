"""Read-only inventory of the supplied release record and known dataset cache.

No model binaries or dataset objects are downloaded. Credentials stay in their
existing configuration files and are never included in the saved inventory.
"""
import contextlib
import io
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'evidence/input-contract-review-20261027'
URL = 'https://li.feishu.cn/record/PUBREF000000000000000000001'


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    checked_at = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run([shutil.which('lark-cli') or 'lark-cli', 'base', '+url-resolve',
                           '--url', URL, '--as', 'user'], capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError('Release record read failed')
    value = json.loads(proc.stdout)
    if not value.get('ok'):
        raise RuntimeError('Release record response is not successful')
    hint = value['data']['hint']
    fields = {row['id']: row['name'] for row in hint['fields']['fields']}
    record = {fields.get(key, key): val for key, val in hint['record_data'].items()}
    # Attachment file tokens are not needed for this inventory.
    for key, val in list(record.items()):
        if isinstance(val, list):
            record[key] = [{k: item[k] for k in ('name', 'size') if k in item}
                           if isinstance(item, dict) else item for item in val]
    save('release-record.json', {'checked_at': checked_at, 'url': URL, 'fields': record})
    print(json.dumps({'stage': 'record', 'version': record.get('版本号'),
                      'onnx_count': len(record.get('onnx') or []),
                      'compiled_archive_count': len(record.get('编译产物') or [])}), flush=True)

    integration = Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
    connection = next(c for c in json.loads(integration.read_text(encoding='utf-8-sig'))['connections']
                      if c['id'] == 'lpai-sdk-closedloop-20261022')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import asset
        cfg = asset.config(env='prod', bearer_file=connection['settings']['asset_bearer_file'])
    versions = []
    for version in ('26-09-08-2', '26-09-08-1'):
        row = {'version': version, 'files': []}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                resource = asset.resource('models/roadmodel-version-release/versions/' + version, config=cfg)
                storage = resource._get_s3_path(use='download')
                prefix = storage['path'].rstrip('/') + '/'
                client = resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
                for page in client.get_paginator('list_objects_v2').paginate(Bucket=storage['bucket'], Prefix=prefix):
                    for obj in page.get('Contents', []):
                        relative = obj['Key'][len(prefix):]
                        if relative and not relative.endswith('/'):
                            row['files'].append({'path': relative, 'bytes': obj['Size']})
                        if len(row['files']) > 1000:
                            raise ValueError('Referenced model inventory exceeds scope')
            row['state'] = 'listed'
        except Exception as error:
            row.update(state='unavailable', error_type=type(error).__name__)
        versions.append(row)
    save('model-files.json', {'checked_at': checked_at, 'versions': versions, 'binary_downloads': 0})
    print(json.dumps({'stage': 'models', 'versions': versions}), flush=True)

    manifest = json.loads((BASE / 'evidence/requested-dataset/compile-tensor-required-files.json').read_text())
    rows = manifest['files']
    samples = [rows[i] for i in sorted({0, 1, 2, 3, len(rows)//2, len(rows)-3, len(rows)-2, len(rows)-1})]
    root = '/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1'
    remote = '''import json, hashlib
from pathlib import Path
root=Path(ROOT_VALUE)
out={'root':str(root),'root_exists':root.is_dir(),'receipt_files':{},'sample_files':[]}
if root.is_dir():
 out['top_level_preview']=sorted(p.name for p in root.iterdir())[:12]
 for name in ['compile-tensor-sync-receipt.json','compile-with-occ-selected-85337a13554c-sync-receipt.json']:
  p=root/name
  if p.is_file(): out['receipt_files'][name]=json.loads(p.read_text())
 p=root/'data_link_26-10-20-1_train859_eval135_v7.json'
 if p.is_file():
  raw=p.read_bytes(); data=json.loads(raw)
  out['data_link']={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'eval_scenes':len(data.get('eval',{})),'eval_frames':sum(len(s.get('frames',[])) for s in data.get('eval',{}).values())}
 for row in SAMPLE_VALUE:
  p=root/row['path']; exists=p.is_file()
  actual=p.stat().st_size if exists else None
  out['sample_files'].append({'path':row['path'],'exists':exists,'bytes':actual,'expected_bytes':row.get('size'),'size_matches':actual==row.get('size')})
print(json.dumps(out))
'''.replace('ROOT_VALUE', repr(root)).replace('SAMPLE_VALUE', repr(samples))
    proc = subprocess.run([shutil.which('ssh.exe') or 'ssh', '-o', 'BatchMode=yes', '-o',
                           'ConnectTimeout=15', 'ubuntu-workstation', shlex.join(['python3', '-'])],
                          input=remote.encode(), capture_output=True, timeout=80)
    if proc.returncode:
        status = {'checked_at': checked_at, 'state': 'unavailable', 'exit_code': proc.returncode}
    else:
        status = {'checked_at': checked_at, 'state': 'metadata_and_sample_stat_checked', **json.loads(proc.stdout)}
    save('workstation-dataset.json', status)
    print(json.dumps({'stage': 'workstation', **status}), flush=True)


if __name__ == '__main__':
    main()
