"""Prepare only missing OCC GT for the authorized full validation set."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import load_json, save_json
from pcm_eval.planning import ssh_command

TASK = ROOT / 'evidence/formal-v322-20261029-01'
REMOTE = '/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'
DATASET = '/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1'


def ssh(argv, payload=None, timeout=120):
    p = subprocess.run(ssh_command('ubuntu-workstation', argv), input=payload, capture_output=True, timeout=timeout)
    if p.returncode: raise RuntimeError('Task SSH failed; exit=' + str(p.returncode))
    return p.stdout


def state(**values):
    d = {'updated_at': datetime.now(timezone.utc).isoformat(), **values}
    save_json(TASK / 'data-state.json', d, overwrite=True)
    writer = 'import pathlib,sys,os;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(".tmp");q.write_bytes(sys.stdin.buffer.read());os.replace(q,p)'
    ssh(['python3', '-c', writer, REMOTE + '/data-state.json'], json.dumps(d).encode())


def main():
    rows = load_json(TASK / 'occ-required.json')['files']
    writer = 'import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b) if not p.exists() else None'
    for local, name in [('tools/check_required_dataset.py', 'check_required_dataset.py')]:
        ssh(['python3', '-c', writer, REMOTE + '/code/' + name], (ROOT / local).read_bytes())
    worker = (ROOT / 'tools/workstation_download_worker.py').read_text()
    worker = worker.replace("'/home/lixiang/developer/pcm-roadmodel-eval'", "'/mnt/data/developer/pcm-roadmodel-eval'")
    worker = worker.replace("(root / '.stop-after-batch').exists()", "pathlib.Path(" + repr(REMOTE + '/STOP_AFTER_FRAME') + ").exists()")
    (TASK / 'download-worker.py').write_text(worker, encoding='utf-8')
    ssh(['python3', '-c', writer, REMOTE + '/code/download-worker.py'], worker.encode())
    pre = json.loads(ssh(['python3', REMOTE + '/code/check_required_dataset.py'], json.dumps({'root': DATASET, 'files': rows}).encode()))
    assert not pre['conflicts'], 'Existing differing data must be preserved'
    verified = set(pre['verified_paths'])
    pending = [r for r in rows if r['path'] not in verified]
    done = len(verified)
    state(state='DOWNLOADING_GT', completed_files=done, total_files=len(rows), reused_files=done, new_bytes=0)
    cfg = Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
    connection = next(c for c in load_json(cfg)['connections'] if c['id'] == 'lpai-sdk-closedloop-20261022')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import asset
        resource = asset.resource('datasets/smartchassisdata-v3/versions/26-10-20-1', config=asset.config(env='prod', bearer_file=connection['settings']['asset_bearer_file']))
        storage = resource._get_s3_path()
        client = resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
    assert storage['path'].rstrip('/') == 'datasets/DataReorganized-V161-minidataset-test'
    new_bytes = 0
    for start in range(0, len(pending), 120):
        files = []
        for row in pending[start:start + 120]:
            url = client.generate_presigned_url('get_object', Params={'Bucket': storage['bucket'], 'Key': storage['path'].rstrip('/') + '/' + row['path']}, ExpiresIn=3600)
            files.append(dict(row, url='https://' + url[7:] if url.startswith('http://') else url))
        r = json.loads(ssh(['python3', REMOTE + '/code/download-worker.py'], json.dumps({'root': DATASET, 'files': files, 'workers': 16}).encode(), timeout=600))
        assert r['state'] == 'complete', 'GT transfer failed; signed URLs are never logged'
        done += len(files)
        new_bytes += r['downloaded_bytes']
        state(state='DOWNLOADING_GT', completed_files=done, total_files=len(rows), reused_files=len(verified), new_bytes=new_bytes)
        print(json.dumps({'completed_GT': done, 'total': len(rows), 'new_bytes': new_bytes}), flush=True)
    state(state='READY', completed_files=done, total_files=len(rows), reused_files=len(verified), new_bytes=new_bytes)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        state(state='NEEDS_REVIEW', error_type=type(e).__name__)
        raise
