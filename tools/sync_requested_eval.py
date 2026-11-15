"""Download only an explicit compilation-input allowlist to the workstation.

The personal bearer stays on this PC. The workstation receives short-lived,
object-scoped signed HTTPS URLs over SSH stdin; they are not written to logs.
"""
import argparse
import contextlib
import hashlib
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
LOCAL = BASE / 'evidence/requested-dataset'
SOURCE = 'datasets/DataReorganized-V161-minidataset-test/'
DATA_LINK = 'data_link_26-10-20-1_train859_eval135_v7.json'
REMOTE_BASE = '/mnt/data/developer/pcm-roadmodel-eval'
REMOTE_ROOT = REMOTE_BASE + '/datasets/smartchassisdata-v3/26-10-20-1'
PROGRESS_LABEL = None


class Quiet:
    def write(self, value): return len(value)
    def flush(self): pass


class BatchStopped(RuntimeError):
    pass


def ssh(args, payload=None, timeout=180):
    command = [shutil.which('ssh.exe') or 'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               'ubuntu-workstation', shlex.join(args)]
    proc = subprocess.run(command, input=payload, capture_output=True, timeout=timeout)
    if proc.returncode == 75:
        raise BatchStopped('Stopped between verified batches')
    if proc.returncode:
        # Do not print a failed command's stdin or potentially signed URL errors.
        raise RuntimeError('Workstation command failed with exit ' + str(proc.returncode))
    return proc.stdout


def main():
    global PROGRESS_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--selected-frames-receipt', type=Path,
                        help='Restrict verification/downloads to exact existing run scene/frame pairs')
    parser.add_argument('--reuse-inventory', action='store_true', help='Compatibility flag; the frozen inventory is always used')
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--required-manifest', type=Path, default=LOCAL/'compile-tensor-required-files.json')
    parser.add_argument('--plan-only', action='store_true', help='Show the selected file scope without SSH or cloud access')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 480:
        parser.error('--batch-size must be in 1..480')
    if not 1 <= args.workers <= 64:
        parser.error('--workers must be in 1..64')
    policy = json.loads(args.required_manifest.read_text(encoding='utf-8'))
    frozen = (LOCAL/'eval135-file-inventory.json').read_bytes()
    source_inventory = json.loads(frozen)
    if (policy.get('schema') != 'pcm.compile-data-manifest/v1'
            or policy.get('profile') not in ('tensor_only', 'with_occ')
            or policy.get('source_inventory_sha256') != hashlib.sha256(frozen).hexdigest()):
        raise ValueError('Required-input manifest does not match the frozen source inventory')
    if policy.get('data_link_sha256') != hashlib.sha256((LOCAL/DATA_LINK).read_bytes()).hexdigest():
        raise ValueError('Required-input manifest has a different data_link')
    rows = {r['path']: r for r in source_inventory['files']}
    if len({r['path'] for r in policy['files']}) != len(policy['files']):
        raise ValueError('Duplicate required-input paths')
    inventory = []
    for row in policy['files']:
        original = rows.get(row['path'])
        if original is None or any(original[k] != row.get(k) for k in ('size', 'etag')):
            raise ValueError('Required-input metadata differs from source inventory')
        inventory.append(original)
    data = json.loads((LOCAL / DATA_LINK).read_text(encoding='utf-8'))
    scenes = data['eval']
    if args.selected_frames_receipt:
        if args.smoke:
            parser.error('Choose --smoke or --selected-frames-receipt')
        selection = json.loads(args.selected_frames_receipt.read_text(encoding='utf-8'))['selected_frames']
        selected = {(s, str(f)) for s, f in selection}
        if not selected or len(selected) > 100 or len(selected) != len(selection):
            raise ValueError('A bounded, unique existing-run frame selection is required (1..100)')
        scenes = {s: dict(info, frames=[f for f in info['frames'] if (s, str(f['frame_index'])) in selected])
                  for s, info in scenes.items()}
        scenes = {s: info for s, info in scenes.items() if info['frames']}
        if sum(len(info['frames']) for info in scenes.values()) != len(selected):
            raise ValueError('Selected frame not in frozen eval split')
        inventory = [r for r in inventory if len(r['path'].split('/')) >= 4
                     and (r['path'].split('/')[0], r['path'].split('/')[2]) in selected
                     and Path(r['path']).name.startswith(('occ_frame_', 'height_field_frame_'))]
        if len(inventory) != 2 * len(selected):
            raise ValueError('Each selected frame must have OCC and height-field inventory entries')
    if args.smoke:
        scene = next(iter(scenes))
        scenes = {scene: dict(scenes[scene], frames=scenes[scene]['frames'][:3])}
        selected = {(s, str(f['frame_index'])) for s, info in scenes.items() for f in info['frames']}
        def selected_path(row):
            parts = row['path'].split('/')
            if len(parts) == 1:
                return True
            if parts[0] not in scenes:
                return False
            return len(parts) < 4 or (parts[0], parts[2]) in selected
        inventory = [row for row in inventory if selected_path(row)]
    label = ('compile-tensor' if policy['profile'] == 'tensor_only' else 'compile-with-occ') + ('-smoke' if args.smoke else '')
    if args.selected_frames_receipt:
        label += '-selected-' + hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:12]
    PROGRESS_LABEL = label
    if args.plan_only:
        print(json.dumps({'profile':policy['profile'],'scenes':len(scenes),
                          'frames':sum(len(s['frames']) for s in scenes.values()),
                          'files':len(inventory),'bytes':sum(r['size'] for r in inventory),
                          'required_manifest':str(args.required_manifest),'network_access':False}),flush=True)
        return 0
    # Honor the stop before touching cloud credentials or minting object URLs.
    ssh(['python3', '-c', 'import pathlib,sys; sys.exit(75 if pathlib.Path(sys.argv[1]).exists() else 0)',
         REMOTE_ROOT+'/.stop-after-batch'])
    writer = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
    checker = REMOTE_BASE + '/tools/check_required_dataset.py'
    ssh(['python3', '-c', writer, checker], (BASE/'tools/check_required_dataset.py').read_bytes())
    preflight = json.loads(ssh(['python3', checker], json.dumps({'root':REMOTE_ROOT,'files':inventory}).encode()))
    if preflight['conflicts']:
        (LOCAL/(label+'-conflicts.json')).write_text(json.dumps(preflight['conflicts'],indent=2),encoding='utf-8')
        raise ValueError('Unverified conflicting files exist; no data was overwritten')
    verified = set(preflight['verified_paths'])
    pending = [r for r in inventory if r['path'] not in verified]
    completed_files, completed_bytes = preflight['verified_files'], preflight['verified_bytes']
    total_bytes = sum(r['size'] for r in inventory)
    progress = {'state':'downloading', 'updated_at':datetime.now(timezone.utc).isoformat(),
                'completed_files':completed_files,'total_files':len(inventory),
                'completed_bytes':completed_bytes,'total_bytes':total_bytes,
                'reused_files':completed_files,'downloaded_files_this_process':0,
                'downloaded_bytes_this_process':0,'failed':[],'workstation_root':REMOTE_ROOT}
    (LOCAL/(label+'-progress.json')).write_text(json.dumps(progress,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'preflight','verified_files':completed_files,'verified_bytes':completed_bytes,
                      'pending_files':len(pending),'pending_bytes':total_bytes-completed_bytes}),flush=True)
    root = Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data')
    connection = next(c for c in json.loads((root/'integrations.local.json').read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
    with contextlib.redirect_stdout(Quiet()), contextlib.redirect_stderr(Quiet()):
        import asset
        resource = asset.resource('datasets/smartchassisdata-v3/versions/26-10-20-1',
                                  config=asset.config(env='prod', bearer_file=connection['settings']['asset_bearer_file']))
        storage = resource._get_s3_path()
        if storage['path'].rstrip('/') != SOURCE.rstrip('/'):
            raise ValueError('Unexpected dataset version source')
        client = resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
    worker = (BASE/'tools/workstation_download_worker.py').read_bytes()
    writer = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
    remote_worker = REMOTE_BASE + '/tools/workstation_download_worker.py'
    ssh(['python3','-c',writer,remote_worker], worker)
    smoke = {key: {} for key in data}
    smoke['eval'] = scenes; smoke['joint_eval'] = scenes
    if args.smoke:
        ssh(['python3','-c',writer,REMOTE_ROOT+'/smoke_data_link.json'], json.dumps(smoke).encode())
    record = {'source_uri':'s3://s3.bj.bcebos.com/'+storage['bucket']+'/'+SOURCE, 'root':REMOTE_ROOT,
              'scenes':len(scenes),'frames':sum(len(s['frames']) for s in scenes.values()),'files':inventory}
    (LOCAL/(label+'-file-inventory.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    for offset in range(0,len(pending),args.batch_size):
        files = []
        with contextlib.redirect_stdout(Quiet()), contextlib.redirect_stderr(Quiet()):
            # Re-enter the SDK client cache to refresh expiring temporary access.
            client = resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
            for item in pending[offset:offset+args.batch_size]:
                url = client.generate_presigned_url('get_object',Params={'Bucket':storage['bucket'],'Key':SOURCE+item['path']},ExpiresIn=3600)
                if url.startswith('http://'): url='https://'+url[7:]
                files.append(dict(item,url=url))
        response = ssh(['python3',remote_worker],json.dumps({'root':REMOTE_ROOT,'files':files,'workers':args.workers}).encode(),timeout=600)
        result = json.loads(response)
        completed_files += result['files']-len(result['failures']); completed_bytes += result['bytes']
        progress.update({'state':'downloading' if not result['failures'] else 'failed',
                  'updated_at':datetime.now(timezone.utc).isoformat(),
                  'completed_files':completed_files,'total_files':len(inventory),
                  'completed_bytes':completed_bytes,'total_bytes':sum(x['size'] for x in inventory),
                  'failed':result['failures'],'workstation_root':REMOTE_ROOT,
                  'downloaded_files_this_process':progress['downloaded_files_this_process']+result.get('downloaded_files',result['files']-result.get('reused_files',0)-len(result['failures'])),
                  'downloaded_bytes_this_process':progress['downloaded_bytes_this_process']+result.get('downloaded_bytes',result['bytes'])})
        checkpoint=LOCAL/(label+'-progress.json')
        temporary=checkpoint.with_suffix('.new')
        temporary.write_text(json.dumps(progress,indent=2),encoding='utf-8')
        temporary.replace(checkpoint)
        print(json.dumps({'stage':'download','completed_files':completed_files,'total_files':len(inventory),
                          'completed_bytes':completed_bytes,'failed':result['failures']}),flush=True)
        if result['failures']:
            return 2
    progress.update(state='verifying',updated_at=datetime.now(timezone.utc).isoformat())
    (LOCAL/(label+'-progress.json')).write_text(json.dumps(progress,indent=2),encoding='utf-8')
    validation = json.loads(ssh(['python3',checker],json.dumps({'root':REMOTE_ROOT,'files':inventory,'rehash':True}).encode(),timeout=1800))
    validation.pop('verified_paths',None)
    (LOCAL/(label+'-validation.json')).write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if validation['state'] != 'complete':
        raise ValueError('Final content validation is incomplete')
    record = {k:v for k,v in record.items() if k!='files'} | {'file_count':completed_files,'verified_bytes':completed_bytes,'state':'complete','content_rehashed':True}
    ssh(['python3','-c',writer,REMOTE_ROOT+'/'+label+'-sync-receipt.json'],json.dumps(record).encode())
    (LOCAL/(label+'-sync-receipt.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    progress.update(state='complete',updated_at=datetime.now(timezone.utc).isoformat())
    (LOCAL/(label+'-progress.json')).write_text(json.dumps(progress,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        label=PROGRESS_LABEL
        if label is None:
            # Argument/manifest validation did not start a download scope.
            # Never overwrite an unrelated completed scope's progress.
            print(json.dumps({'state':'failed','error_type':type(error).__name__,'scope_started':False}),flush=True)
            raise SystemExit(2)
        path=LOCAL/(label+'-progress.json')
        value=json.loads(path.read_text()) if path.exists() else {}
        value.update(state='paused' if isinstance(error,(BatchStopped,KeyboardInterrupt)) else 'failed',
                     updated_at=datetime.now(timezone.utc).isoformat(),error_type=type(error).__name__)
        path.write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps({'state':value['state'],'error_type':type(error).__name__}),flush=True)
        raise SystemExit(75 if isinstance(error,BatchStopped) else 2)
