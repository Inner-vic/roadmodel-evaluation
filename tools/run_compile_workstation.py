"""Execute one frozen, GT-free compilation-alignment request on the workstation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path('/mnt/data/developer/pcm-roadmodel-eval')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select_frames(link, evaluation):
    if link.get('joint_eval') != link.get('eval'):
        raise ValueError('Different eval and joint_eval scopes require review')
    scenes = list(link['eval'])
    selection = evaluation.get('selection', {})
    explicit = selection.get('frames')
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError('selection.frames must contain scene/frame_index pairs')
        if set(selection) != {'frames'} or evaluation.get('frame_fraction', 1.0) != 1.0:
            raise ValueError('Explicit frames cannot be combined with scene sampling or a fraction')
        if evaluation.get('max_frames', len(explicit)) != len(explicit):
            raise ValueError('Explicit frames cannot be silently truncated')
        grouped, frame_ids = {}, []
        for pair in explicit:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not isinstance(pair[0], str) or type(pair[1]) is not int):
                raise ValueError('Invalid explicit scene/frame_index pair')
            scene, index = pair
            if scene not in link['eval'] or (scene, index) in frame_ids:
                raise ValueError('Unknown scene or duplicate explicit frame')
            matches = [f for f in link['eval'][scene]['frames'] if f['frame_index'] == index]
            if len(matches) != 1:
                raise ValueError('Explicit frame must uniquely identify an eval frame')
            grouped.setdefault(scene, dict(link['eval'][scene], frames=[]))['frames'].append(matches[0])
            frame_ids.append((scene, index))
        # The original dataloader groups frames by scene; preserve that actual order.
        actual = [(scene, f['frame_index']) for scene, row in grouped.items() for f in row['frames']]
        if actual != frame_ids:
            raise ValueError('Explicit frames must be grouped by scene')
        return dict({key: {} for key in link}, eval=grouped, joint_eval=grouped), frame_ids
    count = selection.get('scene_count', len(scenes))
    per_scene = selection.get('frames_per_scene')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(scenes):
        raise ValueError('selection.scene_count exceeds available eval scenes')
    if per_scene is not None and (isinstance(per_scene, bool) or not isinstance(per_scene, int) or per_scene < 1):
        raise ValueError('selection.frames_per_scene must be positive')
    indices = [(i * (len(scenes) - 1)) // (count - 1) for i in range(count)] if count > 1 else [0]
    frames = [(scenes[i], frame) for i in indices for frame in link['eval'][scenes[i]]['frames'][:per_scene]]
    fraction = evaluation.get('frame_fraction', 1.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Invalid frame_fraction')
    take = max(1, int(len(frames) * fraction))
    frames = [frames[i * len(frames) // take] for i in range(take)]
    limit = evaluation.get('max_frames')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('max_frames must be positive')
        frames = frames[:limit]
    if not frames:
        raise ValueError('No eval frames selected')
    grouped = {}
    for scene, frame in frames:
        grouped.setdefault(scene, dict(link['eval'][scene], frames=[]))['frames'].append(frame)
    effective = {key: {} for key in link}
    effective.update(eval=grouped, joint_eval=grouped)
    return effective, [(scene, f['frame_index']) for scene, f in frames]


def run(request):
    import fcntl

    run_id = request['run_id']
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('Invalid run ID')
    if request.get('kind') != 'compile_alignment' or request['evaluation'].get('compute_gt_metrics', False):
        raise ValueError('This driver supports GT-free compilation alignment only')
    execution, dataset, model = request['execution'], request['dataset'], request['model']
    task_root = Path(execution.get('workspace_root', str(ROOT))).resolve()
    package = Path(execution['repo_dir']).resolve()
    run_dir = (Path(execution['output_root']) / run_id).resolve()
    if task_root not in run_dir.parents or run_dir.exists():
        raise ValueError('Use a new output directory inside this task')
    if shutil.disk_usage(task_root).free < 2 * 1024 ** 3:
        raise ValueError('Workstation needs at least 2 GiB free for this small validation run')
    if dataset['class_name'] != 'SmartChassisDatasetV7':
        raise ValueError('This verified input contract is V7 only')
    manifest_path = Path(dataset['required_manifest'])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('profile') != 'tensor_only':
        raise ValueError('Use the audited tensor-only input allowlist')
    dataset_root = Path(dataset['path']).resolve()
    mounted = execution.get('data_access') == 'mounted'
    if not mounted:
        receipt_path = dataset_root / 'compile-tensor-sync-receipt.json'
        data_receipt = json.loads(receipt_path.read_text())
        if (data_receipt.get('state') != 'complete' or not data_receipt.get('content_rehashed')
            or data_receipt.get('file_count') != manifest['file_count']
            or data_receipt.get('verified_bytes') != manifest['total_bytes']
            or data_receipt.get('source_uri') != manifest['source_uri']):
            raise ValueError('Required dataset files are not completely content-verified')
    data_link = Path(dataset['data_link_file'])
    if digest(data_link) != manifest['data_link_sha256']:
        raise ValueError('Dataset index changed')
    onnx, packup = Path(model['onnx']), Path(model['packup_tar'])
    for path, expected in ((onnx, model['onnx_sha256']), (packup, model['packup_sha256'])):
        if not re.fullmatch('[a-fA-F0-9]{64}', expected) or digest(path) != expected.lower():
            raise ValueError('Model input hash changed')
    effective, frame_ids = select_frames(json.loads(data_link.read_text()), request['evaluation'])
    if request['evaluation'].get('expected_frames', len(frame_ids)) != len(frame_ids):
        raise ValueError('Selected scope differs from expected_frames')
    # The selected scene/frame pairs and their source metadata remain unchanged.
    required = {row['path']: row for row in manifest['files']}
    by_frame = {}
    for name, row in required.items():
        by_frame.setdefault(str(Path(name).parent) + '/', []).append(row)
    for scene, index in frame_ids:
        prefix = scene + '/dataset/' + str(index) + '/'
        rows = by_frame.get(prefix, [])
        if {r['role'] for r in rows} != {'raw_lidar', 'camera_120'}:
            raise ValueError('Selected frame is outside the audited two-file input contract')
        for row in rows:
            path = dataset_root / row['path']
            if not path.is_file() or path.stat().st_size != row['size']:
                raise ValueError('A selected input is missing or changed')
            if mounted and row.get('sha256') and digest(path) != row['sha256']:
                raise ValueError('Mounted selected input SHA256 differs')
    for scene in dict.fromkeys(scene for scene,_ in frame_ids):
        name=scene+'/dataset/sensor_param.json'
        row=required.get(name)
        path=dataset_root/name
        if row is None or not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('Selected scene calibration is missing or differs from the input contract')
    lock_path = Path(execution.get('lock_file','/home/lixiang/bench-eval/locks/bench.lock'))
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock = open(lock_path, 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'state': 'bench_busy', 'run_id': run_id}), flush=True)
        return 73
    run_dir.mkdir(parents=True)
    run_evidence = task_root / 'run-evidence' / run_id
    run_evidence.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'pcm.live-compile-receipt/v1', 'state': 'preparing', 'run_id': run_id,
              'started_at': datetime.now(timezone.utc).isoformat(), 'version_name': request['version_name'],
              'requested_frames': len(frame_ids), 'selected_frames': frame_ids,
              'selected_scenes': list(effective['eval']), 'metric_scope': 'tensor_alignment',
              'onnx_sha256': model['onnx_sha256'], 'packup_sha256': model['packup_sha256'],
              'runtime': {'python': sys.version.split()[0]},
              'required_manifest_sha256': digest(manifest_path), 'data_link_sha256': digest(data_link),
              'code_sha256': {name: digest(package / name) for name in
                              ('run_bench_eval.py', 'board_infer_client.py', 'numpy_data_pipeline.py')}}
    record['driver_sha256'] = digest(Path(__file__))
    record['validation_script_sha256'] = digest(task_root / 'tools/validate_compile_outputs.py')
    record['data_access'] = 'mounted' if mounted else 'verified_workstation_cache'
    record['selected_input_sha256'] = {row['path']:digest(dataset_root/row['path']) for scene,index in frame_ids
                                      for row in by_frame[scene+'/dataset/'+str(index)+'/']}
    record['selected_input_sha256'].update({scene+'/dataset/sensor_param.json':digest(dataset_root/scene/'dataset/sensor_param.json')
                                           for scene in dict.fromkeys(scene for scene,_ in frame_ids)})

    def save():
        payload = json.dumps(record, indent=2)
        (run_dir / 'execution_receipt.json').write_text(payload)
        (run_evidence / 'execution.json').write_text(payload)

    save()
    log = (run_evidence / 'console.log').open('wb')
    try:
        effective_path = run_dir / 'effective-data-link.json'
        effective_path.write_text(json.dumps(effective))
        record['effective_data_link_sha256'] = digest(effective_path)
        (run_dir / 'request.json').write_text(json.dumps(request, indent=2))
        with tarfile.open(packup) as archive:
            for member in archive:
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe model archive member')
        sys.path.insert(0, str(package))
        from board_infer_client import BoardInferClient, BoardInferConfig
        from local_defaults import parse_board_ssh
        user, host, port = parse_board_ssh(execution['board_ssh'])
        board_dir = '/userdata/benchEval/models/' + run_id
        with BoardInferClient(BoardInferConfig(host=host, port=port, username=user, packup_dir=board_dir)) as client:
            def checked(command):
                code, stdout, stderr = client._exec(command)
                if code:
                    raise RuntimeError('Board preparation failed (exit ' + str(code) + ')')
                return stdout
            available = int(checked('df -Pk /userdata').splitlines()[-1].split()[3]) * 1024
            with tarfile.open(packup) as archive:
                needed = sum(member.size for member in archive) + packup.stat().st_size + 512 * 1024 ** 2
            if available < needed:
                raise ValueError('Insufficient board storage for an isolated model and frame outputs')
            record['board_disk_preflight'] = {'available_bytes': available, 'required_bytes': needed}
            checked('test ! -e ' + shlex.quote(board_dir) + ' && mkdir ' + shlex.quote(board_dir))
            client._sftp.put(str(packup), board_dir + '/packup.tar.gz')
            if client._exec('sha256sum ' + shlex.quote(board_dir + '/packup.tar.gz'))[1].split()[0] != model['packup_sha256'].lower():
                raise ValueError('Packup transfer hash mismatch')
            checked('cd ' + shlex.quote(board_dir) + ' && tar -xzf packup.tar.gz')
            matches = client._exec('find ' + shlex.quote(board_dir) + ' -name model_info.json -type f')[1].splitlines()
            if len(matches) != 1:
                raise ValueError('Model archive did not contain one unique model_info.json')
            board_packup = str(Path(matches[0]).parent)
        argv = [execution['python'], '-u', 'run_bench_eval.py', '--eval-dataset-root', str(dataset_root),
                '--dataset-path', str(dataset_root), '--data-link-file', str(effective_path),
                '--dataset', dataset['class_name'], '--onnx', str(onnx), '--board-packup', board_packup,
                '--board-ssh', execution['board_ssh'], '--max-frames', str(len(frame_ids)), '--eval-subsample', '100',
                '--skip-onnx-convert', '--device', 'cpu', '--ort-provider', 'CPUExecutionProvider',
                '--skip-metrics', '--no-shm', '--skip-git-check', '--skip-mount', '--no-shared-inputs',
                '--keep-frame-inputs', '--save-dir', str(run_dir)]
        record.update(state='running', argv=argv, board_packup=board_packup)
        save()
        # Keep the bench locked while the child is alive, even if the driver exits.
        result = subprocess.run(argv, cwd=str(package), stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=(lock.fileno(),))
        if result.returncode:
            raise RuntimeError('Evaluation process failed with exit ' + str(result.returncode))
        for script in ('finalize_tensor_alignment.py', 'validate_compile_outputs.py'):
            result = subprocess.run([execution['python'], str(task_root / 'tools' / script), '--run-dir', str(run_dir)],
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Result finalization failed: ' + script)
        summary = json.loads((run_dir / 'cosine_summary.json').read_text())
        if summary.get('num_frames') != len(frame_ids):
            raise ValueError('Compared frame count differs from frozen selected scope')
        record.update(state='succeeded', exit_code=0)
    except Exception as error:
        record.update(state='failed', error_type=type(error).__name__, error=str(error))
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(json.dumps({k: record[k] for k in ('state', 'run_id', 'requested_frames', 'selected_scenes')}), flush=True)
    return 0 if record['state'] == 'succeeded' else 2


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--request')
    source.add_argument('--request-json')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text()) if args.request else json.loads(args.request_json)
    try:
        if request.get('execution',{}).get('gateway'):
            from copy import deepcopy
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
            from pcm_eval.gateway import GatewayTunnel
            isolated=deepcopy(request)
            with GatewayTunnel(isolated['execution']['gateway']) as board_ssh:
                isolated['execution']['board_ssh']=board_ssh
                return run(isolated)
        return run(request)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'state': 'not_started', 'error_type': type(error).__name__, 'message': str(error)}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
