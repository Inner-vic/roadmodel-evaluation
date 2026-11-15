"""Small-scope compilation workflow: immutable attempts, verified resume, batch retries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_compile_workstation import ROOT, digest, run as run_leaf, select_frames


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_once(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Evidence changed: ' + str(path))
        return
    with path.open('x') as stream:
        stream.write(payload)


def save_progress(root, value):
    temp = root / 'workflow.next.json'
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(root / 'workflow.json')


def verify_batch(path, pairs, commit=None):
    receipt = json.loads((path / 'execution_receipt.json').read_text())
    if receipt['state'] != 'succeeded' or receipt['selected_frames'] != pairs:
        raise ValueError('Batch is not complete for the frozen frame list')
    integrity = json.loads((path / 'output_integrity.json').read_text())
    validation = json.loads((path / 'input_validation.json').read_text())
    if (integrity['state'] != 'fresh_outputs_checked'
            or validation['finite_outputs']['state'] != 'passed'):
        raise ValueError('Batch outputs did not pass provenance/content validation')
    files = sorted(set(path.glob('*.json')) | set(path.glob('frame_*/*.json')))
    hashes = {p.relative_to(path).as_posix(): digest(p) for p in files}
    if commit is not None and hashes != commit:
        raise ValueError('Committed batch metadata changed; preserve and investigate it')
    details = {}
    for row in validation['finite_outputs']['checks']:
        frame = row['frame']
        if frame not in details:
            details[frame] = json.loads((path / frame / 'cosine_board_onnx.json').read_text())
        detail = details[frame]
        output = next(o for o in detail['outputs'] if o['onnx_output_name'] == row['output'])
        if row['backend'] == 'board':
            tensor = Path(detail['board_dir']) / output['board_file']
        else:
            candidates = list(Path(detail['ort_dir']).glob(output['onnx_stem'] + '.*.bin'))
            if len(candidates) != 1:
                raise ValueError('Missing or ambiguous retained ONNX tensor')
            tensor = candidates[0]
        tensor.resolve().relative_to(path.resolve())
        if digest(tensor) != row['sha256']:
            raise ValueError('Retained output content changed')
    for row in validation['input_tensors']['checks']:
        manifest = json.loads((path / row['frame'] / 'frame_manifest.json').read_text())
        tensor = path / row['frame'] / manifest['board_inputs'][row['input']]
        tensor.resolve().relative_to(path.resolve())
        if digest(tensor) != row['sha256']:
            raise ValueError('Retained input content changed')
    return hashes


def fingerprint(request):
    package = Path(request['execution']['repo_dir'])
    paths = [Path(__file__), ROOT / 'tools/run_compile_workstation.py',
             ROOT / 'tools/finalize_tensor_alignment.py', ROOT / 'tools/validate_compile_outputs.py',
             Path(request['dataset']['required_manifest']), Path(request['dataset']['data_link_file']),
             Path(request['model']['onnx']), Path(request['model']['packup_tar'])]
    paths += [package / n for n in ('run_bench_eval.py', 'board_infer_client.py',
                                   'numpy_data_pipeline.py', 'cosine_compare.py')]
    return {'request_sha256': hashlib.sha256(canonical(request).encode()).hexdigest(),
            'files': {str(p): digest(p) for p in paths}}


def aggregate(root, request, workflow):
    frames, selected, index = {}, [], 0
    sample = None
    for batch in workflow['batches']:
        leaf = root / batch['committed_path']
        verify_batch(leaf, batch['pairs'], batch['hashes'])
        state = json.loads((leaf / 'state.json').read_text())
        sample = state
        if state['total_frames'] != len(batch['pairs']):
            raise ValueError('Batch frame coverage mismatch')
        for local, pair in enumerate(batch['pairs']):
            name = 'frame_{:06d}'.format(index)
            target = root / name
            source = leaf / 'frame_{:06d}'.format(local)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError('Aggregate frame link changed')
            elif target.exists():
                raise ValueError('Unexpected aggregate frame directory')
            else:
                target.symlink_to(source.relative_to(root), target_is_directory=True)
            frames[str(index)] = state['frames'][str(local)]
            selected.append(pair)
            index += 1
    write_once(root / 'state.json', {'config': sample['config'], 'total_frames': index,
                                    'summary': {'done': index, 'failed': 0, 'pending': 0}, 'frames': frames})
    write_once(root / 'execution_receipt.json', {
        'schema': 'pcm.live-compile-receipt/v1', 'state': 'succeeded', 'exit_code': 0,
        'run_id': request['run_id'], 'version_name': request['version_name'],
        'requested_frames': index, 'selected_frames': selected,
        'selected_scenes': list(dict.fromkeys(p[0] for p in selected)),
        'metric_scope': 'tensor_alignment', 'batch_workflow': True,
        'onnx_sha256': request['model']['onnx_sha256'], 'packup_sha256': request['model']['packup_sha256'],
        'frozen_fingerprint': workflow['fingerprint'], 'batches': workflow['batches']})
    for script in ('finalize_tensor_alignment.py', 'validate_compile_outputs.py'):
        result = subprocess.run([request['execution']['python'], str(ROOT / 'tools' / script),
                                 '--run-dir', str(root), '--verify-existing'], check=False)
        if result.returncode:
            raise ValueError('Aggregate finalization failed: ' + script)


def run(request, *, resume=False, batch_limit=None, leaf_runner=run_leaf):
    import fcntl
    run_id = request['run_id']
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,39}', run_id):
        raise ValueError('Batch run ID must be at most 40 letters/digits/underscores/hyphens')
    if request.get('kind') != 'compile_alignment' or request['evaluation'].get('compute_gt_metrics', False):
        raise ValueError('Only GT-free compilation alignment is supported')
    root = (Path(request['execution']['output_root']) / run_id).resolve()
    if ROOT.resolve() not in root.parents or root.exists() != resume:
        raise ValueError('Use a new run ID, or explicit --resume for an existing workflow')
    link = json.loads(Path(request['dataset']['data_link_file']).read_text())
    effective, selected = select_frames(link, request['evaluation'])
    pairs = [list(x) for x in selected]
    # Business has requested small-scene validation; full eval must not be implicit.
    if len(pairs) > 20 or request['evaluation'].get('expected_frames') != len(pairs):
        raise ValueError('Small validation requires expected_frames matching at most 20 selected frames')
    size = request['evaluation'].get('batch_size', 2)
    retries = request['evaluation'].get('batch_retries', 0)
    if type(size) is not int or not 1 <= size <= 20 or type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError('batch_size must be 1..20 and batch_retries 0..2')
    if batch_limit is not None and (type(batch_limit) is not int or batch_limit < 1):
        raise ValueError('batch-limit must be positive')
    required = 2 * 1024 ** 3 + len(pairs) * 256 * 1024 ** 2
    available = shutil.disk_usage(ROOT).free
    if available < required:
        raise ValueError('Insufficient workstation storage for retained validation artifacts')
    current = fingerprint(request)
    if resume:
        workflow = None  # Read under the lock so progress cannot be stale.
    else:
        root.mkdir(parents=True)
        write_once(root / 'request.json', request)
        write_once(root / 'effective-data-link.json', effective)
        workflow = {'schema': 'pcm.compile-batches/v1', 'state': 'running', 'fingerprint': current,
                    'artifact_retention': 'keep_all; no automatic deletion',
                    'disk_budget': {'available_bytes': available, 'required_bytes': required,
                                    'estimate_bytes_per_frame': 256 * 1024 ** 2, 'reserve_bytes': 2 * 1024 ** 3},
                    'batches': [{'pairs': pairs[i:i + size], 'attempts': []}
                                for i in range(0, len(pairs), size)]}
        save_progress(root, workflow)
    lock = (root / '.workflow.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('This workflow is already running')
    try:
        if resume:
            workflow = json.loads((root / 'workflow.json').read_text())
            if workflow['fingerprint'] != current or json.loads((root / 'request.json').read_text()) != request:
                raise ValueError('Resume refused: request/model/data/code identity changed')
            if json.loads((root / 'effective-data-link.json').read_text()) != effective:
                raise ValueError('Frozen frame metadata changed')
        performed = 0
        for number, batch in enumerate(workflow['batches']):
            if 'committed_path' in batch:
                verify_batch(root / batch['committed_path'], batch['pairs'], batch['hashes'])
                continue
            # A successful leaf may have finished just before controller interruption.
            for attempt in batch['attempts']:
                path = root / attempt['path']
                receipt = path / 'execution_receipt.json'
                if receipt.is_file() and json.loads(receipt.read_text()).get('state') == 'succeeded':
                    batch.update(committed_path=attempt['path'], hashes=verify_batch(path, batch['pairs']))
                    save_progress(root, workflow)
                    break
            if 'committed_path' in batch:
                continue
            if batch_limit is not None and performed >= batch_limit:
                workflow['state'] = 'checkpointed'
                save_progress(root, workflow)
                return 75
            for _ in range(retries + 1):
                child = copy.deepcopy(request)
                child_id = run_id + '-b{:03d}-a{:02d}'.format(number, len(batch['attempts']) + 1)
                child['run_id'] = child_id
                child['execution']['output_root'] = str(root / 'batches')
                child['execution']['driver_script'] = str(ROOT / 'tools/run_compile_workstation.py')
                child['evaluation'].update(selection={'frames': batch['pairs']}, frame_fraction=1.0,
                                           max_frames=len(batch['pairs']), expected_frames=len(batch['pairs']))
                relative = 'batches/' + child_id
                attempt = {'path': relative, 'state': 'running'}
                batch['attempts'].append(attempt)
                save_progress(root, workflow)
                code = leaf_runner(child)
                attempt.update(state='succeeded' if code == 0 else 'failed', exit_code=code)
                if code == 0:
                    batch.update(committed_path=relative, hashes=verify_batch(root / relative, batch['pairs']))
                    break
                # Do not retry another user's busy bench.
                if code == 73:
                    break
            performed += 1
            save_progress(root, workflow)
            if 'committed_path' not in batch:
                workflow['state'] = 'failed'
                save_progress(root, workflow)
                return 2
        aggregate(root, request, workflow)
        workflow['state'] = 'succeeded'
        save_progress(root, workflow)
        print(json.dumps({'state': 'succeeded', 'frames': len(pairs), 'batches': len(workflow['batches'])}))
        return 0
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-json', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-limit', type=int)
    args = parser.parse_args()
    try:
        code = run(json.loads(args.request_json), resume=args.resume, batch_limit=args.batch_limit)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'state': 'failed', 'error': str(error)}))
        code = 2
    raise SystemExit(code)
