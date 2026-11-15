"""Run on the workstation: check every retained board/ORT output in a real run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if (root / 'input_validation.json').exists() and not args.verify_existing:
        raise ValueError('Validation evidence already exists; preserve the original receipt')
    state = json.loads((root / 'state.json').read_text())
    tensor_only = state.get('config', {}).get('skip_metrics') is True
    checks, issues, frames, freshness, input_checks = [], [], [], [], []
    for folder in sorted(root.glob('frame_*')):
        detail_file = folder / 'cosine_board_onnx.json'
        if not detail_file.is_file():
            continue
        detail = json.loads(detail_file.read_text())
        manifest = json.loads((folder / 'frame_manifest.json').read_text())
        inputs = manifest.get('board_inputs', {})
        if len(inputs) != 11:
            issues.append(folder.name + ': expected 11 input tensors for the verified reference contract')
        for alias, name in inputs.items():
            path = (folder / name).resolve()
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size < 1:
                issues.append(folder.name + ': missing input ' + alias)
                continue
            input_checks.append({'frame': folder.name, 'input': alias, 'size': path.stat().st_size,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        index = str(int(folder.name.split('_')[-1]))
        reset = state.get('frames', {}).get(index, {}).get('meta', {}).get('output_reset', {})
        if (reset.get('policy') != 'declared_outputs_removed_before_inference'
                or reset.get('declared_output_count') != detail.get('num_onnx_outputs')):
            issues.append(folder.name + ': missing verified pre-inference output reset')
        freshness.append({'frame': folder.name, 'output_reset': reset})
        expected = detail.get('num_onnx_outputs')
        if not isinstance(expected, int) or expected < 1:
            issues.append(folder.name + ': missing expected ONNX output count')
        outputs = detail.get('outputs', [])
        counts = {'board': 0, 'onnx': 0}
        for entry in outputs:
            for backend, directory, file_key, dtype_key in (
                ('board', 'board_dir', 'board_file', 'dtype_board'),
                ('onnx', 'ort_dir', 'onnx_file', 'dtype_onnx'),
            ):
                name = entry.get(file_key)
                if backend == 'onnx':
                    candidates = list(Path(detail[directory]).glob(entry['onnx_stem'] + '.*.bin'))
                    name = str(candidates[0]) if len(candidates) == 1 else None
                if not name:
                    issues.append(folder.name + ': missing ' + file_key)
                    continue
                path = Path(name)
                if not path.is_absolute():
                    path = Path(detail[directory]) / path
                path = path.resolve()
                try:
                    path.relative_to(root)
                    dtype = np.dtype(entry[dtype_key])
                    value = np.fromfile(path, dtype=dtype)
                    finite = bool(value.size == entry.get('numel') and value.size > 0 and np.isfinite(value).all())
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                except (ValueError, OSError, KeyError):
                    issues.append(folder.name + ': invalid ' + backend + ' output')
                    continue
                counts[backend] += 1
                checks.append({'frame': folder.name, 'backend': backend,
                               'output': entry['onnx_output_name'], 'finite': finite,
                               'elements': int(value.size), 'sha256': digest})
        if any(count != expected for count in counts.values()):
            issues.append(folder.name + ': output coverage mismatch')
        frames.append({'frame': folder.name, 'expected_each_backend': expected, 'counts': counts})
    if len(frames) != state.get('total_frames') or not frames:
        issues.append('Compared frame count differs from the run state')
    if tensor_only and list(root.rglob('gt_*')):
        issues.append('Unexpected GT sidecars exist in a tensor-only run')
    passed = not issues and all(item['finite'] for item in checks)
    record = {'schema': 'pcm.input-validation/v1',
              'finite_outputs': {'state': 'passed' if passed else 'failed',
                                 'backend_counts': {b: sum(c['backend'] == b for c in checks)
                                                    for b in ('board', 'onnx')},
                                 'frames': frames, 'checks': checks, 'issues': issues},
              'ground_truth': {'state': 'not_requested' if tensor_only else 'not_validated'}}
    record['input_tensors'] = {'frames': len(frames), 'tensor_files': len(input_checks), 'checks': input_checks}
    integrity = {
        'schema': 'pcm.output-integrity/v1', 'state': 'fresh_outputs_checked' if passed else 'validation_failed',
        'frames': freshness, 'issues': issues}
    for name, value in [('input_validation.json', record), ('output_integrity.json', integrity)]:
        path = root / name
        if path.exists() and args.verify_existing:
            if json.loads(path.read_text()) != value:
                raise ValueError('Existing validation differs from retained content: ' + name)
        else:
            with path.open('x') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'frames': len(frames), 'tensor_checks': len(checks),
                      'finite': record['finite_outputs']['state'], 'issues': issues}))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
