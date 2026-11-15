"""Real-data preprocessing check: only camera_120, lidar and sensor calibration.

No board/GPU task is launched. Compare every exported input byte-for-byte with
the retained inputs of the earlier real board/ONNX run; record actual file opens.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path('/mnt/data/developer/pcm-roadmodel-eval')
PACKAGE = ROOT / 'package/benchEval-skill-20261020'
SOURCE = ROOT / 'datasets/smartchassisdata-v3/26-10-20-1'
AUDIT = ROOT / 'dataset-audit/compile-input-check'


def main():
    view = AUDIT / 'minimal-view'
    output = AUDIT / 'exported-inputs'
    if view.exists() or output.exists():
        raise ValueError('Audit evidence already exists; preserve it')
    link = json.loads((SOURCE / 'data_link_26-10-20-1_train859_eval135_v7.json').read_text())
    scene, info = next(iter(link['eval'].items()))
    frames = info['frames'][:3]
    selected = dict(info, frames=frames)
    small_link = {key: {} for key in link}
    small_link.update(eval={scene: selected}, joint_eval={scene: selected})
    view.mkdir(parents=True)
    link_path = view / 'data_link.json'
    link_path.write_text(json.dumps(small_link))
    payloads = [Path(scene) / 'dataset/sensor_param.json']
    for frame in frames:
        folder = Path(scene) / 'dataset' / str(frame['frame_index'])
        files = list((SOURCE / folder).iterdir())
        for predicate in (lambda n: n.startswith(str(frame['frame_index']) + '_f_ca_120') and n.endswith('.jpg'),
                          lambda n: n.startswith('lidar_original') and n.endswith('.npy')):
            matches = [p for p in files if predicate(p.name)]
            if len(matches) != 1:
                raise ValueError('Ambiguous required source')
            payloads.append(matches[0].relative_to(SOURCE))
    for relative in payloads:
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(str(SOURCE / relative), str(target))
    opened, original_dataset_reads = set(), set()

    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if view in path.parents:
            opened.add(str(path.relative_to(view)))
        elif SOURCE.resolve() in path.parents:
            original_dataset_reads.add(str(path))

    sys.addaudithook(hook)
    sys.path.insert(0, str(PACKAGE))
    spec = importlib.util.spec_from_file_location('pcm_audited_numpy_pipeline', AUDIT / 'numpy_data_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    os.environ.update(DATASET_PATH=str(view), DATA_LINK_FILE=str(link_path), DATASET='SmartChassisDatasetV7')
    pipe = module.NumpyBenchDataPipeline()
    pipe.setup(save_dir=str(output), prefer_shm=False, max_frames=3,
               eval_subsample=100, load_ground_truth=False)
    manifests = list(pipe.iter_export())
    checks = []
    for index, manifest in enumerate(manifests):
        frame = Path(manifest.frame_dir)
        baseline = ROOT / 'runs/pcm-v7-tensor-smoke-004' / ('frame_{:06d}'.format(index))
        if manifest.gt_paths:
            raise ValueError('GT sidecars must not be synthesized for tensor-only evaluation')
        for name, filename in manifest.board_inputs.items():
            actual = hashlib.sha256((frame / filename).read_bytes()).hexdigest()
            previous = hashlib.sha256((baseline / filename).read_bytes()).hexdigest()
            checks.append({'frame': index, 'input': name, 'sha256': actual,
                           'baseline_sha256': previous, 'identical': actual == previous})
    allowed = {str(p) for p in payloads} | {'data_link.json'}
    passed = (len(manifests) == 3 and len(checks) == 33 and all(c['identical'] for c in checks)
              and opened == allowed and not original_dataset_reads
              and not list(output.rglob('gt_*')))
    result = {'schema': 'pcm.minimal-compile-input-check/v1',
              'state': 'passed' if passed else 'failed', 'frames': len(manifests),
              'input_comparisons': len(checks), 'all_inputs_identical': all(c['identical'] for c in checks),
              'dataset_files_opened': sorted(opened), 'opened_file_count': len(opened),
              'minimal_source_bytes': sum((view / name).stat().st_size for name in allowed),
              'original_full_dataset_reads': sorted(original_dataset_reads),
              'gt_sidecars': len(list(output.rglob('gt_*'))), 'checks': checks,
              'baseline_run': 'pcm-v7-tensor-smoke-004',
              'adapted_numpy_script_sha256': hashlib.sha256((AUDIT / 'numpy_data_pipeline.py').read_bytes()).hexdigest()}
    (AUDIT / 'input-check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'dataset_files_opened')}))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
