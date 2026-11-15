"""Validated, bounded GT worker configuration for the paused native controller."""
from pathlib import Path
from .common import load_json, digest


def validated_settings(config_path):
    config = load_json(config_path)
    proof = Path(config['verification_path'])
    if digest(proof) != config['verification_sha256']:
        raise ValueError('GT verification receipt changed')
    verification = load_json(proof)
    if verification.get('state') != 'VERIFIED' or not verification.get('all_voxel_labels_metrics_and_counts_exact'):
        raise ValueError('GT optimization has not passed exact validation')
    if not verification.get('production_checkpoint_unchanged') or verification.get('native_inference_frames') != 0:
        raise ValueError('Invalid bounded validation provenance')
    if config.get('backend') != 'memory_optimized' or digest(Path(__file__).with_name('memory_gt.py')) != config['memory_kernel_sha256']:
        raise ValueError('Unreviewed GT kernel')
    workers = config['workers']
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError('Unvalidated GT worker count')
    cases = [r for r in verification['cases'] if r.get('backend') == config['backend'] and r['workers'] == workers and r['threads_per_worker'] == 1]
    if len(cases) != 1 or not cases[0]['all_voxel_labels_and_metrics_exact']:
        raise ValueError('Selected GT configuration was not measured')
    return config


def initialize_native_gt(threads, backend):
    from .parallel_gt import initialize_gt_worker
    initialize_gt_worker(threads, backend)
    if backend == 'memory_optimized':
        from .memory_gt import install
        install()
