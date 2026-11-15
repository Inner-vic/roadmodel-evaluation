"""Aggregate three exact v3.2.2 frame predictions and reuse the original CPU splat.

One existing frame plus two freshly executed frames; this worker itself performs
no model or board inference and does not claim a hardware lock.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path('/mnt/data/developer/pcm-roadmodel-eval')
PACKAGE = BASE / 'package/benchEval-skill-20261020'
ROOT = BASE / 'subnet-runs/pcm-v322-subnets-coverage-03'
CHILDREN = ['pcm-v322-subnets-one-02', 'pcm-v322-subnets-cover-01', 'pcm-v322-subnets-cover-02']
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_adapter import Counts, adapt_sparse, check_height_geometry, DRIVABLE_IDS
from onnx_localagg_backend import localagg_from_gaussian_bins, ort_outputs_to_packup_bins
from eval_metrics_bridge import MetricsAccumulator


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    assert not ROOT.exists(), 'Preserve prior aggregation; no automatic replay'
    roots = [BASE / 'subnet-runs' / name for name in CHILDREN]
    receipts = [json.loads((r / 'execution_receipt.json').read_text()) for r in roots]
    assert all(r['state'] == 'succeeded' and r['version_name'] == 'v3.2.2' and r['release']['state'] == 'RELEASED' for r in receipts)
    for key in ('onnx_sha256', 'source_onnx_sha256', 'packup_sha256', 'data_link_sha256', 'subnet_archives'):
        assert all(r[key] == receipts[0][key] for r in receipts), key
    selected = [r['selected_frames'][0] for r in receipts]
    assert len(set(map(tuple, selected))) == 3
    ROOT.mkdir()
    dataset = BASE / 'datasets/smartchassisdata-v3/26-10-20-1'
    source_link = dataset / 'data_link_26-10-20-1_train859_eval135_v7.json'
    assert sha(source_link) == receipts[0]['data_link_sha256']
    link = json.loads(source_link.read_text())
    effective = {'eval': {}}
    for scene, fr in selected:
        info = link['eval'][scene]
        effective['eval'][scene] = dict(info, frames=[f for f in info['frames'] if f['frame_index'] == fr])
    save(ROOT / 'effective-data-link.json', effective)
    frames = []
    checks = []
    evidence = []
    acc = {b: Counts() for b in ('board', 'onnx')}
    oracle = {b: MetricsAccumulator(use_cuda=False) for b in acc}
    independent_drive = {b: np.zeros(3, dtype=np.int64) for b in acc}
    cosine_frames = []
    for i, (child, receipt) in enumerate(zip(roots, receipts)):
        assert sha(child / 'reference_fp16.onnx') == receipt['onnx_sha256']
        assert sha(child / 'payload.tar.gz') == receipt['packup_sha256']
        fm = json.loads((child / 'frame_000000/frame_manifest.json').read_text())
        scene, fr = selected[i]
        assert fm['meta']['scene_name'] == scene and fm['meta']['frame_idx'] == fr
        meta = json.loads((child / 'all-subnet-outputs.json').read_text())
        public = json.loads((child / 'public-outputs.json').read_text())
        compared = json.loads((child / 'tensor-comparison.json').read_text())
        reference = json.loads((child / 'onnx_outputs/manifest.json').read_text())
        ref = {r['onnx_output_name']: r for r in reference['outputs']}
        proof = []
        for port in public:
            row = meta[port['tensor']]
            path = child / ('sub' + str(row['stage'])) / row['filename']
            assert sha(path) == row['sha256']
            a = np.load(child / 'onnx_outputs' / ref[port['onnx_name']]['npy_file'], allow_pickle=False)
            b = np.fromfile(path, dtype=row['dtype']).reshape(row['shape'])
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
            den = np.linalg.norm(af) * np.linalg.norm(bf)
            cosine = float(np.dot(af, bf) / den) if den else 1.0 if np.array_equal(a, b) else 0.0
            expected = next(r for r in compared['outputs'] if r['name'] == port['onnx_name'])
            assert abs(expected['cosine'] - cosine) < 1e-10
            proof.append({'name': port['onnx_name'], 'board_sha256': sha(path), 'onnx_sha256': sha(child / 'onnx_outputs' / ref[port['onnx_name']]['npy_file']), 'cosine_recheck': cosine})
            checks.extend({'backend': backend, 'frame_id': fm['frame_id'], 'name': port['onnx_name'], 'finite': True} for backend in acc)
        frame_dir = ROOT / f'frame_{i:06d}'
        frame_dir.mkdir()
        fm['frame_dir'] = str(frame_dir)
        save(frame_dir / 'frame_manifest.json', fm)
        board_bins = {port['filename']: (child / ('sub' + str(meta[port['tensor']]['stage'])) / port['filename']).read_bytes()
                      for port in public if port['onnx_name'] in ('/encoder/layers.67/Concat_3_output_0', '/Reshape_34_output_0', '/Concat_28_output_0', '/MatMul_7_output_0', '/aggregator/Min_output_0', '/aggregator/Cast_1_output_0')}
        assert len(board_bins) == 6
        ort_dump = frame_dir / 'onnx_dump'
        ort_outputs_to_packup_bins(child / 'onnx_outputs', ort_dump)
        bins = {'board': board_bins, 'onnx': {p.name: p.read_bytes() for p in ort_dump.glob('*.bin')}}
        item = next(r for r in link['eval'][scene]['frames'] if r['frame_index'] == fr)
        occ = dataset / scene / 'dataset' / str(fr) / item['occ_file']
        height = dataset / scene / 'dataset' / str(fr) / item['height_field_file']
        raw, hf = np.load(occ, allow_pickle=False), np.load(height, allow_pickle=False)
        geometry = check_height_geometry(raw, hf, [5, -3.6, -2], [.05, .15, .02])
        gt, mask, drive, crop = adapt_sparse(raw, [5, -3.6, -2], [.05, .15, .02], [5, -3.6, -1], [400, 48, 100])
        frame = {'scene': scene, 'frame_index': fr, 'frame_id': fm['frame_id'], 'occ_path': str(occ), 'occ_sha256': sha(occ),
                 'height_sha256': sha(height), 'geometry_check': geometry, 'adaptation': crop, 'predictions': {}}
        for backend in acc:
            body, metadata = localagg_from_gaussian_bins(bins[backend], work_dir=frame_dir / (backend + '_splat'), keep_work=True, adapt_legacy_radii=True)
            folder = frame_dir / ('infer_output' if backend == 'board' else 'onnx_infer_output')
            folder.mkdir()
            prediction = folder / 'final_occ.npy'
            prediction.write_bytes(body)
            pred = np.load(prediction, allow_pickle=False)
            acc[backend].update(pred, gt, mask, drive)
            oracle[backend].update(pred, gt, mask, compute_pillar=False)
            pd = np.isin(pred, DRIVABLE_IDS).reshape(-1, 100)
            gd = (drive == 1).reshape(-1, 100)
            independent_drive[backend] += [int(gd.sum()), sum(int(gd[n, np.flatnonzero(row)[0]]) for n, row in enumerate(pd) if row.any()), int(pd.sum())]
            frame['predictions'][backend] = {'path': str(prediction), 'sha256': sha(prediction), 'postprocess': metadata}
        frames.append(frame)
        cosine_frames.append(json.loads((child / 'cosine_summary.json').read_text())['frames'][0])
        evidence.append({'run_id': receipt['run_id'], 'source_execution_receipt_sha256': sha(child / 'execution_receipt.json'), 'execution_receipt': receipt,
                         'public_tensor_integrity': proof, 'inference_reused': i == 0})
        save(ROOT / 'progress.json', {'state': 'postprocessing', 'completed_frames': len(frames), 'total_frames': 3})
        print(json.dumps({'gt_completed_frames': len(frames), 'total_frames': 3}), flush=True)
    names = [r['onnx_output_name'] for r in cosine_frames[0]['outputs']]
    assert all([r['onnx_output_name'] for r in f['outputs']] == names for f in cosine_frames)
    tensors = {}
    for name in names:
        vals = [next(r['cosine'] for r in f['outputs'] if r['onnx_output_name'] == name) for f in cosine_frames]
        tensors[name] = {'mean_cosine': sum(vals) / 3, 'min_cosine': min(vals), 'num_frames': 3}
    save(ROOT / 'cosine_summary.json', {'num_frames': 3, 'mean_cosine_all': sum(f['mean_cosine'] for f in cosine_frames) / 3,
              'min_cosine_all': min(f['mean_cosine'] for f in cosine_frames), 'tensors': tensors, 'frames': cosine_frames})
    save(ROOT / 'input_validation.json', {'state': 'validated', 'frame_ids': [f['frame_id'] for f in frames],
              'finite_outputs': {'state': 'passed', 'checks': checks}, 'source': 'same-version frozen child input and output hashes', 'dataset_downloads': 0})
    save(ROOT / 'output_integrity.json', {'state': 'fresh_outputs_checked', 'frame_count': 3, 'children': evidence,
              'new_inference_frames': 2, 'reused_inference_frames': 1, 'sample_golden_files_deployed': False})
    execution = deepcopy(receipts[0])
    execution.update(run_id=ROOT.name, requested_frames=3, selected_frames=selected, gt_metrics_computed=True,
                     started_at=min(r['started_at'] for r in receipts), finished_at=max(r['finished_at'] for r in receipts),
                     code_sha256={'driver': sha(Path(__file__)), 'gt_adapter': sha(Path(__file__).with_name('gt_adapter.py'))},
                     guard={'mode': 'original locks verified separately for each child'}, release={'state': 'RELEASED', 'mode': 'each original child released'},
                     composition={'role': 'existing_prediction_aggregation_and_gt_only', 'children': CHILDREN, 'new_inference_frames': 2, 'reused_inference_frames': 1,
                                  'child_execution_receipts': evidence})
    save(ROOT / 'execution_receipt.json', execution)
    save(ROOT / 'state.json', {'total_frames': 3, 'summary': {'done': 3, 'failed': 0, 'pending': 0},
              'config': {'skip_metrics': False, 'dataset': 'SmartChassisDatasetV7', 'version': 'v3.2.2', 'subnet_chain': True, 'postprocess': 'original CPU localagg'},
              'frames': {str(i): {'index': i, 'status': 'done', 'meta': {'scene_name': s, 'frame_idx': fr}} for i, (s, fr) in enumerate(selected)}})
    g = ROOT / 'gt-adapted'
    g.mkdir()
    for backend in acc:
        metrics = acc[backend].summary()
        old = oracle[backend].summary()
        for key in ('present_miou', 'occ_iou'):
            assert abs(metrics[key] - old[key]) < 1e-10
        assert all({k: v for k, v in row.items() if k != 'status'} == original for row, original in zip(metrics['present_class_ious'], old['present_class_ious']))
        assert independent_drive[backend].tolist() == [metrics['drivable_counts'][k] for k in ('seen', 'correct', 'positive')]
        save(g / ('metrics_' + backend + '.json'), metrics)
    adaptation = {'schema': 'pcm.gt-adaptation/v1', 'state': 'validated', 'created_at': datetime.now(timezone.utc).isoformat(),
                  'source_run_id': ROOT.name, 'selected_frames': selected, 'num_frames': 3, 'frames': frames,
                  'geometry': {'source_min': [5, -3.6, -2], 'voxel_size': [.05, .15, .02], 'target_min': [5, -3.6, -1], 'target_shape': [400, 48, 100]},
                  'source_execution_receipt_sha256': sha(ROOT / 'execution_receipt.json'), 'source_cosine_summary_sha256': sha(ROOT / 'cosine_summary.json'),
                  'source_output_integrity_sha256': sha(ROOT / 'output_integrity.json'), 'inference_reused': True, 'new_inference_frames': 0,
                  'implementation_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('gt_adapter.py'), PACKAGE / 'run_localagg_cpu.py', PACKAGE / 'local_aggregate_cpu.py']},
                  **{b + '_metrics_sha256': sha(g / ('metrics_' + b + '.json')) for b in acc}}
    save(g / 'gt_receipt.json', adaptation)
    save(g / 'crosscheck.json', {'state': 'passed', 'frames': 3, 'backends': 2, 'semantic_oracle': 'unmodified original MetricsAccumulator',
              'semantic_oracle_sha256': sha(PACKAGE / 'eval_metrics_bridge.py'), 'drivable_oracle': 'independent per-column first valid Z enumeration',
              'all_class_counts_equal': True, 'ds_counts_equal': True})
    save(ROOT / 'progress.json', {'state': 'succeeded', 'completed_frames': 3, 'total_frames': 3, 'new_inference_frames': 2})
    print(json.dumps({'state': 'succeeded', 'frames': 3, 'new_inference_frames': 2,
                      'classes_with_gt': sum(r['seen'] > 0 for r in acc['board'].summary()['class_support'])}), flush=True)


if __name__ == '__main__':
    main()
