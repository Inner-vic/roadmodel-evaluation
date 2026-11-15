"""Bounded ordered prefetch of unchanged per-frame CPU reference evaluations."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import load_json, digest
from .reference_workers import initialize_reference, prepare_reference


def settings(path, frozen_sha):
    path = Path(path)
    if not path.exists():return {'workers': 1}
    config = load_json(path); proof = Path(config['verification_path'])
    if digest(proof) != config['verification_sha256']:raise ValueError('Reference verification changed')
    value = load_json(proof)
    if value.get('state') != 'VERIFIED' or not value.get('all_selected18_bytes_exact') or not value.get('same_original_session_options'):
        raise ValueError('Reference preparation has not passed exact validation')
    workers = config['workers']
    if type(workers) is not int or workers not in (1, 2, 4) or workers != value['selected_workers']:
        raise ValueError('Unvalidated reference worker configuration')
    if value['frozen_request_sha256'] != frozen_sha or config['onnx_fp16_sha256'] != value['reference_onnx_sha256']:
        raise ValueError('Reference validation covers a different frozen task')
    case = [r for r in value['cases'] if r['workers'] == workers]
    if len(case) != 1 or not case[0]['all18_bytes_exact']:raise ValueError('Reference worker count is not verified')
    return config


def requires_reference(root):
    return not ((root / 'execution_receipt.json').exists() or (root / 'prepare.json').exists())


class ReferencePrefetch:
    def __init__(self, engine, indices, frames, exporter, output_root, task_root, config):
        if engine.binding['onnx_fp16_sha256'] != config['onnx_fp16_sha256']:raise ValueError('Reference model changed')
        self.engine = engine; self.frames = frames; self.exporter = exporter
        self.output_root = Path(output_root); self.task_root = str(task_root)
        self.pool = ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn'), initializer=initialize_reference)
        self.ordered = iter(indices); self.pending = {}; self.closed = False
        for _ in range(config['workers'] * 2):self._enqueue()

    def _enqueue(self):
        try:index = next(self.ordered)
        except StopIteration:return
        self.pending[index] = self.engine.producers.submit(self._submit, index)

    def _submit(self, index):
        root = self.output_root / f'frame_{index:06d}'
        if (root / 'execution_receipt.json').exists():
            if load_json(root / 'execution_receipt.json')['selected_frames'] != [self.frames[index]]:
                raise ValueError('Retained native frame differs')
            return root
        if (root / 'prepare.json').exists():
            prepared = load_json(root / 'prepare.json'); meta = load_json(root / 'input-reuse.json')['frame']['meta']
            if prepared['state'] != 'READY' or prepared['onnx_fp16_sha256'] != self.engine.binding['onnx_fp16_sha256'] or [meta['scene_name'], meta['frame_idx']] != self.frames[index]:
                raise ValueError('Retained reference frame differs')
            return root
        if root.exists():raise ValueError('Partial preparation requires reconciliation, never replay')
        manifest = next(self.exporter)
        if [manifest.meta['scene_name'], manifest.meta['frame_idx']] != self.frames[index]:raise ValueError('Reference prefetch order differs')
        return self.pool.submit(prepare_reference, {'root': str(root), 'cache_frame': str(manifest.frame_dir),
                                'task_root': self.task_root, 'onnx_fp16_sha256': self.engine.binding['onnx_fp16_sha256']})

    def get(self, index):
        value = self.pending.pop(index).result()
        root = value if isinstance(value, Path) else Path(value.result()['root'])
        self._enqueue()
        return root

    def close(self):
        if self.closed:return
        self.closed = True
        for future in self.pending.values():future.cancel()
        self.engine.producers.shutdown(wait=True, cancel_futures=True)
        self.pool.shutdown(wait=True, cancel_futures=True)
