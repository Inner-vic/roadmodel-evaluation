"""Independent original CPU reference sessions; no board access or metric changes."""
import atexit
import os
import time
from pathlib import Path
from types import SimpleNamespace

_ENGINE = None


def initialize_reference(barrier=None):
    global _ENGINE
    from fast_subnet_driver import Engine
    _ENGINE = Engine()
    atexit.register(_ENGINE.close)
    if barrier is not None:barrier.wait(timeout=120)


def ready():
    if _ENGINE is None:raise ValueError('Reference worker is not initialized')
    return os.getpid()


def prepare_reference(job):
    from pcm_eval.common import load_json, digest
    if _ENGINE is None:raise ValueError('Reference worker is not initialized')
    if _ENGINE.binding['onnx_fp16_sha256'] != job['onnx_fp16_sha256']:raise ValueError('Reference model differs')
    root = Path(job['root']); root.mkdir(exist_ok=False)
    start = time.monotonic()
    _ENGINE.prepare(SimpleNamespace(cache_frame=job['cache_frame'], task_root=job['task_root']), root)
    duration = time.monotonic() - start
    if job.get('baseline_root'):
        original = Path(job['baseline_root']) / 'onnx_outputs'
        expected = {p.name: digest(p) for p in original.glob('*.bin')}
        actual = {p.name: digest(p) for p in (root / 'onnx_outputs').glob('*.bin')}
        exact = len(expected) == 18 and expected == actual
    else:exact = None
    return dict(root=str(root), pid=os.getpid(), prepare_seconds=duration,
                all18_bytes_exact=exact, provider='CPUExecutionProvider', threads=0, new_ONNX_inferences=1)
