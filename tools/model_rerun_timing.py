"""Monotonic stage timing for a fresh model run; no inference/result reuse."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def event(path, stage, state, **fields):
    row = dict(stage=stage, state=state, utc=utc_now(), monotonic_ns=time.monotonic_ns(), **fields)
    with Path(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


@contextmanager
def span(path, stage, **fields):
    token = uuid.uuid4().hex
    start = event(path, stage, 'STARTED', token=token, **fields)
    try:
        yield
    except BaseException as exc:
        event(path, stage, 'FAILED', token=token,
              duration_seconds=(time.monotonic_ns()-start['monotonic_ns'])/1e9,
              error_type=type(exc).__name__)
        raise
    else:
        event(path, stage, 'SUCCEEDED', token=token,
              duration_seconds=(time.monotonic_ns()-start['monotonic_ns'])/1e9)


def classify(argv):
    command = ' '.join(map(str, argv)) if isinstance(argv, (tuple, list)) else str(argv)
    for text, stage in [
        ('formal_model_stage_remote.py', 'data_export_and_GT_adaptation'),
        ('cuda.mem_get_info', 'GPU_memory_probe'),
        ('shutil.rmtree', 'staged_input_cleanup'),
        ("files=[p for p in b.rglob", 'result_download'),
        ("with tarfile.open(b.parent/'input-transfer.tar')", 'input_unpack_and_validation'),
        ('actual=h.hexdigest()', 'input_upload'),
        ('subprocess.Popen', 'native_dispatch'),
        ('observed_progress', 'remote_progress_poll'),
        ("'observed':", 'remote_progress_poll'),
        ('File content differs', 'material_upload'),
    ]:
        if text in command:
            return stage
    return 'remote_command'


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines()]
    totals = {}
    for row in rows:
        if 'duration_seconds' in row:
            entry = totals.setdefault(row['stage'], {'calls': 0, 'seconds': 0., 'failed': 0})
            entry['calls'] += 1
            entry['seconds'] += row['duration_seconds']
            entry['failed'] += row['state'] == 'FAILED'
    return totals
