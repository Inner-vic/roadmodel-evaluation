"""One private board process; original five executables and 49 validators."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time

from board_validate_subnet import checked_root, initialize_frame, validate_stage


def sha(body):
    return hashlib.sha256(body).hexdigest()


def add_bytes(archive, name, body):
    member = tarfile.TarInfo(name); member.size = len(body); member.mode = 0o600
    archive.addfile(member, io.BytesIO(body))


def read_packet(packet, expected_sha, root, cache):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha):raise ValueError('Invalid packet digest')
    if packet != Path(str(root) + '.request.tar') or packet.is_symlink():raise ValueError('Wrong packet namespace')
    if packet.stat().st_size > 128 * 1024**2:raise ValueError('Input packet too large')
    raw = packet.read_bytes()
    if sha(raw) != expected_sha:raise ValueError('Input packet digest differs')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        if len(names) != len(set(names)):raise ValueError('Duplicate input packet member')
        for m in members:
            path = PurePosixPath(m.name)
            if not m.isfile() or path.is_absolute() or '..' in path.parts or m.size > 64 * 1024**2:
                raise ValueError('Unsafe input packet member')
        request = json.loads(archive.extractfile('request.json').read())
        if request['schema'] != 'roadmodel.board-batch/v1' or request['root'] != str(root) or request['cache'] != str(cache):
            raise ValueError('Packet belongs to a different private frame')
        if len(request['inputs']) != 11 or len({p['filename'] for p in request['inputs']}) != 11 or [s['stage'] for s in request['stages']] != list(range(5)) or len(request['public_outputs']) != 18:
            raise ValueError('Incomplete five-subnet inventory')
        expected = {'request.json'}
        for p in request['inputs']:
            if Path(p['filename']).name != p['filename'] or p['filename'] in ('.', '..'):raise ValueError('Unsafe input filename')
            expected.add('inputs/' + p['filename'])
        for stage in request['stages']:
            expected.update('sub%d/%s' % (stage['stage'], name) for name in ('model_info.json', 'port-contract.json'))
        if set(names) != expected:raise ValueError('Input packet inventory differs')
        files = {name: archive.extractfile(name).read() for name in names if name != 'request.json'}
    for p in request['inputs']:
        body = files['inputs/' + p['filename']]
        if len(body) != p['nbytes'] or sha(body) != p['sha256']:raise ValueError('Input tensor identity differs')
    for stage in request['stages']:
        for name, key in (('model_info.json', 'model_info_sha256'), ('port-contract.json', 'contract_sha256')):
            if sha(files['sub%d/%s' % (stage['stage'], name)]) != stage[key]:raise ValueError('Subnet metadata differs')
    contracts = [json.loads(files['sub%d/port-contract.json' % s['stage']]) for s in request['stages']]
    if [c['stage'] for c in contracts] != list(range(5)) or sum(len(c['outputs']) for c in contracts) != 49:
        raise ValueError('Incomplete output inventory')
    ports = {(c['stage'], p['tensor']): p for c in contracts for p in c['outputs']}
    if len(ports) != 49:raise ValueError('Duplicate output tensor')
    for public in request['public_outputs']:
        p = ports[(public['stage'], public['tensor'])]
        if any(p[k] != public[k] for k in ('filename', 'nbytes', 'dtype')):raise ValueError('Public output contract differs')
    if len({(p['stage'], p['tensor']) for p in request['public_outputs']}) != 18:raise ValueError('Duplicate public output')
    return request, files, contracts


def execute(root, cache, packet, expected_sha, run=subprocess.run):
    root = checked_root(root); cache = Path(cache); packet = Path(packet)
    if cache.parent != root.parent or not cache.name.startswith('pcm-v322-fast-cache-'):
        raise ValueError('Wrong immutable cache')
    if root.exists():raise ValueError('Never replay an existing private frame')
    started = time.monotonic(); request, files, contracts = read_packet(packet, expected_sha, root, cache)
    input_check_seconds = time.monotonic() - started
    initialize_frame(root, cache)
    for name, body in files.items():
        target=root/name
        # Replace only private metadata links; never write through into the model cache.
        if target.is_symlink() and name.endswith('/port-contract.json'):target.unlink()
        with target.open('xb') as stream:stream.write(body)
    result = {'state': 'RUNNING', 'request_sha256': expected_sha, 'root': str(root), 'stages': [], 'outputs': {},
              'input_verification_seconds': input_check_seconds, 'original_validator_reused': True}
    try:
        for contract in contracts:
            stage = contract['stage']; directory = root / ('sub' + str(stage)); tick = time.monotonic()
            # Same executable, arguments, environment and working directory as the original client.
            value = run(['InferenceExec', 'model_info.json'], cwd=str(directory), capture_output=True, timeout=180,
                        env=dict(os.environ, PWD=str(directory)))
            native_seconds = time.monotonic() - tick
            (root / ('sub%d.stdout.log' % stage)).write_bytes(value.stdout)
            (root / ('sub%d.stderr.log' % stage)).write_bytes(value.stderr)
            row = {'stage': stage, 'exit_code': value.returncode, 'input_count': len(contract['inputs']),
                   'output_count': len(contract['outputs']), 'InferenceExec_seconds': native_seconds}
            result['stages'].append(row)
            if value.returncode:raise ValueError('Original subnet failed: %d' % stage)
            tick = time.monotonic(); checked = validate_stage(root, contract)
            row.update(state='SUCCEEDED', outputs=checked, validation_seconds=time.monotonic() - tick)
            row['seconds'] = native_seconds + row['validation_seconds']
            for meta in checked:result['outputs'][meta['tensor']] = meta
        if len(result['outputs']) != 49:raise ValueError('Missing checked outputs')
        result.update(state='SUCCEEDED', native_total_seconds=time.monotonic() - started)
        bundle = root / 'result.tar'
        with tarfile.open(str(bundle), mode='w:') as archive:
            add_bytes(archive, 'result.json', json.dumps(result).encode())
            for stage in range(5):
                for suffix in ('stdout', 'stderr'):
                    name = 'sub%d.%s.log' % (stage, suffix); add_bytes(archive, name, (root / name).read_bytes())
            for public in request['public_outputs']:
                meta = result['outputs'][public['tensor']]; body = Path(meta['remote_file']).read_bytes()
                if len(body) != meta['nbytes'] or sha(body) != meta['sha256']:raise ValueError('Output changed during collection')
                add_bytes(archive, 'sub%d/%s' % (meta['stage'], meta['filename']), body)
        body = bundle.read_bytes()
        return {'state': 'SUCCEEDED', 'bundle_path': str(bundle), 'nbytes': len(body), 'sha256': sha(body),
                'request_sha256': expected_sha, 'checked_outputs': 49, 'public_outputs': 18}
    except Exception as error:
        result.update(state='NEEDS_REVIEW', error_type=type(error).__name__, error=str(error)[:500])
        (root / 'batch-failure.json').write_text(json.dumps(result)); raise


if __name__ == '__main__':
    if len(sys.argv) != 5:raise ValueError('Expected private root, cache, input packet and SHA256')
    print(json.dumps(execute(*sys.argv[1:])))
