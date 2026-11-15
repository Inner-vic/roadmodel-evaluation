"""Two-frame, stage-major execution of the original five compiled subnetworks."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
from board_batch_subnet import read_packet, add_bytes
from board_validate_subnet import checked_root, initialize_frame


def public_ports(graph, ports):
    sources = {p['tensor']:s['stage'] for s in graph['stages'] for p in s['outputs']}
    if len(sources) != sum(len(s['outputs']) for s in graph['stages']):
        raise ValueError('Ambiguous tensor producer')
    return [dict(p, stage=sources[p['tensor']]) for p in ports]


def stage_info(templates, stage, directories):
    """Preserve actual subnet port order; never use the whole-network INPUT_SPECS."""
    count=len(templates)
    if count not in (1,2,4,8) or len(directories) != count:
        raise ValueError('Supported frame counts: 1, 2, 4, 8')
    info = copy.deepcopy(templates[0])
    if len(info['input']) != len(stage['inputs']) or len(info['output']) != len(stage['outputs']):
        raise ValueError('Subnet port count differs')
    for slot, other in enumerate(templates):
        if len(other['input']) != len(info['input']) or len(other['output']) != len(info['output']):
            raise ValueError('Frame subnet metadata differs')
    for i, port in enumerate(info['input']):
        paths = [t['input'][i]['data_paths'] for t in templates]
        if any(len(p) != 1 for p in paths):
            raise ValueError('Expected one source path per frame')
        port['data_paths'] = [p[0] for p in paths]
    for i, port in enumerate(info['output']):
        basename = Path(stage['outputs'][i]['declared_data_path']).name
        paths = [str(p/'output'/basename) for p in directories]
        port.update(data_paths=paths, data_path=paths[0], golden_paths=[])
        if 'fm_golden_paths' in port:
            port['fm_golden_paths'] = []
    info.update(infer_num=count, buffer_batch_num=1, do_infer=True, do_compare=False, do_dump=True)
    return info


def collect_stage(shared, frame, stage, slot, peer_frames=(), *, allow_existing=False):
    """Attribute prefixed dumps by slot and bare dumps only by private frame directory."""
    own = frame/('sub'+str(stage['stage']))
    exports = own/'export'
    result = []
    for port in stage['outputs']:
        names = {port['filename'], Path(port['declared_data_path']).name}
        candidates = []
        for directory in (shared, shared/'output', shared/'data'/f'frame{slot}', shared/f'Frame_{slot}'):
            for name in names:
                p = directory/(f'Frame_{slot}_'+name)
                if p.is_file() and not p.is_symlink():
                    candidates.append(p)
        # InferenceExec may use only output.data_path (slot 0's directory),
        # while preserving Frame_i_ prefixes for every frame. Never accept a
        # bare peer-frame filename as evidence for this slot.
        for peer in peer_frames:
            directory = peer/('sub'+str(stage['stage']))/'output'
            for name in names:
                p = directory/(f'Frame_{slot}_'+name)
                if p.is_file() and not p.is_symlink():
                    candidates.append(p)
        # Explicit output paths are already frame-specific; accept bare names only here.
        for directory in (own/'output', own):
            for name in names | {f'Frame_{slot}_'+x for x in names}:
                p = directory/name
                if p.is_file() and not p.is_symlink():
                    candidates.append(p)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError('Missing frame-specific output: slot %d %s' % (slot, port['tensor']))
        hashes = []
        for p in candidates:
            if p.stat().st_size != port['nbytes']:
                raise ValueError('Output size differs')
            body = p.read_bytes()
            if not np.isfinite(np.frombuffer(body, dtype=port['dtype'])).all():
                raise ValueError('Non-finite output')
            hashes.append(hashlib.sha256(body).hexdigest())
        if len(set(hashes)) != 1:
            raise ValueError('Conflicting frame-specific outputs')
        target = exports/port['filename']
        if target.exists():
            if (not allow_existing or target.is_symlink()
                    or hashlib.sha256(target.read_bytes()).hexdigest() != hashes[0]):
                raise ValueError('Never overwrite exported output')
        else:
            os.link(candidates[0], target)
        result.append(dict(port, stage=stage['stage'], remote_file=str(target), sha256=hashes[0],
                           finite=True, validation_backend='board-numpy', fresh_candidate_count=len(candidates),
                           frame_slot=slot))
    return result


def execute(manifest_path, expected_sha, run=subprocess.run):
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Manifest identity differs')
    manifest = json.loads(raw)
    root = checked_root(manifest['root'])
    recovery = manifest.get('recover_completed_stage0')
    expected_path = Path(str(root)+('.resume-1.json' if recovery else '.json'))
    if manifest_path != expected_path or manifest_path.is_symlink():
        raise ValueError('Wrong manifest namespace')
    cache = Path(manifest['cache'])
    frames = manifest['frames']
    count=len(frames)
    if manifest['schema'] not in ('roadmodel.two-frame-subnets/v1','roadmodel.multiframe-subnets/v1') or count not in (1,2,4,8):
        raise ValueError('Wrong bounded execution scope')
    roots = [checked_root(f['root']) for f in frames]
    if roots != [Path(str(root)+'-f'+str(i)) for i in range(count)]:
        raise ValueError('Wrong frame namespaces')
    if not recovery and any(p.exists() for p in [root, *roots]):
        raise ValueError('Never repeat existing board attempt')
    packets = [read_packet(Path(f['packet']), f['sha256'], r, cache) for f, r in zip(frames, roots)]
    if any(p[2] != packets[0][2] for p in packets):
        raise ValueError('Frame contracts differ')
    if recovery:
        if count!=2:raise ValueError('Stage-0 recovery is bound to original two frames')
        failure = json.loads((root/'failure.json').read_bytes())
        if failure['completed_stages'] or failure['error'] != 'Missing frame-specific output: slot 1 @Reshape_1_output_0':
            raise ValueError('Only the proven stage-0 collection failure can be recovered')
        for name, expected in recovery['stage0_files'].items():
            if name not in ('stdout.log', 'stderr.log', 'model_info.json'):
                raise ValueError('Wrong recovery evidence')
            if hashlib.sha256((root/'sub0'/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Retained stage-0 evidence changed')
        stdout = (root/'sub0/stdout.log').read_text(errors='replace')
        if not all(('Already successfully infer %d times' % i) in stdout for i in (1,2)):
            raise ValueError('Retained log does not prove two completed frames')
        for i in range(1,5):
            if any((root/f'sub{i}'/n).exists() for n in ('model_info.json','stdout.log','stderr.log')):
                raise ValueError('Unknown later-stage attempt; never retry')
        for frame_root, (_, files, _) in zip(roots, packets):
            for name, body in files.items():
                if (frame_root/name).read_bytes() != body:
                    raise ValueError('Retained frame inputs or metadata changed')
    else:
        initialize_frame(root, cache)
        for frame_root, (_, files, _) in zip(roots, packets):
            initialize_frame(frame_root, cache)
            for name, body in files.items():
                target = frame_root/name
                if target.is_symlink() and name.endswith('/port-contract.json'):
                    target.unlink()
                with target.open('xb') as stream:
                    stream.write(body)
    outputs = [{} for _ in frames]
    stages = [[] for _ in frames]
    measured = []
    started = time.monotonic()
    try:
        for contract in packets[0][2]:
            i = contract['stage']
            directory = root/('sub'+str(i))
            templates = [json.loads((p/('sub'+str(i))/'model_info.json').read_bytes()) for p in roots]
            info = stage_info(templates, contract, [p/('sub'+str(i)) for p in roots])
            if recovery and i == 0:
                if json.loads((directory/'model_info.json').read_bytes()) != info:
                    raise ValueError('Retained model_info does not match current frozen frames')
                elapsed = recovery['stage0_invocation_seconds']
            else:
                with (directory/'model_info.json').open('x') as f:
                    json.dump(info, f)
                tick = time.monotonic()
                value = run(['InferenceExec', 'model_info.json'], cwd=str(directory), capture_output=True,
                            timeout=180, env=dict(os.environ, PWD=str(directory)))
                elapsed = time.monotonic()-tick
                (directory/'stdout.log').write_bytes(value.stdout)
                (directory/'stderr.log').write_bytes(value.stderr)
                if value.returncode:
                    raise ValueError('Original subnet failed: %d' % i)
            checked_at = time.monotonic()
            for slot, frame_root in enumerate(roots):
                checked = collect_stage(directory, frame_root, contract, slot, roots,
                                        allow_existing=bool(recovery and i==0))
                outputs[slot].update({p['tensor']:p for p in checked})
                stages[slot].append(dict(stage=i, state='SUCCEEDED', exit_code=0, outputs=checked,
                    input_count=len(contract['inputs']), output_count=len(checked),
                    InferenceExec_seconds=elapsed, timing_scope='shared_multiframe_invocation'))
            measured.append(dict(stage=i, InferenceExec_seconds=elapsed,
                                 validation_seconds=time.monotonic()-checked_at, infer_num=count,
                                 adopted_without_reinference=bool(recovery and i==0)))
        bundles = []
        for slot, (frame_root, (request, _, _)) in enumerate(zip(roots, packets)):
            if len(outputs[slot]) != 49:
                raise ValueError('Incomplete per-frame outputs')
            result = dict(state='SUCCEEDED', request_sha256=frames[slot]['sha256'], root=str(frame_root),
                          stages=stages[slot], outputs=outputs[slot], frame_slot=slot)
            bundle = frame_root/'result.tar'
            with tarfile.open(str(bundle), mode='w:') as archive:
                add_bytes(archive, 'result.json', json.dumps(result).encode())
                for i in range(5):
                    for suffix in ('stdout', 'stderr'):
                        add_bytes(archive, f'sub{i}.{suffix}.log', (root/f'sub{i}/{suffix}.log').read_bytes())
                for public in request['public_outputs']:
                    meta = outputs[slot][public['tensor']]
                    body = Path(meta['remote_file']).read_bytes()
                    if hashlib.sha256(body).hexdigest() != meta['sha256']:
                        raise ValueError('Output changed during collection')
                    add_bytes(archive, 'sub%d/%s' % (meta['stage'], meta['filename']), body)
            blob = bundle.read_bytes()
            bundles.append(dict(slot=slot, path=str(bundle), nbytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        result = dict(state='SUCCEEDED', manifest_sha256=expected_sha, frames=count, invocations=5,
                      stages=measured, board_seconds=time.monotonic()-started, bundles=bundles)
        (root/'result.json').write_text(json.dumps(result))
        return result
    except Exception as error:
        (root/'failure.json').write_text(json.dumps(dict(state='NEEDS_REVIEW', completed_stages=measured,
            error_type=type(error).__name__, error=str(error))))
        raise


def execute_bundle(path, expected_sha):
    """One bounded upload/download; preserve the original per-frame packet checks."""
    packet=Path(path)
    if packet.is_symlink() or packet.stat().st_size>128*1024**2:
        raise ValueError('Invalid batch packet')
    raw=packet.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Batch packet identity differs')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() for m in members):raise ValueError('Unsafe batch packet')
        body=archive.extractfile('manifest.json').read();manifest=json.loads(body)
        root=checked_root(manifest['root']);frames=manifest['frames']
        if packet!=Path(str(root)+'.bundle.tar') or len(frames) not in (1,2,4,8):raise ValueError('Wrong batch namespace')
        if set(names)!={'manifest.json'}|{f'frame_{i}.tar' for i in range(len(frames))}:raise ValueError('Batch inventory differs')
        if (root.exists() or Path(str(root)+'.json').exists()
                or any(Path(f['packet']).exists() or Path(f['root']).exists() for f in frames)):
            raise ValueError('Existing batch attempt must not replay')
        for i,f in enumerate(frames):
            if f['root']!=str(root)+'-f'+str(i) or f['packet']!=f['root']+'.request.tar':raise ValueError('Wrong frame namespace')
            blob=archive.extractfile(f'frame_{i}.tar').read()
            if hashlib.sha256(blob).hexdigest()!=f['sha256']:raise ValueError('Frame packet identity differs')
            with Path(f['packet']).open('xb') as stream:stream.write(blob)
    manifest_path=Path(str(root)+'.json')
    with manifest_path.open('xb') as stream:stream.write(body)
    result=execute(manifest_path,hashlib.sha256(body).hexdigest())
    combined=root/'batch-result.tar'
    with tarfile.open(str(combined),mode='w:') as archive:
        add_bytes(archive,'result.json',json.dumps(result).encode())
        for row in result['bundles']:add_bytes(archive,f"frame_{row['slot']}.tar",Path(row['path']).read_bytes())
    blob=combined.read_bytes()
    return dict(state='SUCCEEDED',bundle_path=str(combined),nbytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                batch_packet_sha256=expected_sha,frames=len(frames),invocations=5)


def cleanup(path, expected_sha):
    """Remove only this verified, isolated batch's disposable board artifacts."""
    import shutil
    path=Path(path);body=path.read_bytes()
    if path.is_symlink() or hashlib.sha256(body).hexdigest()!=expected_sha:raise ValueError('Wrong cleanup manifest')
    manifest=json.loads(body);root=checked_root(manifest['root'])
    if path!=Path(str(root)+'.json'):raise ValueError('Wrong cleanup path')
    result=json.loads((root/'result.json').read_bytes())
    if result['state']!='SUCCEEDED' or result['manifest_sha256']!=expected_sha:raise ValueError('Batch not verified')
    dirs=[root]+[checked_root(f['root']) for f in manifest['frames']]
    if dirs[1:]!=[Path(str(root)+'-f'+str(i)) for i in range(len(manifest['frames']))]:raise ValueError('Cleanup frame mismatch')
    if len(manifest['frames']) not in (1,2,4,8):raise ValueError('Invalid cleanup scope')
    if any(f['packet']!=f['root']+'.request.tar' for f in manifest['frames']):raise ValueError('Wrong cleanup packet')
    for p in dirs:
        if p.is_symlink():raise ValueError('Unsafe cleanup link')
    for p in dirs:shutil.rmtree(p)
    for p in [path,Path(str(root)+'.bundle.tar')]+[Path(f['packet']) for f in manifest['frames']]:
        if p.exists():p.unlink()
    return {'state':'CLEANED','removed_frame_directories':len(manifest['frames'])}


if __name__ == '__main__':
    if len(sys.argv)==4 and sys.argv[1] in ('bundle','cleanup'):
        print(json.dumps((execute_bundle if sys.argv[1]=='bundle' else cleanup)(*sys.argv[2:])))
    elif len(sys.argv)==3:print(json.dumps(execute(*sys.argv[1:])))
    else:raise ValueError('Expected manifest/bundle path and SHA256')
