"""Batch transport and remote orchestration, preserving the native evaluator."""
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import fast_subnet_driver as base


def add_bytes(archive, name, body):
    info = tarfile.TarInfo(name); info.size = len(body); info.mode = 0o600
    archive.addfile(info, io.BytesIO(body))


def unpack_result(body, root, public, expected_request_sha):
    with tarfile.open(fileobj=io.BytesIO(body), mode='r:') as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        expected = {'result.json'} | {'sub%d.%s.log' % (i, suffix) for i in range(5) for suffix in ('stdout', 'stderr')}
        expected |= {'sub%d/%s' % (p['stage'], p['filename']) for p in public}
        if len(names) != len(set(names)) or set(names) != expected or any(not m.isfile() for m in members):
            raise ValueError('Unexpected batch result inventory')
        result = json.loads(archive.extractfile('result.json').read())
        if result['state'] != 'SUCCEEDED' or result['request_sha256'] != expected_request_sha:
            raise ValueError('Batch belongs to a different execution')
        outputs = result['outputs']
        if len(outputs) != 49 or [s['stage'] for s in result['stages']] != list(range(5)):
            raise ValueError('Missing original subnet checks')
        if not all(m['finite'] and m['validation_backend'] == 'board-numpy' for m in outputs.values()):
            raise ValueError('Missing original finite checks')
        for stage in result['stages']:
            if stage['state'] != 'SUCCEEDED' or stage['exit_code'] != 0:raise ValueError('Original native execution failed')
        for p in public:
            meta = outputs[p['tensor']]
            if any(meta[k] != p[k] for k in ('stage', 'filename', 'nbytes', 'dtype')):raise ValueError('Public tensor contract changed')
            name = 'sub%d/%s' % (p['stage'], p['filename']); blob = archive.extractfile(name).read()
            if len(blob) != p['nbytes'] or hashlib.sha256(blob).hexdigest() != meta['sha256']:
                raise ValueError('Batch tensor bytes differ')
            if not np.isfinite(np.frombuffer(blob, dtype=p['dtype'])).all():raise ValueError('Non-finite public tensor')
            dest = root / name; dest.parent.mkdir(exist_ok=True); dest.write_bytes(blob)
        for stage in range(5):
            for suffix in ('stdout', 'stderr'):
                name = 'sub%d.%s.log' % (stage, suffix)
                (root / name).write_bytes(archive.extractfile(name).read())
    return result


class Engine(base.Engine):
    def ensure_batch_helper(self, client):
        helper = Path(__file__).with_name('board_batch_subnet.py'); signature = base.sha(helper)
        remote = self.cache + '/board_batch_' + signature[:20] + '.py'
        if getattr(self, '_batch_helper_ready', None) == remote:return remote, signature
        if not client._sftp_exists(remote):client._sftp_write_bytes(remote, helper.read_bytes())
        if client._exec('sha256sum ' + shlex.quote(remote), check=True)[1].split()[0] != signature:
            raise ValueError('Private batch helper differs')
        self._batch_helper_ready = remote
        return remote, signature

    def input_packet(self, root, board):
        request = {'schema': 'roadmodel.board-batch/v1', 'root': board, 'cache': self.cache,
                   'inputs': [], 'stages': [], 'public_outputs': []}
        files = {}; reuse = json.loads((root / 'input-reuse.json').read_text())
        for port in self.graph['external_inputs']:
            path = root / 'inputs' / port['external_filename']; body = path.read_bytes()
            expected = next(p['sha256'] for p in reuse['inputs'] if p['filename'] == path.name)
            if len(body) != port['nbytes'] or hashlib.sha256(body).hexdigest() != expected:raise ValueError('Prepared input differs')
            request['inputs'].append(dict(filename=path.name, nbytes=len(body), sha256=expected))
            files['inputs/' + path.name] = body
        for stage in self.graph['stages']:
            i = stage['stage']; info = base.stage_info(self.infos[i], stage, board)
            info_raw = json.dumps(info).encode(); contract_raw = json.dumps(stage).encode()
            files['sub%d/model_info.json' % i] = info_raw; files['sub%d/port-contract.json' % i] = contract_raw
            request['stages'].append(dict(stage=i, model_info_sha256=hashlib.sha256(info_raw).hexdigest(),
                                         contract_sha256=hashlib.sha256(contract_raw).hexdigest()))
            request['public_outputs'].extend(dict(stage=i, **{k: p[k] for k in ('tensor', 'filename', 'nbytes', 'dtype')})
                                            for p in stage['outputs'] if p['tensor'] in self.public_names)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:') as archive:
            add_bytes(archive, 'request.json', json.dumps(request).encode())
            for name, body in files.items():add_bytes(archive, name, body)
        return stream.getvalue(), request['public_outputs']

    def run_board(self, args, root, graph=None, infos=None, *, reference_check=False, admission_after_receipt=False):
        root = Path(root)
        if (root / 'execution_receipt.json').exists() or (root / 'execution.json').exists():
            raise ValueError('Existing or unknown hardware execution must never be replayed')
        if admission_after_receipt:raise ValueError('Batch validation only dispatches after explicit original Runner admission')
        board = '/userdata/benchEval/models/' + args.run_id
        if PurePosixPath(board).parent != PurePosixPath('/userdata/benchEval/models') or not args.run_id.startswith('pcm-v322-subnets-full-'):
            raise ValueError('Wrong private board namespace')
        sys.path.insert(0, str(base.old.RUNNER / 'current/automation')); import runner_state
        while True:
            validation_mode=bool(os.environ.get('ROADMODEL_OPT_VALIDATION'))
            if not validation_mode and (base.FULL/'STOP_AFTER_FRAME').exists():
                raise ValueError('User paused before bench admission')
            queue = json.loads(base.old.subprocess.run([str(base.old.RUNNER / 'bin/bench-eval'), 'queue'],
                               capture_output=True, text=True, check=True, timeout=20).stdout)
            if not (queue.get('active') or queue.get('queued') or queue.get('waiting') or (base.old.RUNNER / 'locks/bench.lock').exists()):break
            if validation_mode:raise ValueError('Shared bench occupied; no validation dispatch')
            time.sleep(1)
        for name,source in (('executed-native-driver.py',Path(__file__)),('executed-board-batch-helper.py',Path(__file__).with_name('board_batch_subnet.py'))):
            target=root/name
            if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Prepared native source changed')
            if not target.exists():target.write_bytes(source.read_bytes())
        leases = base.old.BASE / 'external-bench-leases'; lease = leases / args.run_id; lease.mkdir(parents=True, exist_ok=False)
        runner_state.atomic_write_json(lease / 'manifest.json', {'submitted_by': 'developer', 'purpose': 'authorized RoadModel batch optimization validation'})
        runner_state.atomic_write_json(lease / 'state.json', {'run_id': args.run_id, 'state': 'READY'})
        runner_state.atomic_write_json(lease / 'plan.json', {'preflight': {'status': 'READY'}, 'steps': [], 'activation': False,
                                                        'replay': False, 'frame_count': 1, 'request_source': 'explicit Codex user request'})
        claimed = runner_state.claim_bench(leases, base.old.RUNNER / 'locks', args.run_id)
        board = '/userdata/benchEval/models/' + args.run_id
        if PurePosixPath(board).parent != PurePosixPath('/userdata/benchEval/models') or not args.run_id.startswith('pcm-v322-subnets-full-'):
            raise ValueError('Wrong private board namespace')
        record = {'schema': 'roadmodel.five-subnet-execution/v1', 'state': 'RUNNING', 'run_id': args.run_id, 'version': 'v3.2.2',
                  'requested_frames': 1, 'frame_id': json.loads((root / 'input-reuse.json').read_text())['frame']['frame_id'],
                  'stages': [], 'guard': claimed, 'board_inputs_from_existing_cache': True, 'original_active_app_modified': False,
                  'code_sha256': {'driver': base.sha(Path(__file__)), 'contract': base.sha(base.FULL / 'code/subnet_contract.py'),
                                  'board_validator': self.helper_sha,'preparation_driver':base.sha(root/'executed-adapter.py')},
                  'code_files':{'driver':'executed-native-driver.py','preparation_driver':'executed-adapter.py','board_batch_helper':'executed-board-batch-helper.py'},
                  'optimization': {'all49_checks_retained': True, 'public_downloads': 18, 'shared_runner_unchanged': True,
                                   'batch_input_uploads': 1, 'batch_output_downloads': 1}}
        base.save(root / 'execution.json', record); client = None; safe = False; started = time.monotonic(); packet_path = board + '.request.tar'
        try:
            client = self.connect()
            if client._exec('pgrep -x InferenceExec', check=False)[1].strip():raise ValueError('Residual native process; do not dispatch')
            tick = time.monotonic(); self.ensure_cache(client); helper, helper_sha = self.ensure_batch_helper(client)
            record['cache_setup_seconds'] = time.monotonic() - tick; record['code_sha256']['board_batch_helper'] = helper_sha
            packet, public = self.input_packet(root, board); packet_sha = hashlib.sha256(packet).hexdigest()
            if client._sftp_exists(packet_path) or client._sftp_exists(board):raise ValueError('Unknown private board attempt exists; never overwrite')
            tick = time.monotonic(); client._sftp.putfo(io.BytesIO(packet), packet_path, file_size=len(packet), confirm=True)
            record['batch_input_upload_seconds'] = time.monotonic() - tick; record['batch_input_bytes'] = len(packet)
            tick = time.monotonic()
            _, text, _ = client._exec(shlex.join(['python3', helper, board, self.cache, packet_path, packet_sha]), timeout=360, check=True)
            record['batch_remote_command_seconds'] = time.monotonic() - tick; reply = json.loads(text)
            if reply['state'] != 'SUCCEEDED' or reply['request_sha256'] != packet_sha or reply['bundle_path'] != board + '/result.tar':
                raise ValueError('Unknown batch execution reply')
            if reply['checked_outputs'] != 49 or reply['public_outputs'] != 18 or reply['nbytes'] > 8 * 1024**2:
                raise ValueError('Invalid batch reply inventory')
            tick = time.monotonic(); body = base.stream_read(client, reply['bundle_path'], reply['nbytes'], reply['sha256'])
            result = unpack_result(body, root, public, packet_sha)
            record['batch_output_download_and_unpack_seconds'] = time.monotonic() - tick
            record['stages'] = result['stages']; record['board_batch_timing'] = {k: result[k] for k in ('input_verification_seconds', 'native_total_seconds')}
            outputs = result['outputs']
            expected_all={p['tensor']:(stage['stage'],p) for stage in self.graph['stages'] for p in stage['outputs']}
            if set(outputs)!=set(expected_all):raise ValueError('Full output inventory changed')
            for name,meta in outputs.items():
                i,port=expected_all[name]
                if meta['stage']!=i or any(meta[k]!=port[k] for k in ('filename','nbytes','dtype')) or meta['remote_file']!=board+'/sub%d/export/'%i+port['filename']:
                    raise ValueError('Full output contract changed')
            base.save(root / 'all-subnet-outputs.json', outputs)
            self.compare(root, record, outputs)
            record['optimized_board_seconds_before_diagnostic'] = time.monotonic() - started
            record['public_tensor_bytes_downloaded'] = sum(p['nbytes'] for p in public)
            record['batch_transport_download_bytes'] = reply['nbytes']
            if reference_check:self.reference_public_check(client, root, record, outputs, public)
            safe = True; record.update(state='SUCCEEDED', public_outputs=18, tensor_only=True)
        except Exception as error:
            record.update(state='FAILED', error=repr(error))
            try:safe = client is not None and not client._exec('pgrep -x InferenceExec', timeout=10, check=False)[1].strip()
            except Exception:safe = False
            raise
        finally:
            try:
                if safe and record.get('state') == 'SUCCEEDED':
                    try:
                        client._exec('rm -rf -- ' + shlex.quote(board) + ' && rm -f -- ' + shlex.quote(packet_path), check=True)
                        record['private_board_directory_removed_after_output_verification'] = True
                    except Exception as error:
                        # Verified outputs remain valid. Failed private-file cleanup
                        # must not skip lease release or lose the native receipt.
                        record['cleanup_issue'] = str(error)[:300]
                if safe:
                    try:
                        record['release'] = runner_state.release_bench(leases, base.old.RUNNER / 'locks', args.run_id, 'NO_ACTIVATION', None)
                    except Exception as error:
                        record.update(state='NEEDS_REVIEW', lock_retained=True, release_issue=str(error)[:300])
                else:record.update(state='NEEDS_REVIEW', lock_retained=True)
            finally:
                base.save(root / 'execution.json', record)
        if record['state'] == 'SUCCEEDED':base.old.write_compatible_results(root, record)

    def reference_public_check(self, client, root, record, outputs, public):
        started = time.monotonic(); ref = root / 'legacy-public-collector-validation'; ref.mkdir()
        for port in public:
            meta = outputs[port['tensor']]; blob = client._sftp_read_bytes(meta['remote_file'])
            if len(blob) != meta['nbytes'] or hashlib.sha256(blob).hexdigest() != meta['sha256']:
                raise ValueError('Legacy public collector differs')
            if blob != (root / ('sub%d/%s' % (meta['stage'], meta['filename']))).read_bytes():raise ValueError('Batch public bytes differ')
            dest = ref / ('sub%d/%s' % (meta['stage'], meta['filename'])); dest.parent.mkdir(exist_ok=True); dest.write_bytes(blob)
        for name in ('public-outputs.json', 'onnx_outputs'):(ref / name).symlink_to(root / name, target_is_directory=(root / name).is_dir())
        self.compare(ref, record, outputs)
        if json.loads((ref / 'tensor-comparison.json').read_text()) != json.loads((root / 'tensor-comparison.json').read_text()):
            raise ValueError('Legacy tensor metrics differ')
        record['reference_validation'] = {'all18_bytes_and_metrics_exact': True, 'original49_validator_unchanged': True,
                                          'diagnostic_seconds': time.monotonic() - started, 'native_inferences': 1}
