"""Real validator/packet tests with a fake executable, never touching a board."""
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

if not Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020').is_dir():
    raise unittest.SkipTest('Run batch native-package tests on the isolated Linux workstation')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import board_batch_subnet as helper
import board_validate_subnet as validator
from batch_subnet_driver import unpack_result
import batch_subnet_driver as batch_driver


class BatchTests(unittest.TestCase):
    def fixture(self, base, *, extra=None):
        root = base / 'pcm-v322-subnets-full-test'; cache = base / 'pcm-v322-fast-cache-fixture'; cache.mkdir()
        for stage in range(5):(cache / ('sub%d' % stage)).mkdir()
        files = {}; request = dict(schema='roadmodel.board-batch/v1', root=str(root), cache=str(cache), inputs=[], stages=[], public_outputs=[])
        for i in range(11):
            body = np.array([i], np.float16).tobytes(); name = 'input%d.bin' % i; files['inputs/' + name] = body
            request['inputs'].append(dict(filename=name, nbytes=2, sha256=helper.sha(body)))
        for stage, count in enumerate((10, 10, 10, 10, 9)):
            outputs = [dict(tensor='s%d_o%d' % (stage, i), filename='s%d_o%d.1x2.float16.bin' % (stage, i),
                            declared_data_path='output/RESULT_s%d_o%d_float16.bin' % (stage, i), dtype='float16', nbytes=4) for i in range(count)]
            contract = dict(stage=stage, inputs=[], outputs=outputs)
            info = json.dumps({'fixture': True}).encode(); raw = json.dumps(contract).encode()
            files['sub%d/model_info.json' % stage] = info; files['sub%d/port-contract.json' % stage] = raw
            request['stages'].append(dict(stage=stage, model_info_sha256=helper.sha(info), contract_sha256=helper.sha(raw)))
            for p in outputs:
                if len(request['public_outputs']) < 18:
                    request['public_outputs'].append(dict(stage=stage, **{k:p[k] for k in ('tensor','filename','nbytes','dtype')}))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:') as archive:
            helper.add_bytes(archive, 'request.json', json.dumps(request).encode())
            for name, body in files.items():helper.add_bytes(archive, name, body)
            if extra:helper.add_bytes(archive, extra, b'bad')
        packet = Path(str(root) + '.request.tar'); packet.write_bytes(stream.getvalue())
        return root, cache, packet, helper.sha(stream.getvalue()), request

    def execute(self, fixture, *, fail_stage=None, omit=None):
        root, cache, packet, signature, request = fixture; calls = []
        def native(args, cwd, capture_output, timeout, env):
            stage = int(Path(cwd).name[3:]); calls.append(stage)
            self.assertEqual(['InferenceExec', 'model_info.json'], args)
            self.assertEqual(cwd,env['PWD'])
            if stage == fail_stage:return SimpleNamespace(returncode=2, stdout=b'failed', stderr=b'fixture error')
            contract = json.loads((Path(cwd) / 'port-contract.json').read_text())
            for i, p in enumerate(contract['outputs']):
                if (stage, i) != omit:
                    (Path(cwd) / p['declared_data_path']).write_bytes(np.array([stage, i], np.float16).tobytes())
            return SimpleNamespace(returncode=0, stdout=b'exact original native fixture', stderr=b'')
        with patch.object(helper, 'checked_root', return_value=root), patch.object(validator, 'checked_root', return_value=root):
            reply = helper.execute(root, cache, packet, signature, run=native)
        return reply, calls

    def test_all_five_executed_once_all49_validated_and18_round_trip_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); root, _, _, signature, request = fixture
            reply, calls = self.execute(fixture); self.assertEqual(list(range(5)), calls)
            self.assertEqual(49, reply['checked_outputs']); self.assertEqual(18, reply['public_outputs'])
            body = Path(reply['bundle_path']).read_bytes(); self.assertEqual(helper.sha(body), reply['sha256'])
            dest = Path(temp) / 'collected'; dest.mkdir(); result = unpack_result(body, dest, request['public_outputs'], signature)
            self.assertEqual(49, len(result['outputs']))
            for p in request['public_outputs']:
                meta = result['outputs'][p['tensor']]
                self.assertEqual(Path(meta['remote_file']).read_bytes(), (dest / ('sub%d/%s' % (p['stage'], p['filename']))).read_bytes())

    def test_existing_frame_is_never_executed_again(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); self.execute(fixture)
            with self.assertRaisesRegex(ValueError, 'Never replay'):self.execute(fixture)

    def test_private_metadata_does_not_write_through_to_model_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=self.fixture(Path(temp)); cache=fixture[1]/'sub0/port-contract.json'
            cache.write_bytes(b'immutable cached contract')
            self.execute(fixture)
            self.assertEqual(b'immutable cached contract',cache.read_bytes())

    def test_input_packet_hash_rejected_before_initializing_or_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache, packet, signature, request = self.fixture(Path(temp)); packet.write_bytes(b'corrupted')
            with patch.object(helper, 'checked_root', return_value=root), self.assertRaisesRegex(ValueError, 'digest'):
                helper.execute(root, cache, packet, signature, run=lambda *a, **k:self.fail('Native must not run'))
            self.assertFalse(root.exists())

    def test_unexpected_or_traversal_member_rejected_before_inference(self):
        for name in ('../escape', 'unexpected.bin'):
            with tempfile.TemporaryDirectory() as temp:
                fixture = self.fixture(Path(temp), extra=name)
                with self.assertRaisesRegex(ValueError, 'packet member|inventory differs'):self.execute(fixture)
                self.assertFalse(fixture[0].exists())

    def test_native_failure_stops_remaining_subnets_and_retains_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'subnet failed'):self.execute(fixture, fail_stage=2)
            self.assertTrue((fixture[0] / 'batch-failure.json').exists())
            self.assertFalse((fixture[0] / 'sub3.stdout.log').exists())

    def test_missing_intermediate_is_rejected_before_next_subnet(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'Missing fresh output'):self.execute(fixture, omit=(0, 0))
            self.assertFalse((fixture[0] / 'sub1.stdout.log').exists())

    def test_changed_result_digest_or_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); reply, _ = self.execute(fixture)
            root, _, _, signature, request = fixture; body = Path(reply['bundle_path']).read_bytes()
            dest = Path(temp) / 'rejected'; dest.mkdir()
            with self.assertRaisesRegex(ValueError, 'different execution'):unpack_result(body, dest, request['public_outputs'], '0' * 64)
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:') as archive:helper.add_bytes(archive, '../escape', b'no')
            with self.assertRaisesRegex(ValueError, 'inventory'):unpack_result(stream.getvalue(), dest, request['public_outputs'], signature)


class AdmissionTests(unittest.TestCase):
    def finish_fixture(self, *, cleanup_fails=False, release_fails=False, disconnect=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
        full=base/'full';(full/'code').mkdir(parents=True);(full/'code/subnet_contract.py').write_text('fixture')
        frame=base/'frame';frame.mkdir();(frame/'executed-adapter.py').write_text('fixture')
        (frame/'input-reuse.json').write_text(json.dumps({'frame':{'frame_id':'fixture_1'}}))
        runner=SimpleNamespace(claim_bench=Mock(return_value={'state':'CLAIMED'}),atomic_write_json=Mock(),
                               release_bench=Mock(return_value={'state':'RELEASED'}))
        if release_fails:runner.release_bench.side_effect=OSError('release unavailable')
        engine=batch_driver.Engine.__new__(batch_driver.Engine);engine.cache='/userdata/benchEval/models/pcm-v322-fast-cache-fixture'
        engine.helper_sha='fixture';engine.graph={'stages':[]};outputs={}
        for stage,count in enumerate((15,15,9,8,2)):
            ports=[]
            for i in range(count):
                port={'tensor':f's{stage}_{i}','filename':f's{stage}_{i}.bin','nbytes':2,'dtype':'float16'}
                ports.append(port);outputs[port['tensor']]={**port,'stage':stage,
                    'remote_file':f'/userdata/benchEval/models/pcm-v322-subnets-full-cleanup/sub{stage}/export/'+port['filename']}
            engine.graph['stages'].append({'stage':stage,'outputs':ports})
        reply={'state':'SUCCEEDED','request_sha256':hashlib.sha256(b'packet').hexdigest(),
               'bundle_path':'/userdata/benchEval/models/pcm-v322-subnets-full-cleanup/result.tar',
               'nbytes':1,'sha256':'fixture','checked_outputs':49,'public_outputs':18}
        def command(text,**kwargs):
            if disconnect:raise OSError('connection lost; process state unknown')
            if text.startswith('pgrep'):return 0,'',''
            if text.startswith('rm ') and cleanup_fails:raise OSError('private cleanup failed')
            return 0,json.dumps(reply),''
        client=SimpleNamespace(_exec=Mock(side_effect=command),_sftp_exists=lambda p:False,
                               _sftp=SimpleNamespace(putfo=Mock()))
        engine.connect=Mock(return_value=client);engine.ensure_cache=Mock()
        engine.ensure_batch_helper=Mock(return_value=('fixture-helper','fixture-sha'))
        engine.input_packet=Mock(return_value=(b'packet',[{'nbytes':2}]*18));engine.compare=Mock()
        patches=[patch.dict(sys.modules,{'runner_state':runner}),patch.object(batch_driver.base,'FULL',full),
                 patch.object(batch_driver.base.old,'BASE',base),patch.object(batch_driver.base.old,'RUNNER',base/'runner'),
                 patch.object(batch_driver.base.old.subprocess,'run',return_value=SimpleNamespace(stdout='{}')),
                 patch.object(batch_driver.base,'stream_read',return_value=b'x'),
                 patch.object(batch_driver,'unpack_result',return_value={'outputs':outputs,'stages':[],
                     'input_verification_seconds':0,'native_total_seconds':0}),
                 patch.object(batch_driver.base.old,'write_compatible_results')]
        for p in patches:p.start();self.addCleanup(p.stop)
        return engine,frame,runner

    def test_private_cleanup_failure_still_releases_lease_and_saves_success(self):
        engine,frame,runner=self.finish_fixture(cleanup_fails=True)
        engine.run_board(SimpleNamespace(run_id='pcm-v322-subnets-full-cleanup'),frame)
        runner.release_bench.assert_called_once()
        record=json.loads((frame/'execution.json').read_text())
        self.assertEqual(record['state'],'SUCCEEDED');self.assertIn('cleanup_issue',record)

    def test_release_failure_keeps_unknown_lock_and_writes_review_receipt(self):
        engine,frame,runner=self.finish_fixture(release_fails=True)
        engine.run_board(SimpleNamespace(run_id='pcm-v322-subnets-full-cleanup'),frame)
        record=json.loads((frame/'execution.json').read_text())
        self.assertEqual(record['state'],'NEEDS_REVIEW');self.assertTrue(record['lock_retained'])
        batch_driver.base.old.write_compatible_results.assert_not_called()

    def test_disconnect_never_releases_lock_without_process_state_proof(self):
        engine,frame,runner=self.finish_fixture(disconnect=True)
        with self.assertRaises(OSError):engine.run_board(SimpleNamespace(run_id='pcm-v322-subnets-full-cleanup'),frame)
        runner.release_bench.assert_not_called()
        self.assertTrue(json.loads((frame/'execution.json').read_text())['lock_retained'])

    def test_user_pause_while_queued_prevents_claim_and_native_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); full=root/'full';full.mkdir(); frame=root/'frame';frame.mkdir()
            runner=SimpleNamespace(claim_bench=Mock(),atomic_write_json=Mock())
            engine=batch_driver.Engine.__new__(batch_driver.Engine)
            with patch.dict(sys.modules,{'runner_state':runner}),patch.dict('os.environ',{'ROADMODEL_OPT_VALIDATION':''}),\
                 patch.object(batch_driver.base,'FULL',full),patch.object(batch_driver.base.old,'RUNNER',root/'runner'),\
                 patch.object(batch_driver.base.old.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps({'active':'another-user','queued':[],'waiting':[]}))),\
                 patch.object(batch_driver.time,'sleep',side_effect=lambda seconds:(full/'STOP_AFTER_FRAME').write_text('user pause')):
                with self.assertRaisesRegex(ValueError,'User paused'):engine.run_board(SimpleNamespace(run_id='pcm-v322-subnets-full-queue-test'),frame)
            runner.claim_bench.assert_not_called()

    def test_invalid_namespace_is_rejected_before_runner_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            engine=batch_driver.Engine.__new__(batch_driver.Engine)
            with self.assertRaisesRegex(ValueError,'namespace'):
                engine.run_board(SimpleNamespace(run_id='../../shared'),Path(temp))


if __name__ == '__main__':unittest.main()
