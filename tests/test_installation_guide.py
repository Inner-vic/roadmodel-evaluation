import argparse
from pathlib import Path
import tempfile
import unittest

from pcm_eval.common import load_json
from pcm_eval.installation_guide import add_arguments, configure, make_request, template_from_space
from tests.test_installation import request, space


class GuideTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/'request.json'
        self.raw = space('reference-space', owner='someone-else@example.com')
        self.raw['spec']['resources']['gpuType'] = 'nvidia.com/NVIDIA-L4'
        self.raw['spec']['resources']['affinity'] = {
            'kubernetes.io/hostname': 'old-node', 'lpai.org/share-gpu-type': 'NVIDIA-L4'}

    def args(self, interactive=False):
        argv = ['--from-space', 'reference-space', '--namespace', 'team', '--cluster', 'approved-cluster',
                '--lpai-login', 'newengineer', '--lpai-account', 'newengineer@example.com',
                '--ssh-credential', 'newengineer-key', '--identity-file', '/new/existing/key',
                '--output', str(self.output)]
        if not interactive:
            argv.append('--non-interactive')
        return add_arguments(argparse.ArgumentParser()).parse_args(argv)

    def run_guide(self, args=None, answer=None):
        calls = []
        result = configure(args or self.args(), ask=answer or (lambda _: self.fail('Unexpected prompt')),
                           emit=lambda _: None, query=lambda argv: calls.append(argv) or {'data': self.raw},
                           verify_identity=lambda: {'open_id': 'ou_newengineer', 'app_id': 'cli_newapp'})
        return result, calls

    def test_live_template_is_portable_and_does_not_bind_other_owners_space(self):
        result, calls = self.run_guide()
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ['devspaces', 'get'])
        value = load_json(self.output)
        self.assertEqual(value['template']['resources']['gpuType'], 'nvidia.com/NVIDIA-L4')
        self.assertNotIn('kubernetes.io/hostname', value['template']['resources']['affinity'])
        self.assertEqual(self.raw['spec']['resources']['affinity']['kubernetes.io/hostname'], 'old-node')
        self.assertEqual(value['account']['lpai_account'], 'newengineer@example.com')
        self.assertEqual(value['ssh']['credential_name'], 'newengineer-key')
        self.assertNotIn('adopt_existing', value)
        self.assertNotIn('uid', value)

    def test_reference_identity_mismatch_does_not_save(self):
        self.raw['spec']['clusterName'] = 'another-cluster'
        with self.assertRaises(ValueError):
            self.run_guide()
        self.assertFalse(self.output.exists())

    def test_cpu_mode_is_not_silently_supported(self):
        self.raw['spec']['resources']['gpuType'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'CPU'):
            self.run_guide()
        self.assertFalse(self.output.exists())

    def test_missing_gpu_or_unknown_ssh_entry_is_rejected(self):
        self.raw['spec']['accesses'] = []
        with self.assertRaisesRegex(ValueError, 'SSH'):
            template_from_space(self.raw)
        original = request()
        original['template']['resources'].pop('gpuType')
        with self.assertRaises(ValueError):
            make_request(original['template'], original['account'], original['ssh'])

    def test_existing_deployment_file_is_never_overwritten(self):
        self.output.write_text('keep existing')
        with self.assertRaisesRegex(ValueError, '已存在'):
            self.run_guide()
        self.assertEqual(self.output.read_text(), 'keep existing')

    def test_interactive_cancel_and_missing_field_do_not_save(self):
        with self.assertRaisesRegex(ValueError, '已取消'):
            self.run_guide(self.args(interactive=True), answer=lambda _: 'no')
        self.assertFalse(self.output.exists())
        args = self.args();args.ssh_credential = None
        with self.assertRaisesRegex(ValueError, '缺少配置'):
            self.run_guide(args)
        self.assertFalse(self.output.exists())

    def test_template_file_import_ignores_old_account_credentials_and_uid(self):
        original = request()
        original['adopt_existing'] = {'name': 'old-space', 'uid': '12345678-abcd'}
        path = Path(self.temp.name)/'template.json'
        from pcm_eval.common import save_json
        save_json(path, original)
        args = self.args();args.from_space = None;args.template_file = str(path)
        _, calls = self.run_guide(args)
        self.assertEqual(calls, [])
        value = load_json(self.output)
        self.assertEqual(value['account']['feishu_user_id'], 'ou_newengineer')
        self.assertNotIn('adopt_existing', value)
        self.assertEqual(value['template']['idle_shutdown']['minutes'], 240)


if __name__ == '__main__':
    unittest.main()
