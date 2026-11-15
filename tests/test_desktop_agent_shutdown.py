import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import delivery_demo_agent as demo
from tools import evaluation_agent as evaluation
from tools import model_space_desktop as space
from tools import connect_active_entry as active


class DesktopAgentShutdownTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'config.json'

    def save_config(self, **extra):
        cfg = {'local_state': str(self.root/'state'),
               'connection_index': str(self.root/'connection'),
               'agent_id': 'agent', 'owner': 'owner', **extra}
        self.config.write_text(json.dumps(cfg), encoding='utf-8')
        return cfg

    def test_windows_rpc_hides_console_and_does_not_retry(self):
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b'null'))
        with patch.object(demo, 'os', SimpleNamespace(name='nt', environ={})), \
             patch.object(demo, 'subprocess', SimpleNamespace(run=run, CREATE_NO_WINDOW=0x08000000)):
            result = demo.rpc({'python': '/python', 'release': '/release', 'state_root': '/state'}, {})
        self.assertIsNone(result)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs['creationflags'], 0x08000000)

    def test_non_windows_rpc_omits_windows_flags(self):
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b'null'))
        with patch.object(demo, 'os', SimpleNamespace(name='posix')), \
             patch.object(demo, 'subprocess', SimpleNamespace(run=run)):
            demo.rpc({'python': '/python', 'release': '/release', 'state_root': '/state'}, {})
        self.assertNotIn('creationflags', run.call_args.kwargs)

    def test_active_entry_windows_connection_hides_console(self):
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b'{}'))
        with patch.object(active, 'os', SimpleNamespace(name='nt')), \
             patch.object(active, 'subprocess', SimpleNamespace(run=run, CREATE_NO_WINDOW=0x08000000)):
            self.assertEqual(active.call('print(1)'), '{}')
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs['creationflags'], 0x08000000)

    def test_evaluation_pause_prevents_all_remote_calls(self):
        self.save_config(paused=True)
        with patch.object(evaluation, 'exchange') as remote:
            result = evaluation.main(self.config)
        remote.assert_not_called()
        self.assertEqual(result['state'], 'PAUSED')

    def test_evaluation_connection_failure_exits_without_retry(self):
        self.save_config()
        with patch.object(evaluation, 'exchange', side_effect=ValueError('offline')) as remote, \
             patch.object(evaluation.time, 'sleep', side_effect=AssertionError('Must not retry')):
            result = evaluation.main(self.config)
        remote.assert_called_once()
        self.assertEqual(result['state'], 'NEEDS_REVIEW')
        self.assertEqual(json.loads((self.root/'state/status.json').read_text())['state'], 'NEEDS_REVIEW')

    def test_demo_pause_prevents_all_remote_calls(self):
        self.save_config(paused=True)
        with patch.object(demo, 'rpc') as remote:
            result = demo.main(self.config)
        remote.assert_not_called()
        self.assertEqual(result['state'], 'PAUSED')

    def test_demo_connection_failure_exits_without_retry(self):
        self.save_config()
        with patch.object(demo, 'rpc', side_effect=ValueError('offline')) as remote, \
             patch.object(demo.time, 'sleep', side_effect=AssertionError('Must not retry')):
            result = demo.main(self.config)
        remote.assert_called_once()
        self.assertEqual(result['state'], 'NEEDS_REVIEW')

    def test_space_pause_prevents_all_remote_calls(self):
        self.save_config(paused=True)
        with patch.object(space.sys, 'argv', ['agent', '--config', str(self.config)]), \
             patch.object(space, 'remote') as remote:
            space.main()
        remote.assert_not_called()
        self.assertEqual(json.loads((self.root/'connection/status.json').read_text())['state'], 'PAUSED')

    def test_space_connection_failure_exits_without_retry(self):
        self.save_config()
        with patch.object(space.sys, 'argv', ['agent', '--config', str(self.config)]), \
             patch.object(space, 'remote', side_effect=ValueError('offline')) as remote, \
             patch.object(space.time, 'sleep', side_effect=AssertionError('Must not retry')):
            space.main()
        remote.assert_called_once()
        self.assertEqual(json.loads((self.root/'connection/status.json').read_text())['state'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
