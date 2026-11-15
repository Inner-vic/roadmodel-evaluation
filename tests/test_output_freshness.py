import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/benchEval-skill-20261020'))
from board_infer_client import BoardInferClient, BoardInferConfig, INPUT_BIN_NAMES


class LocalSFTP:
    def __init__(self, root, reject_remove=False):
        self.root, self.reject_remove = root, reject_remove

    def path(self, remote):
        return self.root / remote.lstrip('/')

    def file(self, path, mode):
        return self.path(path).open(mode)

    def listdir(self, path):
        return [p.name for p in self.path(path).iterdir()]

    def stat(self, path):
        return self.path(path).stat()

    def lstat(self, path):
        return self.path(path).lstat()

    def mkdir(self, path):
        self.path(path).mkdir()

    def remove(self, path):
        if self.reject_remove:
            raise PermissionError('regression fixture')
        self.path(path).unlink()

    def rename(self, source, destination):
        self.path(source).rename(self.path(destination))


class OutputFreshnessTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / '.tmp'
        base.mkdir(exist_ok=True)
        self.root = base / uuid.uuid4().hex
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.pack = self.root / 'packup'
        frame = self.pack / 'data/frame_0'
        frame.mkdir(parents=True)
        self.name = 'tensor.1.float16.bin'
        (frame / self.name).write_bytes(b'archived-old-output')
        (self.pack / 'model_info.json').write_text(json.dumps({'output': [{'data_path': self.name}]}))
        (self.pack / 'MergedNet.bin').write_bytes(b'weights')
        self.input_name = next(iter(INPUT_BIN_NAMES))
        (frame / self.input_name).write_bytes(b'current-input')

    def client(self, reject_remove=False):
        client = BoardInferClient(BoardInferConfig(packup_dir='/packup'))
        client._sftp = LocalSFTP(self.root, reject_remove)
        self.executed = False

        def infer(*args, **kwargs):
            self.executed = True
            (self.pack / self.name).write_bytes(b'fresh-output')
            return 0, '', ''

        client._exec = infer
        return client

    def test_sample_frame_dump_cannot_overwrite_fresh_cwd_dump(self):
        client = self.client()
        client.run_inference()
        self.assertEqual((self.pack / 'data/frame_0/infer_output' / self.name).read_bytes(), b'fresh-output')
        self.assertEqual((self.pack / 'MergedNet.bin').read_bytes(), b'weights')
        self.assertEqual((self.pack / 'data/frame_0' / self.input_name).read_bytes(), b'current-input')
        self.assertEqual(len(client.last_output_reset['removed_files']), 1)

    def test_unremovable_stale_output_stops_before_inference(self):
        client = self.client(reject_remove=True)
        with self.assertRaises(PermissionError):
            client.run_inference()
        self.assertFalse(self.executed)


if __name__ == '__main__':
    unittest.main()
