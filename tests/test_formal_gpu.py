import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from pcm_eval.common import save_json
from tools.formal_gpu import PROBE, selected_gpu


class FormalGpuTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / 'evidence' / ('gpu-unit-' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        assert self.root.resolve().is_relative_to((Path(__file__).resolve().parents[1] / 'evidence').resolve())
        shutil.rmtree(self.root)

    def test_current_explicit_gpu_and_original_default(self):
        self.assertEqual(selected_gpu(self.root), '1')
        save_json(self.root / 'model/gpu-selection.json', {'physical_index': 2, 'visible_indices': list(range(8))})
        self.assertEqual(selected_gpu(self.root), '2')

    def test_invalid_or_unallocated_gpu_is_rejected(self):
        for value in (True, -1, 8, '2'):
            save_json(self.root / 'model/gpu-selection.json', {'physical_index': value, 'visible_indices': list(range(8))}, overwrite=True)
            with self.assertRaises(ValueError): selected_gpu(self.root)

    def test_probe_reads_logical_zero_after_binding_requested_physical_card(self):
        def mem(device):
            self.assertEqual(device, 0)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
            return 12 * 1024**3, 40 * 1024**3
        fake = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=mem))
        output = io.StringIO()
        with patch.dict(sys.modules, {'torch': fake}), patch.object(sys, 'argv', ['probe', '2']), patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(output):
            exec(PROBE, {})
        result = json.loads(output.getvalue())
        self.assertEqual(result['physical_GPU'], '2')
        self.assertEqual(result['free_GiB'], 12)


if __name__ == '__main__': unittest.main()
