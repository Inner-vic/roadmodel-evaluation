"""Exact arithmetic and boundary checks for the GT memory optimization."""
import sys
import unittest
from pathlib import Path

import numpy as np
from pcm_eval.memory_gt import combine_geosem, finalize_logits, tiled_road


class ArrayTests(unittest.TestCase):
    def test_combine_exact_with_zero_one_and_random_probability(self):
        rng = np.random.default_rng(13)
        for kind in ('owned', 'view', 'readonly'):
            scores = rng.normal(size=(1003, 27)).astype(np.float32)
            bins = rng.random(1003, dtype=np.float32)
            bins[:3] = [0, 1, .5]
            expected = np.concatenate([scores[:, :-1] * bins[:, None], (1.0 - bins)[:, None]], axis=-1).astype(np.float32)
            if kind == 'view':
                scores = scores[::-1]; bins = bins[::-1]; expected = expected[::-1]
            elif kind == 'readonly':
                scores.flags.writeable = False
            np.testing.assert_array_equal(combine_geosem(scores, bins), expected)

    def test_combine_aliasing_bins_is_exact(self):
        scores = np.arange(270, dtype=np.float32).reshape(10, 27).copy() / 270
        bins = scores[:, -1]
        expected = np.concatenate([scores[:, :-1] * bins[:, None], (1.0 - bins)[:, None]], axis=-1).astype(np.float32)
        np.testing.assert_array_equal(combine_geosem(scores, bins), expected)

    def test_normalization_matches_fallback_and_threshold_exactly(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=(1031, 27)).astype(np.float32)
        sums = rng.random(1031, dtype=np.float32)
        sums[:5] = [0, 1e-12, 1e-9, np.nextafter(np.float32(1e-9), np.float32(1)), -1]
        expected = np.zeros_like(scores); mask = sums > 1e-9
        expected[mask] = scores[mask] / sums[mask, None]
        expected[~mask, :26] = 1.0 / 26
        actual = finalize_logits(scores.copy(), sums, 27, rows=13)
        np.testing.assert_array_equal(actual, expected)


class PillarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
        if not path.exists():
            raise unittest.SkipTest('Supplier CPU module is installed on the workstation')
        sys.path.insert(0, str(path))
        from local_aggregate_cpu import output_single_layer_road
        cls.reference = staticmethod(output_single_layer_road)

    def test_all_reference_options_and_tile_boundaries(self):
        rng = np.random.default_rng(42)
        H, W, D = 7, 5, 11
        scores = rng.random((H * W * D, 27), dtype=np.float32)
        scores[:D * 2] = .5  # Ties must retain the original first-index rule.
        labels = scores.argmax(axis=1).astype(np.int32)
        labels[:D * 3] = 16  # Exercise bumper-top selection.
        density = rng.random(H * W * D, dtype=np.float32)
        for tile in (1, 13, 256):
            for use_density in (False, True):
                for bumper in (False, True):
                    for ds in (None, [1, 2], list(range(27))):
                        options = dict(use_density=use_density, density=density,
                                       output_bumper_top=bumper, ds_labels=ds)
                        expected = self.reference(scores, labels, H, W, D, **options)
                        actual = tiled_road(self.reference, pillars_per_tile=tile)(scores, labels, H, W, D, **options)
                        np.testing.assert_array_equal(actual, expected)

    def test_missing_density_still_fails(self):
        scores = np.zeros((8, 27), np.float32)
        with self.assertRaisesRegex(ValueError, 'density'):
            tiled_road(self.reference)(scores, np.zeros(8), 2, 2, 2, use_density=True)


if __name__ == '__main__':
    unittest.main()
