"""Memory-bounded replacements for the CPU reference's array plumbing.

Gaussian accumulation, class arithmetic, pillar scoring and metric definitions
stay unchanged. Installation is process-local; the supplier package is not edited.
"""
from __future__ import annotations

import functools
import inspect


def finalize_logits(c_acc, prob_sum, num_channels, *, rows=16384):
    """Normalize a disposable accumulator in place using the reference division."""
    import numpy as np
    if c_acc.dtype != np.float32 or not c_acc.flags.writeable or not c_acc.flags.owndata:
        c_acc = np.array(c_acc, dtype=np.float32, copy=True)
    for start in range(0, len(c_acc), rows):
        block = c_acc[start:start + rows]
        sums = prob_sum[start:start + rows]
        valid = sums > 1e-9
        np.divide(block, sums[:, None], out=block, where=valid[:, None])
        block[~valid, :num_channels - 1] = 1.0 / (num_channels - 1)
        block[~valid, num_channels - 1:] = 0
    return c_acc


def combine_geosem(logits, bin_logits):
    """Consume the caller's disposable logits instead of copying ~200 MiB twice."""
    import numpy as np
    result = np.asarray(logits, dtype=np.float32)
    if not result.flags.writeable or not result.flags.owndata:
        result = result.copy()
    bins = np.asarray(bin_logits, dtype=np.float32).reshape(-1)
    # Protect unusual callers whose bins alias the output; production bins do not.
    if np.shares_memory(result, bins):
        bins = bins.copy()
    np.multiply(result[:, :-1], bins[:, None], out=result[:, :-1])
    np.subtract(np.float32(1.0), bins, out=result[:, -1])
    return result


def tiled_road(reference, *, pillars_per_tile=256):
    """Apply the unmodified supplier function to independent pillar blocks."""
    if type(pillars_per_tile) is not int or pillars_per_tile < 1:
        raise ValueError('Invalid pillar tile size')
    signature = inspect.signature(reference)

    @functools.wraps(reference)
    def run(geosem, pred_label, H, W, D, **kwargs):
        import numpy as np
        # Bind once so unsupported keyword arguments keep the supplier's error.
        signature.bind(geosem, pred_label, H, W, D, **kwargs)
        n_pillars = int(H) * int(W)
        scores = np.asarray(geosem, dtype=np.float32).reshape(n_pillars * int(D), -1)
        labels = np.asarray(pred_label, dtype=np.int32).reshape(n_pillars * int(D))
        density = kwargs.get('density')
        if density is not None:
            density = np.asarray(density, dtype=np.float32).reshape(n_pillars, int(D))
        output = np.empty((n_pillars, int(D)), np.int32)
        for start in range(0, n_pillars, pillars_per_tile):
            end = min(start + pillars_per_tile, n_pillars)
            options = dict(kwargs)
            if density is not None:
                options['density'] = density[start:end]
            output[start:end] = reference(scores[start * D:end * D], labels[start * D:end * D],
                                           end - start, 1, D, **options).reshape(end - start, D)
        return output.reshape(int(H), int(W), int(D))
    return run


def install(*, pillars_per_tile=256):
    """Patch only this isolated GT worker, after loading its original modules."""
    import local_aggregate_cpu as cpu
    import run_localagg_cpu as runner
    if getattr(runner, '_roadmodel_memory_gt_installed', False):
        return
    cpu._finalize_logits = finalize_logits
    runner.combine_geosem = combine_geosem
    runner.output_single_layer_road = tiled_road(runner.output_single_layer_road,
                                                pillars_per_tile=pillars_per_tile)
    runner._roadmodel_memory_gt_installed = True
