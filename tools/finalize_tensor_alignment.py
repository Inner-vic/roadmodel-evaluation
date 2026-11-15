"""Persist a GT-independent summary with the supplied script's own accumulator.

Upstream gates cosine_summary.json on the existence of GT metric accumulators.
--skip-metrics still writes real per-frame comparisons, so summarize those files
without rerunning inference or changing the original scoring formula.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if (root / 'cosine_summary.json').exists() and not args.verify_existing:
        raise ValueError('Summary already exists; preserve original evidence')
    state = json.loads((root / 'state.json').read_text())
    if state.get('config', {}).get('skip_metrics') is not True:
        raise ValueError('This finalizer is only for explicitly GT-free runs')
    if state.get('summary', {}).get('done') != state.get('total_frames'):
        raise ValueError('The run did not complete all requested frames')
    package = Path(state['config']['repo']).resolve()
    sys.path.insert(0, str(package))
    from cosine_compare import CosineAccumulator
    accumulator, sources = CosineAccumulator(), []
    for index, frame_state in sorted(state['frames'].items(), key=lambda item: int(item[0])):
        relative = 'frame_{:06d}/cosine_board_onnx.json'.format(int(index))
        path = root / relative
        raw = path.read_bytes()
        report = json.loads(raw)
        accumulator.update(report, frame_id=frame_state['meta']['frame_id'])
        sources.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest()})
    if len(sources) != state['total_frames']:
        raise ValueError('Per-frame evidence is incomplete')
    summary = accumulator.summary()
    receipt = {'schema': 'pcm.tensor-summary-receipt/v1',
               'method': 'supplied cosine_compare.CosineAccumulator',
               'accumulator_sha256': hashlib.sha256((package / 'cosine_compare.py').read_bytes()).hexdigest(),
               'sources': sources}
    for name, value in [('cosine_summary.json', summary), ('tensor_summary_receipt.json', receipt)]:
        path = root / name
        if path.exists() and args.verify_existing:
            if json.loads(path.read_text()) != value:
                raise ValueError('Existing evidence differs from retained frame data: ' + name)
        else:
            with path.open('x') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'frames': summary['num_frames'], 'outputs': len(summary['tensors']),
                      'mean_cosine_all': summary['mean_cosine_all']}))


if __name__ == '__main__':
    main()
