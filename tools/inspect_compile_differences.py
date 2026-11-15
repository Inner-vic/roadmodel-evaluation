"""Read retained tensors; preserve the supplied scoring formula and raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx


def inspect(root, top=3):
    root = Path(root).resolve()
    state = json.loads((root / 'state.json').read_text())
    integrity = json.loads((root / 'output_integrity.json').read_text())
    if integrity.get('state') != 'fresh_outputs_checked':
        raise ValueError('Only inspect runs with verified output provenance')
    ranked = []
    for folder in sorted(root.glob('frame_*')):
        detail = json.loads((folder / 'cosine_board_onnx.json').read_text())
        index = str(int(folder.name.split('_')[-1]))
        for row in detail['outputs']:
            if row.get('status') != 'ok' or not np.isfinite(row.get('cosine', np.nan)):
                raise ValueError('Incomplete or non-finite comparison')
            ranked.append((row['cosine'], folder, state['frames'][index]['meta']['frame_id'], detail, row))
    ranked.sort(key=lambda item: item[0])
    selected, seen = [], set()
    # Inspect the worst occurrence of each distinct output, not three copies of one tensor.
    for _, folder, frame_id, detail, row in ranked:
        name = row['onnx_output_name']
        if name in seen:
            continue
        seen.add(name)
        board = (Path(detail['board_dir']) / row['board_file']).resolve()
        candidates = list(Path(detail['ort_dir']).glob(row['onnx_stem'] + '.*.bin'))
        if len(candidates) != 1:
            raise ValueError('Ambiguous ONNX output')
        ort = candidates[0].resolve()
        board.relative_to(root)
        ort.relative_to(root)
        b = np.fromfile(board, dtype=row['dtype_board']).reshape(row['shape_board']).astype(np.float64)
        a = np.fromfile(ort, dtype=row['dtype_onnx']).reshape(row['shape_onnx']).astype(np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid tensor shape or values')
        delta = np.abs(a - b)
        norm = float(np.linalg.norm(a))
        item = {'frame_folder': folder.name, 'frame_id': frame_id, 'output': name,
                'shape': list(a.shape), 'dtype': row['dtype_onnx'], 'cosine': row['cosine'],
                'max_abs_diff': float(delta.max()), 'mean_abs_diff': float(delta.mean()),
                'abs_diff_percentiles': {str(k): float(np.percentile(delta, k)) for k in (50, 90, 95, 99)},
                'relative_l2': float(np.linalg.norm(a - b) / norm) if norm else None,
                'exact_equal_fraction': float((a == b).mean()),
                'source_hashes': {k: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for k, p in [('board', board), ('onnx', ort)]}}
        if a.ndim > 1 and a.shape[-1] in (2, 27):
            item['last_axis_argmax'] = {'different': int(np.sum(a.argmax(-1) != b.argmax(-1))),
                                       'count': int(a.argmax(-1).size),
                                       'interpretation': 'tensor diagnostic; not a task accuracy metric'}
        if a.size <= 10:
            item.update(onnx_values=a.tolist(), board_values=b.tolist())
        selected.append(item)
        if len(selected) >= top:
            break
    request = json.loads((root / 'request.json').read_text())
    model = onnx.load(request['model']['onnx'], load_external_data=False)
    by_output = {v: node for node in model.graph.node for v in node.output}
    for item in selected:
        node, chain = by_output.get(item['output']), []
        for _ in range(3):
            if node is None:
                break
            chain.append({'name': node.name, 'type': node.op_type,
                          'inputs': list(node.input), 'outputs': list(node.output)})
            node = by_output.get(node.input[0]) if node.input else None
        item['producer_chain_first_input_only'] = chain
    return {'schema': 'pcm.tensor-difference-analysis/v1', 'run_id': root.name,
            'compared_frames': state['total_frames'], 'compared_output_pairs': len(ranked),
            'ranking': 'lowest cosine per distinct output', 'checks': selected,
            'quality_assessment': 'not_assessed',
            'limitations': ['Argmax differences do not establish GT accuracy.',
                            'Operator chains identify outputs, not the cause of compilation differences.',
                            'Exact equality here differs from upstream match_ratio (rtol=atol=1e-3 for floats).']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = inspect(args.run_dir)
    with Path(args.output).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'checks': len(value['checks']), 'frames': value['compared_frames']}))
