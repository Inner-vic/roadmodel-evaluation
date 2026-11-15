"""Preserve the exact inspected dataset/metric source without changing the checkout."""
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '524e079723820582de1ebfe88a2053b072c1893d'
FILES = ['dataset/dataset.py', 'dataset/transform_3d.py', 'dataset/utils.py', 'config/4-0-0.py', 'config/3-0-0-temporal-dit.py',
         'config/_base_/base.py', 'misc/eval/gt_label_utils.py', 'misc/eval/metric_util.py',
         'tools/dataset_process/1-DATASET_MANAGEMENT/check_v7_v15_golden.py',
         'tools/tools_eval/eval_occ_targets_cpp_aligned.py']
FILES += ['config/2-1-0.py', 'tools/dataset_process/Occ&dataLink/occlabel_26labels_v3.py',
          'tools/dataset_process/Occ&dataLink/occlabel_26labels_v3_1.py',
          'tools/dataset_process/Occ&dataLink/smartChassisDatasetProcess_v3_generate_files.py']
FILES += ['tools/tools_eval/auto_convert_dataset.sh',
          'tools/tools_eval/occ-dataset-iterator/step4_process_raw_data.py',
          'tools/dataset_process/Occ&dataLink/occlabel_26labels_v3_noFilter.py',
          'misc/eval/train_eval_utils.py']

if __name__ == '__main__':
    root = Path('inspection/gt-source-' + COMMIT[:12])
    rows = []
    for name in FILES:
        p = subprocess.run(['git', '-C', 'inspection/roadmodel-dev-v41', 'show', COMMIT + ':' + name],
                           capture_output=True, check=True)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == p.stdout
        else:
            with target.open('xb') as stream:
                stream.write(p.stdout)
        rows.append({'path': name, 'bytes': len(p.stdout), 'sha256': hashlib.sha256(p.stdout).hexdigest()})
    record = {'commit': COMMIT, 'root': str(root), 'files': rows}
    (root / 'source-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
