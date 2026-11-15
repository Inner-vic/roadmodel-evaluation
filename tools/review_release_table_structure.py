"""Summarize actual business-report tables without changing published reports."""
import html
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'evidence/source-docs/release-template-review-20261029.json'
response = json.loads(source.read_text(encoding='utf-8-sig'))
if not response.get('ok'):
    raise RuntimeError('Source document could not be read')
doc = response['data']['document']
content = doc['content']


def text(value):
    return html.unescape(re.sub(r'<[^>]+>', '', value)).strip()


tables = []
for table in re.findall(r'<table\b[^>]*>.*?</table>', content, flags=re.S):
    rows = []
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, flags=re.S):
        rows.append([text(cell) for cell in re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', row, flags=re.S)])
    tables.append({'header_and_first_rows': rows[:3], 'row_count': len(rows)})

result = {
    'status': 'design_review_only; published reports unchanged',
    'source_document_id': doc['document_id'],
    'source_revision_id': doc['revision_id'],
    'title': [text(x) for x in re.findall(r'<title[^>]*>(.*?)</title>', content, flags=re.S)],
    'headings': [text(x) for x in re.findall(r'<h[1-6]\b[^>]*>(.*?)</h[1-6]>', content, flags=re.S)],
    'tables': tables,
    'single_run_design': {
        'separate_evaluation_types': ['model_accuracy', 'compile_alignment'],
        'model_accuracy': 'Preserve OCC, trajectory, vision sections; remove historical-version columns; preserve existing vision summary and subclass tables.',
        'compile_alignment': 'Preserve ONNX and evaluated board-variant columns within this run; preserve output-tensor similarity table; keep tensor consistency separate from GT metrics.',
        'comparison_dimension': 'A future release report compares the same evaluation type across two runs/versions; ONNX-versus-board is an intrinsic within-run comparison.',
        'metadata': ['model_artifact_binding', 'dataset_version', 'selection', 'gt_definition', 'metric_definition', 'successful_and_expected_frames', 'raw_result_references'],
        'missing_data': 'Retain business rows, distinguish unmeasured, missing GT, unsupported metric and measured zero; never copy historical values.',
    },
}
output = root / 'evidence/report-template-review-20261029/table-structure-and-design.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
