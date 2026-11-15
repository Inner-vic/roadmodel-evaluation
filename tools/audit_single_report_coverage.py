"""Describe available source fields versus the current rendered single report."""
import json
from collections import Counter
from pathlib import Path

root=Path('evidence/actual-runs/pcm-v7-multiscene-001')
report=json.loads(Path('evidence/reports/compile-gt-single-20261027.json').read_text(encoding='utf-8'))
result=json.loads((root/'gt-adapted-20261027/result.json').read_text(encoding='utf-8'))
cos=json.loads((root/'collected/cosine_summary.json').read_text(encoding='utf-8'))
metrics=result['metrics']
def key(metric):
    return (metric['name'],tuple(sorted(metric.get('dimensions',{}).items())))
shown=set()
for table in report['tables']:
    for row in table['rows']:
        for cell in row.values():
            if isinstance(cell,dict) and cell.get('status')=='available':
                shown.add((cell['metric'],tuple(sorted(cell.get('dimensions',{}).items()))))
# Report cells intentionally use only meaningful dimensions, omitting class_id.
def rendered(m):
    return any(n==m['name'] and all(m.get('dimensions',{}).get(k)==v for k,v in dims) for n,dims in shown)
missing=[m for m in metrics if not rendered(m)]
perframe=[o for f in cos.get('frames',[]) for o in f.get('outputs',[])]
backends={b:json.loads((root/f'gt-adapted-20261027/metrics_{b}.json').read_text()) for b in ['board','onnx']}
audit={'schema':'pcm.single-report-coverage-audit/v1','source_run':'pcm-v7-multiscene-001',
       'current_report_schema':report['schema'],
       'normalized_metrics':len(metrics),'visible_numeric_metric_cells':report['filled_metric_cells'],
       'normalized_metrics_not_shown_as_table_values':len(missing),
       'not_shown_by_metric_name':dict(Counter(m['name'] for m in missing)),
       'tensor_summary_fields':list(cos),
       'tensor_aggregate_fields':sorted({k for v in cos.get('tensors',{}).values() for k in v}),
       'per_frame_tensor_pairs':len(perframe),'per_frame_tensor_fields':sorted({k for o in perframe for k in o}),
       'gt_backend_fields':list(backends['board']),
       'gt_semantic_classes':len(backends['board']['class_support']),
       'gt_classes_with_samples':len(backends['board']['present_class_ious']),
       'gt_classes_without_samples':sum(r['seen']==0 for r in backends['board']['class_support']),
       'current_report_class_rows':len(report['tables'][0]['rows'])-3,
       'source_backends':list(backends),
       'notes':['The current JSON attachment retains normalized metrics and GT provenance; this does not mean the Feishu body displays every source field.',
                'Differences, postprocessing checks and repeatability are separate diagnostics with their own frame scopes. They are not 20-frame quality metrics.']}
path=Path('evidence/reports/single-report-design-20261027/coverage-audit.json')
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=True,indent=2))
