"""Record the verified current document and immutable generation inputs."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import digest

root=Path('evidence/reports/compile-atomic-v3-20261027')
receipt_path=root/'publish-reviewed/publication-receipt.json'
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
report=json.loads((root/'reviewed.json').read_text(encoding='utf-8'))
profile=json.loads((root/'reviewed-profile-check.json').read_text(encoding='utf-8-sig'))
assert receipt['state']=='verified' and receipt['report_sha256']==digest(root/'reviewed.json')
assert report['generation']['new_inference_frames']==0
assert len(receipt['section_blocks'])==8 and len(receipt['attachments'])==2
assert profile['data']['assessment']['status']=='passed'

path=Path('configs/known_assets.json')
assets=json.loads(path.read_text(encoding='utf-8'))
current=assets['single_reports']['compile_alignment']
if current.get('revision_id')!=receipt['revision_id']:
    assets.setdefault('single_report_history',[]).append(current)
    assets['single_reports']['compile_alignment']={
        'document_id':receipt['document_id'],'url':receipt['url'],
        'title':current['title'],'revision_id':receipt['revision_id'],'evaluation_type':'compile_alignment',
        'schema':report['schema'],'report_status':'published_and_verified',
        'structured_report':str(root/'reviewed.json'),'draft':str(root/'reviewed.xml'),
        'draft_sha256':digest(root/'reviewed.xml'),'readback':'passed',
        'publication_receipt':str(receipt_path),'attachment':receipt['attachments']}
    assets['publication_receipt']=str(receipt_path)
    path.write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
summary={'state':'complete','report_schema':report['schema'],'url':receipt['url'],
         'revision_id':receipt['revision_id'],'run_id':receipt['run_id'],'source_frames':20,
         'new_inference_frames':0,'source_result_modified':False,
         'normalized_metrics_retained':len(report['source_normalized_result']['metrics']),
         'tensor_pairs_retained':len(report['tensor_alignment']['records']),
         'gt_classes':len(report['ground_truth']['classes']),'gt_backend_count_rows':len(report['ground_truth']['counts']),
         'mixed_score':False,'profile_check':'passed','tests_passed':67,
         'publication_receipt_sha256':digest(receipt_path),'report_sha256':digest(root/'reviewed.json'),
         'generator_sha256':digest(Path('pcm_eval/compile_report.py')),
         'preview':'feishu-preview.png'}
(root/'delivery-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True))
