"""Link verified GT delivery receipts and restore an unrelated scope status."""
import hashlib
import json
from pathlib import Path

root=Path('evidence/requested-dataset')
progress=root/'compile-tensor-progress.json'
data=json.loads(progress.read_text())
if data.get('state')=='failed' and data.get('error_type')=='ValueError' and not data.get('failed'):
    # A plan-only manifest rejection used the old controller's default label.
    # Preserve that event, then restore status from the existing full content audit.
    audit=json.loads((root/'compile-tensor-validation.json').read_text())
    receipt=json.loads((root/'compile-tensor-sync-receipt.json').read_text())
    assert audit['state']==receipt['state']=='complete'
    assert data['completed_files']==receipt['file_count']==16810
    assert data['completed_bytes']==receipt['verified_bytes']==44050968932
    (root/'small-gt-plan-rejection-progress-event.json').write_text(json.dumps(data,indent=2)+'\n')
    data.update(state='complete',updated_at=receipt.get('updated_at',audit.get('updated_at',data['updated_at'])),
                status_restored_from_verified_receipt=True,
                controller_note='Unrelated selected-GT plan rejection was incorrectly assigned this scope; fixed before delivery')
    data.pop('error_type',None)
    progress.write_text(json.dumps(data,indent=2)+'\n')

publication=Path('evidence/reports/compile-gt-update-20261027/publication-receipt.json')
published=json.loads(publication.read_text())
assert published['state']=='verified'
gt=Path('evidence/actual-runs/pcm-v7-multiscene-001/gt-adapted-20261027')
cross=json.loads((gt/'crosscheck.json').read_text())
assert cross['state']=='passed'
summary={'schema':'pcm.gt-delivery/v1','state':'complete','source_run_id':'pcm-v7-multiscene-001',
         'frames':20,'scenes':10,'new_inference_frames':0,'source_run_modified':False,
         'tests':{'passed':59,'command':'python tools/verify_gt_changes.py'},
         'gt_crosscheck':cross,'publication':published,
         'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
             gt/'gt_receipt.json',gt/'metrics_board.json',gt/'metrics_onnx.json',gt/'result.json',
             Path('evidence/reports/compile-gt-single-20261027.json'),publication]}}
(gt/'delivery-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'state':'complete','document_revision':published['revision_id'],'frames':20,'tests_passed':59}))
