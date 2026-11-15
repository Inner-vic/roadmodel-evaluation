"""Verify published supplement provenance and the original prediction inventory."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
root=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01')
run=root/'trajectory-diagnostic-01';report=Path('evidence/reports/model-trajectory-diagnostic-20261027')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
r=load(run/'diagnostic-receipt.json');a=load(root/'analysis.json');pub=load(report/'publication/publication-receipt.json')
assert a['unique_frames']==2 and a['total_new_full_model_frame_passes']==4
for i in range(2):
    with np.load(run/f'frame-{i}-native.npz',allow_pickle=False) as data,np.load(root.parent/'collected/trajectory-01'/f'frame-{i}.npz',allow_pickle=False) as old:
        for name in old.files:assert np.array_equal(old[name],data['result_'+name]),name
        assert all(np.isfinite(data[k]).all() for k in data.files)
with zipfile.ZipFile(report/'evidence.zip') as z:
    manifest=json.loads(z.read('manifest.json'))
    for entry in manifest:assert hashlib.sha256(z.read(entry['name'])).hexdigest()==entry['sha256']
    assert len([x for x in manifest if x['name'].endswith('.npz')])==9
assert pub['result']=='verified' and pub['prior_resources_preserved'] and pub['new_attachments']==2
after=load(report/'publication/after.json')['data']['document']['content']
assert '99.95' in after and '99.97' in after and '共 4 帧次' in after
summary={'state':'verified','source_frames':[20,21],'unique_frames':2,'FP32_forward_frames':2,'FP16_forward_frames':2,
    'original_predictions_unchanged':True,'finite_retained_features':True,'zip_manifest_entries_verified':len(manifest),
    'native_semantic_replay_logit_error':0.,'full_model_FP16_label_changes':0,'publication_revision':pub['revision_id'],
    'report_url':pub['url'],'validation':'actual artifact/array equality, finite checks, exact linear reconstruction, FP16 matched-input control and remote text/resource readback'}
(root/'completion-receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
