"""Verify retained evidence, rendered facts and the scoped publication receipt."""
import hashlib,json,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
root=Path('evidence/reports/model-training-semantic-audit-20261027')
report=json.loads((root/'report.json').read_text(encoding='utf-8'));a=report['analysis']
assert report['generation']=={'existing_artifacts_only':True,'new_training_runs':0,'new_inference_frames':0,'PCM_release_template_used':False}
assert len(a['epochs'])==60 and a['checkpoint']['epoch']==44
for r in a['epochs']:assert sum(c['support'] for c in r['classes'].values())==r['validation_valid_points']
with zipfile.ZipFile(root/'evidence.zip') as z:
    manifest=json.loads(z.read('manifest.json'));assert not any(x.endswith('.pth') for x in z.namelist())
    for item in manifest:
        raw=z.read(item['path']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        assert hashlib.sha256(Path(item['source']).read_bytes()).hexdigest()==item['sha256']
pub=root/'publication';receipt=json.loads((pub/'publication-receipt.json').read_text(encoding='utf-8'));assert receipt['result']=='verified'
before=json.loads((pub/'before.json').read_text(encoding='utf-8-sig'))['data']['document'];after=json.loads((pub/'after.json').read_text(encoding='utf-8-sig'))['data']['document']
b=ET.fromstring('<doc>'+before['content']+'</doc>');t=ET.fromstring('<doc>'+after['content']+'</doc>')
text=''.join(t.itertext())
for s in ['14.45%','28.10%','11.43%～15.71%','0.01232','未计算（N/A）','水泥召回门槛']:assert s in text,s
assert all(''.join(e.itertext()) in text for e in b.iter('table'))
assert len(list(t.iter('source')))==len(list(b.iter('source')))+2
proof=pub/'feishu-training-audit-proof.png'
if proof.exists() and proof.read_bytes().startswith(b'\xff\xd8\xff'):
    target=proof.with_suffix('.jpg');assert not target.exists();proof.rename(target)
proof=next(pub.glob('feishu-training-audit-proof.*'))
assert proof.read_bytes().startswith((b'\x89PNG\r\n\x1a\n',b'\xff\xd8\xff'))
result={'state':'verified','epochs_cross_checked':60,'all_tfrecord_crc_verified':a['tensorboard']['all_record_checksums_verified'],'zip_entries_checked':len(manifest),'report_revision':after['revision_id'],'native_tables_and_attachments_verified':True,'screenshot_saved':True,'new_training_runs':0,'new_inference_frames':0}
(pub/'delivery-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
