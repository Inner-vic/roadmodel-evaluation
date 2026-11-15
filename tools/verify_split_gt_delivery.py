import hashlib,json,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
root=Path('evidence/reports/model-split-gt-audit-20261027/final');pub=root/'publication'
report=json.loads((root/'report.json').read_text(encoding='utf-8'));a=report['analysis'];assert a['file_collection']['new_inference_frames']==2
assert a['model_control']['state']=='verified' and a['model_control']['frames'][0]['classes']['2']['correct']==657
with zipfile.ZipFile(root/'evidence.zip') as z:
    files=json.loads(z.read('manifest.json'))
    for f in files:
        b=z.read(f['path']);assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
        assert hashlib.sha256(Path(f['source']).read_bytes()).hexdigest()==f['sha256']
    assert not any(n.endswith('.pth') for n in z.namelist())
before=json.loads((pub/'before.json').read_text(encoding='utf-8-sig'))['data']['document'];after=json.loads((pub/'after.json').read_text(encoding='utf-8-sig'))['data']['document']
b=ET.fromstring('<doc>'+before['content']+'</doc>');t=ET.fromstring('<doc>'+after['content']+'</doc>');text=''.join(t.itertext())
for value in ['657 / 657（100.00%）','0.000598','8.083512','8.6647','619,105','97.06%']:assert value in text,value
assert all(''.join(e.itertext()) in text for e in b.iter('table'))
assert len(list(t.iter('source')))==len(list(b.iter('source')))+2 and len(list(t.iter('img')))==len(list(b.iter('img')))+2
heading=next(e for e in t.iter('h3') if ''.join(e.itertext())=='统一模型与验证预处理的两帧对照')
proof=pub/'feishu-split-control-proof.jpg'
if proof.exists():assert proof.read_bytes().startswith((b'\xff\xd8\xff',b'\x89PNG'))
v={'state':'verified','document_revision':after['revision_id'],'evidence_entries':len(files),'control_source_forward_frames':2,'new_training_runs':0,'generation_and_publication_new_forward_frames':0,'prior_tables_and_resources_preserved':True,'contrast_heading_id':heading.get('id'),'UI_screenshot_saved':proof.exists(),'runtime_restoration_UI_proof':str(Path('evidence/model-evaluation/split-gt-audit-20261027/space-restored.jpg'))}
(pub/'delivery-verification.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
