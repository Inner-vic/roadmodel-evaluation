import json,xml.etree.ElementTree as ET
from pathlib import Path
root=Path('evidence/reports/model-atomic-v1-20261027')
r=json.loads((root/'report.json').read_text(encoding='utf-8'))
print('summaries',[{k:s[k] for k in ['scope','with_gt','without_gt','present_miou']} for s in r['occ_sections']])
print('checkpoints',r['binding']['main_checkpoint'],r['binding']['classifier_checkpoint'])
print('xml',[(e.tag,''.join(e.itertext())[:90]) for e in ET.fromstring('<doc>'+(root/'report.xml').read_text(encoding='utf-8')+'</doc>') if e.tag in ['title','h1','h2']])
prior=json.loads(Path('evidence/model-evaluation/model-reference-20261027/model-report-before.json').read_text(encoding='utf-8-sig'))['data']['document']
tree=ET.fromstring('<doc>'+prior['content']+'</doc>')
print('prior_doc',{'revision':prior.get('revision_id'),'refs':prior.get('reference_map'),
 'top_blocks':len(tree),'first_id':list(tree)[1].get('id'),'last_id':list(tree)[-1].get('id'),
 'resource_tags':[x.tag for x in tree.iter() if x.tag in ['source','img','sheet','whiteboard']],
 'end_text':prior['content'][-1800:]})
