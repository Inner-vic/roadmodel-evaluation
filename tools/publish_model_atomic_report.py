"""Publish one already-generated native-model report; never starts evaluation."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path

DOC='DEMO00000000000000000000008'
ROOT=Path('evidence/reports/model-atomic-v1-20261027-reviewed')
PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
 proc=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=180)
 (PUB/(name+'.json')).write_bytes(proc.stdout)
 if proc.stderr:(PUB/(name+'.stderr.txt')).write_bytes(proc.stderr)
 if proc.returncode:raise RuntimeError('Lark '+name+' failed; inspect bounded receipt')
 data=json.loads(proc.stdout.decode('utf-8-sig'));assert data.get('ok'),name
 return data['data']

current=call('before',['docs','+fetch','--doc',DOC,'--detail','full'])['document']
tree=ET.fromstring('<doc>'+current['content']+'</doc>')
assert '待评测' in ''.join(tree[0].itertext()),'Existing report is no longer the old placeholder'
assert len([x for x in tree.iter('source')])==1,'Preserve unexpected resource changes'
title='RoadModel 模型精度单次评测报告 v3.2.2'
old_title=''.join(tree[0].itertext())
rename=call('title-update',['docs','+update','--doc',DOC,'--command','str_replace','--pattern',old_title,'--content',title,
                           '--revision-id',str(current['revision_id'])])
assert rename['result']=='success' and not rename.get('warnings')
current=call('after-title',['docs','+fetch','--doc',DOC,'--detail','full'])['document']
tree=ET.fromstring('<doc>'+current['content']+'</doc>')
assert tree[-1].tag=='figure' and tree[-1].find('source') is not None
resource=tree[-1].find('source').get('token')
draft=ET.fromstring('<doc>'+(ROOT/'report.xml').read_text(encoding='utf-8')+'</doc>')
body=''.join(ET.tostring(e,encoding='unicode') for e in list(draft)[1:])
body+='<p>以下原待评测模板附件保留供历史追溯，不含本次实测指标。</p>'
body_file=PUB/'body.xml';body_file.write_text(body,encoding='utf-8')
updated=call('body-update',['docs','+update','--doc',DOC,'--command','block_replace',
 '--start-block-id',tree[1].get('id'),'--end-block-id',tree[-2].get('id'),
 '--content','@./'+body_file.as_posix(),'--doc-format','xml','--revision-id',str(current['revision_id'])])
assert updated['result']=='success' and not updated.get('warnings'),updated.get('warnings')
after=call('after',['docs','+fetch','--doc',DOC,'--detail','full'])['document']
actual=ET.fromstring('<doc>'+after['content']+'</doc>')
text=''.join(actual.itertext())
for expected in [title,'53.13','57.62','64.94','62.99','model-v3.2.2-smoke-20261027-run-01','OCC 体素分类','本次未评测']:
 assert expected in text,expected
tokens=[e.get('token') for e in actual.iter('source')]
assert resource in tokens and len(tokens)==3,tokens
receipt={'document_id':DOC,'url':'https://li.feishu.cn/docx/'+DOC,'revision_id':after['revision_id'],
 'run_id':'model-v3.2.2-smoke-20261027-run-01','source_report':str(ROOT/'report.json'),
 'new_attachment_count':2,'historical_attachment_preserved':resource,'new_inference_frames':0,
 'checks':['summary values read back','3 source attachments present','historical placeholder attachment preserved']}
(PUB/'publication-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
