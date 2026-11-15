"""Scoped publication of the reviewed split/GT diagnostic to the original model doc."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('evidence/reports/model-split-gt-audit-20261027/final');PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
DOC='DEMO00000000000000000000008';TITLE='训练与验证的水泥覆盖、GT 和预处理核查'
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
    r=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=240)
    (PUB/(name+'.json')).write_bytes(r.stdout)
    if r.stderr:(PUB/(name+'.stderr.txt')).write_bytes(r.stderr)
    assert r.returncode==0,name+' failed; inspect receipt before retry'
    v=json.loads(r.stdout.decode('utf-8-sig'));assert v.get('ok'),v.get('error');return v['data']
def tree(d):
    t=ET.fromstring('<doc>'+d['content']+'</doc>');return t[0] if len(t)==1 and t[0].tag=='fragment' else t
outline=call('outline',['docs','+fetch','--doc',DOC,'--scope','outline','--max-depth','2','--detail','full'])['document']
section=next(e.get('id') for e in tree(outline).iter('h1') if ''.join(e.itertext())=='轨迹与表面任务结果')
args=['docs','+fetch','--doc',DOC,'--scope','section','--start-block-id',section,'--detail','full']
before=call('before',args)['document'];b=tree(before);assert TITLE not in ''.join(b.itertext()),'Already published; verify instead of duplicating'
resources={(e.tag,e.get('token') or e.get('src')) for e in b.iter() if e.tag in ['source','img']}
tables=[''.join(e.itertext()) for e in b.iter('table')]
old=next(e for e in b.iter('p') if ''.join(e.itertext()).startswith('具体训练成因仍待证据：'))
context=PUB/'training-context.xml';context.write_text('<p>原训练日志仍未提供训练水泥有效采样直方图、全训练/验证集分水泥 CE、完整混淆矩阵及 E44 发布选择依据。本轮已补充数据索引、少量 GT、留存 logits 重计分，以及训练水泥/沥青两帧在统一验证条件下的模型对照，详见下一节。</p>',encoding='utf-8')
v=call('context-update',['docs','+update','--doc',DOC,'--command','block_replace','--block-id',old.get('id'),'--content','@./'+context.as_posix(),'--doc-format','xml','--revision-id',str(before['revision_id'])]);assert v['result']=='success' and not v.get('warnings'),v
current=call('after-context',args)['document'];t=tree(current);assert '训练水泥/沥青两帧在统一验证条件下' in ''.join(t.itertext())
elements=list(t);front=next(i for i,e in enumerate(elements) if e.tag=='h2' and ''.join(e.itertext())=='前沿事件与边界覆盖');anchor=elements[front-1];assert anchor.tag in ['source','figure']
v=call('supplement-insert',['docs','+update','--doc',DOC,'--command','block_insert_after','--block-id',anchor.get('id'),'--content','@./'+(ROOT/'report.xml').as_posix(),'--doc-format','xml','--revision-id',str(current['revision_id'])]);assert v['result']=='success' and not v.get('warnings'),v
after=call('after',args)['document'];a=tree(after);text=''.join(a.itertext())
for e in ET.fromstring('<body>'+(ROOT/'report.xml').read_text(encoding='utf-8')+'</body>'):
    if e.tag not in ['source','img']:assert ''.join(e.itertext()) in text,''.join(e.itertext())[:100]
assert all(x in text for x in tables)
assert resources<={(e.tag,e.get('token') or e.get('src')) for e in a.iter() if e.tag in ['source','img']}
assert len(list(a.iter('source')))==len(list(b.iter('source')))+2
assert len(list(a.iter('img')))==len(list(b.iter('img')))+2
heading=next(e for e in a.iter('h2') if ''.join(e.itertext())==TITLE)
receipt={'schema':'roadmodel.split-gt-publication/v1','document_id':DOC,'revision_id':after['revision_id'],'url':'https://li.feishu.cn/wiki/DEMO00000000000000000000006#'+heading.get('id'),'source_new_forward_frames':2,'new_training_runs':0,'new_inference_during_generation_publication':0,'old_metrics_and_resources_preserved':True,'attachments_added':2,'images_added':2,'result':'verified'}
(PUB/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
