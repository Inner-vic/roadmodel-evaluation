"""Publish the independent audit supplement into the existing model report."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('evidence/reports/model-training-semantic-audit-20261027');PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
DOC='DEMO00000000000000000000008';SECTION='doxcnM02tOlTMlkOqbm3RQPOtzf'
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
    r=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=180)
    (PUB/(name+'.json')).write_bytes(r.stdout)
    if r.stderr:(PUB/(name+'.stderr.txt')).write_bytes(r.stderr)
    assert r.returncode==0,name+' failed; inspect receipt before retry'
    d=json.loads(r.stdout.decode('utf-8-sig'));assert d.get('ok'),d.get('error');return d['data']
def tree(d):
    t=ET.fromstring('<doc>'+d['content']+'</doc>');return t[0] if len(t)==1 and t[0].tag=='fragment' else t
args=['docs','+fetch','--doc',DOC,'--scope','section','--start-block-id',SECTION,'--detail','full']
before=call('before',args)['document'];b=tree(before)
resources={(e.tag,e.get('token') or e.get('src')) for e in b.iter() if e.tag in ['img','source']}
tables=[''.join(e.itertext()) for e in b.iter('table')]
title='原训练任务的语义表现核查'
assert title not in ''.join(b.itertext()),'Already published: verify existing receipt instead'
old=next(e for e in b.iter('p') if ''.join(e.itertext()).startswith('尚未证明造成该权重行为的训练原因。'))
context=PUB/'diagnostic-context.xml';context.write_text('<p>前一轮补跑已定位到这两帧的条件融合与语义决策。本轮继续复用原训练日志、配置和 TensorBoard：已证实水泥识别弱项在历史验证中持续存在，见下节“原训练任务的语义表现核查”。训练样本比例、分水泥损失及具体形成机制仍缺证据，不能仅凭两帧或总体训练准确率断言训练塌缩。</p>',encoding='utf-8')
change=call('context-update',['docs','+update','--doc',DOC,'--command','block_replace','--block-id',old.get('id'),'--content','@./'+context.as_posix(),'--doc-format','xml','--revision-id',str(before['revision_id'])])
assert change['result']=='success' and not change.get('warnings'),change
current=call('after-context',args)['document'];t=tree(current)
assert '本轮继续复用原训练日志' in ''.join(t.itertext())
elements=list(t);front=next(i for i,e in enumerate(elements) if e.tag=='h2' and ''.join(e.itertext())=='前沿事件与边界覆盖')
anchor=elements[front-1];assert anchor.tag in ['source','figure']
change=call('supplement-insert',['docs','+update','--doc',DOC,'--command','block_insert_after','--block-id',anchor.get('id'),'--content','@./'+(ROOT/'report.xml').as_posix(),'--doc-format','xml','--revision-id',str(current['revision_id'])])
assert change['result']=='success' and not change.get('warnings'),change
after=call('after',args)['document'];a=tree(after);text=''.join(a.itertext())
expected=ET.fromstring('<body>'+(ROOT/'report.xml').read_text(encoding='utf-8')+'</body>')
for e in expected:
    if e.tag!='source':assert ''.join(e.itertext()) in text,''.join(e.itertext())[:160]
assert resources<={(e.tag,e.get('token') or e.get('src')) for e in a.iter() if e.tag in ['img','source']}
assert all(x in text for x in tables)
assert len(list(a.iter('source')))==len(list(b.iter('source')))+2
heading=next(e for e in a.iter('h2') if ''.join(e.itertext())==title)
receipt={'schema':'roadmodel.training-semantic-publication/v1','document_id':DOC,'revision_id':after['revision_id'],'url':'https://li.feishu.cn/wiki/DEMO00000000000000000000006#'+heading.get('id'),'new_training_runs':0,'new_inference_frames':0,'original_metrics_and_resources_preserved':True,'attachments_added':2,'result':'verified'}
(PUB/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
