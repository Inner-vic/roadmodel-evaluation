"""Publish only the completed diagnostic supplement into the original model report."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('evidence/reports/model-trajectory-diagnostic-20261027');PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
DOC='DEMO00000000000000000000008'
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
    run=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=180)
    (PUB/(name+'.json')).write_bytes(run.stdout)
    if run.stderr:(PUB/(name+'.stderr.txt')).write_bytes(run.stderr)
    if run.returncode:raise RuntimeError(name+' failed; inspect saved receipt')
    value=json.loads(run.stdout.decode('utf-8-sig'));assert value.get('ok'),value.get('error')
    return value['data']
def tree(d):
    t=ET.fromstring('<doc>'+d['content']+'</doc>')
    return t[0] if len(t)==1 and t[0].tag=='fragment' else t
outline=call('outline',['docs','+fetch','--doc',DOC,'--scope','outline','--max-depth','2','--detail','with-ids'])['document']
headings=[e for e in tree(outline).iter('h1') if ''.join(e.itertext())=='轨迹与表面任务结果'];assert len(headings)==1
section=headings[0].get('id')
args=['docs','+fetch','--doc',DOC,'--scope','section','--start-block-id',section,'--detail','full']
before=call('before',args)['document'];b=tree(before);elements=list(b)
old_resources={(e.tag,e.get('token') or e.get('src')) for e in b.iter() if e.tag in ['img','source']}
title='同两帧 logits 与中间特征诊断'
assert title not in ''.join(b.itertext()),'Diagnostic supplement already exists'
indices=[i for i,e in enumerate(elements) if e.tag=='p' and ''.join(e.itertext()).startswith('证据边界：本次原始文件没有保存')]
assert len(indices)==1
i=indices[0];assert elements[i+1].tag=='p' and '本次核查只读原始结果' in ''.join(elements[i+1].itertext())
history='<p>首轮离线核查确定了标签、GT 与权重配置接入一致，当时未留存 logits 和中间特征；本轮已按相同两帧补齐，见下方诊断。首轮离线核查新增推理 0 帧，本轮 FP32 与 FP16 对照的执行次数单独记录。</p><p>以下为首轮离线核查证据附件，保留供追溯。</p>'
file=PUB/'history-context.xml';file.write_text(history,encoding='utf-8')
changed=call('history-context-update',['docs','+update','--doc',DOC,'--command','block_replace',
    '--start-block-id',elements[i].get('id'),'--end-block-id',elements[i+1].get('id'),
    '--content','@./'+file.as_posix(),'--doc-format','xml','--revision-id',str(before['revision_id'])])
assert changed['result']=='success' and not changed.get('warnings'),changed
current=call('after-history-context',args)['document'];t=tree(current);elements=list(t)
assert '本轮已按相同两帧补齐' in ''.join(t.itertext())
front=[i for i,e in enumerate(elements) if e.tag=='h2' and ''.join(e.itertext())=='前沿事件与边界覆盖'];assert len(front)==1
anchor=elements[front[0]-1];assert anchor.tag in ['figure','source']
inserted=call('diagnostic-insert',['docs','+update','--doc',DOC,'--command','block_insert_after','--block-id',anchor.get('id'),
    '--content','@./'+(ROOT/'report.xml').as_posix(),'--doc-format','xml','--revision-id',str(current['revision_id'])])
assert inserted['result']=='success' and not inserted.get('warnings'),inserted
after=call('after',args)['document'];a=tree(after);text=''.join(a.itertext())
for e in ET.fromstring('<body>'+(ROOT/'report.xml').read_text(encoding='utf-8')+'</body>'):
    expected=''.join(e.itertext())
    if expected:assert expected in text,expected[:120]
assert old_resources<={(e.tag,e.get('token') or e.get('src')) for e in a.iter() if e.tag in ['img','source']}
assert len(list(a.iter('source')))==len(list(b.iter('source')))+2
assert '证据边界：本次原始文件没有保存' not in text
heading=next(e for e in a.iter('h2') if ''.join(e.itertext())==title)
receipt={'schema':'roadmodel.trajectory-diagnostic-publication/v1','document_id':DOC,'revision_id':after['revision_id'],
    'url':'https://li.feishu.cn/wiki/DEMO00000000000000000000006#'+heading.get('id'),
    'unique_frames':2,'new_FP32_frame_passes':2,'new_FP16_control_frame_passes':2,
    'new_frames_during_generation_or_publication':0,'new_attachments':2,'prior_resources_preserved':True,
    'model_accuracy_metric_changed':False,'result':'verified'}
(PUB/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
