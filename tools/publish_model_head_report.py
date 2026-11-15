"""Update only changed sections of the existing model single report."""
import json,subprocess,hashlib,xml.etree.ElementTree as ET
from pathlib import Path

DOC='DEMO00000000000000000000008'
ROOT=Path('evidence/reports/model-heads-v2-20261027')
PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
    proc=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=180)
    (PUB/(name+'.json')).write_bytes(proc.stdout)
    if proc.stderr:(PUB/(name+'.stderr.txt')).write_bytes(proc.stderr)
    if proc.returncode:raise RuntimeError('Lark '+name+' failed; inspect receipt')
    data=json.loads(proc.stdout.decode('utf-8-sig'));assert data.get('ok'),name
    return data['data']
def fetch(name):return call(name,['docs','+fetch','--doc',DOC,'--detail','full'])['document']
def tree(doc):return ET.fromstring('<doc>'+doc['content']+'</doc>')
def ranges(t):
    elements=list(t);starts=[i for i,e in enumerate(elements) if e.tag=='h1'];out={}
    for j,i in enumerate(starts):out[''.join(elements[i].itertext())]=elements[i:(starts[j+1] if j+1<len(starts) else len(elements))]
    return out
def xml(elements):return ''.join(ET.tostring(e,encoding='unicode') for e in elements)

draft=ET.fromstring('<doc>'+(ROOT/'report.xml').read_text(encoding='utf-8')+'</doc>')
desired=ranges(draft);current=fetch('before');original=tree(current)
assert ''.join(original[0].itertext())=='RoadModel 模型精度单次评测报告 v3.2.2'
prior_tokens={e.get('token') for e in original.iter('source')};assert len(prior_tokens)==3
protected_names=['OCC 与柱状结果','指标口径与附加统计']
protected={name:xml(ranges(original)[name]) for name in protected_names}
operations=[]
for ordinal,name in enumerate(['本次评测概览','评测对象与配置','样本覆盖与输入适配','视觉分类结果','轨迹与表面任务结果','明细与复现资料']):
    actual=tree(current);section=ranges(actual)[name]
    body=desired[name][1:]
    if name=='明细与复现资料':
        # Preserve all old resource blocks physically. New attachments are inserted before them.
        targets=[section[1]]
        payload=xml(body)+'<p>以下为本次补齐视觉、轨迹任务之前的 OCC 报告及原待评测模板附件，保留供历史追溯；本次最新结果请使用上方两个附件。</p>'
    else:
        targets=section[1:]
        assert not any(e.tag in ['source','img','whiteboard'] for x in targets for e in x.iter()),'Unexpected resources in replace range'
        payload=xml(body)
    assert targets
    path=PUB/f'{ordinal:02d}-section.xml';path.write_text(payload,encoding='utf-8')
    result=call(f'{ordinal:02d}-update',['docs','+update','--doc',DOC,'--command','block_replace',
        '--start-block-id',targets[0].get('id'),'--end-block-id',targets[-1].get('id'),
        '--content','@./'+path.as_posix(),'--doc-format','xml','--revision-id',str(current['revision_id'])])
    assert result['result']=='success' and not result.get('warnings'),result
    current=fetch(f'{ordinal:02d}-after');actual=tree(current)
    updated=ranges(actual)[name];expected_text=''.join(ET.fromstring('<body>'+payload+'</body>').itertext())
    # Inline resources may contribute their own display text, so validate each textual block.
    for element in body:
        text=''.join(element.itertext()).strip()
        if text:assert text in ''.join(e for x in updated for e in x.itertext()),(name,text[:80])
    assert prior_tokens<={e.get('token') for e in actual.iter('source')}
    for protected_name,old in protected.items():assert xml(ranges(actual)[protected_name])==old,'Protected section changed'
    operations.append({'section':name,'revision_id':current['revision_id'],'result':'verified'})

after=tree(current);text=''.join(after.itertext());tokens={e.get('token') for e in after.iter('source')}
for value in ['426','53.13','57.62','64.94','62.99','50.00','0.72','1360','front-only','26-08-14-0','未覆盖']:
    assert value in text,value
assert '本次未评测' not in text
assert len(tokens-prior_tokens)==2 and prior_tokens<=tokens
assert len(list(after.iter('img')))==1
receipt={'schema':'roadmodel.model-report-publication/v1','document_id':DOC,
    'url':'https://li.feishu.cn/wiki/DEMO00000000000000000000006','revision_id':current['revision_id'],
    'source_report':str(ROOT/'report.json'),'metric_records':426,'new_inference_frames_during_generation':0,
    'head_scopes':{'OCC':2,'vision':4,'trajectory':2},'new_attachments':2,'preserved_historical_attachment_tokens':sorted(prior_tokens),
    'OCC_blocks_preserved':True,'operations':operations,'figure_count':1,
    'report_sha256':hashlib.sha256((ROOT/'report.json').read_bytes()).hexdigest()}
(PUB/'publication-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
