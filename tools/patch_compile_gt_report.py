"""Update only GT table, GT notes, status and structured attachment of one report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

DOC='DEMO00000000000000000000003'
ROOT=Path('evidence/reports/compile-gt-update-20261027')
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']

def call(arguments,name):
    proc=subprocess.run(CLI+arguments+['--as','user','--format','json'],capture_output=True)
    (ROOT/(name+'.json')).write_bytes(proc.stdout)
    if proc.stderr:(ROOT/(name+'.stderr.txt')).write_bytes(proc.stderr)
    data=json.loads(proc.stdout.decode('utf-8-sig'))
    if proc.returncode or not data.get('ok'):raise RuntimeError('Lark call failed; inspect '+name)
    return data

def document(data):
    doc=data['data']['document']
    return ET.fromstring('<root>'+doc['content']+'</root>'),doc['revision_id']

def text(node):return ''.join(node.itertext())
def payload(node):
    import copy
    node=copy.deepcopy(node)
    for n in node.iter():n.attrib.pop('id',None)
    return ET.tostring(node,encoding='unicode')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    before=json.loads((ROOT/'before.json').read_text(encoding='utf-8-sig'))
    old,_=document(before)
    assert old.find('title').get('id')==DOC and 'pcm-v7-multiscene-001' in text(old.find('blockquote'))
    new=ET.fromstring('<root>'+Path('evidence/reports/compile-gt-single-20261027.xml').read_text(encoding='utf-8')+'</root>')
    expected=json.loads(Path('evidence/reports/compile-gt-single-20261027.json').read_text(encoding='utf-8'))
    assert expected['ground_truth_status']=='validated' and expected['report_status']=='complete'
    assert expected['coverage']['observed']['board_metrics']==20
    tables=new.findall('table');oldtables=old.findall('table')
    assert len(tables)==len(oldtables)==3
    assert text(tables[2])==text(oldtables[2]),'Tensor table must remain identical'
    newgt=tables[1]
    cols=ET.Element('colgroup')
    for width in [220,160,160]:ET.SubElement(cols,'col',width=str(width))
    newgt.insert(0,cols)
    explanatory=next(x for x in new.findall('p') if text(x).startswith('GT已按'))
    oldexplanatory=next(x for x in old.findall('p') if text(x).startswith('GT分类指标本次未评测'))
    oldnotes=old.find('ul');newnotes=new.find('ul')
    attachment=new.find('source');assert attachment is not None
    replacement=ET.Element('figure',{'view-type':'Preview'});replacement.append(attachment)
    oldfigure=old.find('figure');assert oldfigure is not None
    status=next(x for x in oldtables[0].iter('p') if text(x)=='部分填报（未取得的指标留空）')
    newstatus=ET.Element('p');newstatus.text='已填报（20帧小场景）'
    jobs=[('gt-table',{'block_id':oldtables[1].get('id')},newgt),
          ('gt-explanation',{'block_id':oldexplanatory.get('id')},explanatory),
          ('metric-notes',{'start_block_id':oldnotes[0].get('id'),'end_block_id':oldnotes[-1].get('id')},newnotes),
          ('structured-result',{'block_id':oldfigure.get('id')},replacement),
          ('report-status',{'block_id':status.get('id')},newstatus)]
    plan=[]
    for name,target,node in jobs:
        path=ROOT/(name+'.xml');path.write_text(payload(node),encoding='utf-8')
        plan.append({'operation':name,'command':'block_replace','target':target,'content':str(path),
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT/'patch-plan.json').write_text(json.dumps({'document_id':DOC,'operations':plan,'preserve':'title, version, run identity, sample scope, model hashes, entire tensor cosine table'},indent=2)+'\n')
    if not args.apply:
        print(json.dumps({'prepared':len(plan),'document_id':DOC,'new_gt_metric_cells':20,'report_metric_cells':expected['filled_metric_cells']}));return
    current=call(['docs','+fetch','--doc',DOC,'--detail','with-ids'],'preflight')
    now,revision=document(current)
    assert payload(now)==payload(old),'Document changed after preparation; inspect fresh contents before retry'
    for job in plan:
        command=['docs','+update','--doc',DOC,'--command','block_replace','--revision-id',str(revision),
                 '--doc-format','xml','--content','@./'+Path(job['content']).as_posix()]
        for key,value in job['target'].items():command+=['--'+key.replace('_','-'),value]
        response=call(command,job['operation']+'-response')
        if response['data'].get('result')!='success' or response['data'].get('warnings'):
            raise RuntimeError('Patch incomplete or warnings; inspect '+job['operation'])
        current=call(['docs','+fetch','--doc',DOC,'--detail','with-ids'],job['operation']+'-verified')
        now,revision=document(current)
        print(json.dumps({'updated':job['operation'],'revision_id':revision}),flush=True)
    currentxml=current['data']['document']['content']
    aftertables=now.findall('table')
    assert text(aftertables[1])==text(newgt)
    assert text(aftertables[2])==text(oldtables[2])
    assert 'GT分类指标本次未评测' not in currentxml
    assert '本次GT指标未评测' not in currentxml
    assert len(now.findall('figure'))==1 and now.find('figure/source').get('token')
    assert '已填报（20帧小场景）' in currentxml
    assert text(now.find('title'))==text(old.find('title'))
    assert text(now.find('blockquote'))==text(old.find('blockquote'))
    for paragraph in old.findall('p'):
        if '_sha256' in text(paragraph):assert text(paragraph) in currentxml
    receipt={'state':'verified','document_id':DOC,'url':'https://li.feishu.cn/docx/'+DOC,
             'revision_id':revision,'frames':20,'scenes':10,'new_inference_frames':0,
             'filled_metric_cells':expected['filled_metric_cells'],'gt_numeric_cells':20,
             'no_gt_sample_classes':['water','bumper','road_joint'],'tensor_table_unchanged':True,
             'report_json_sha256':hashlib.sha256(Path('evidence/reports/compile-gt-single-20261027.json').read_bytes()).hexdigest(),
             'attachment_token':now.find('figure/source').get('token')}
    (ROOT/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
