"""Publish an already generated report into its existing native document.

This adapter does not evaluate models or generate report content. Every edit is
revision-bound and followed by a fresh read; title/identity header stay intact.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from_path=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(from_path))
from pcm_eval.common import digest

CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']

def main():
    p=argparse.ArgumentParser();p.add_argument('--doc',required=True);p.add_argument('--sections',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--apply',action='store_true');args=p.parse_args()
    root=args.evidence_dir;root.mkdir(parents=True,exist_ok=True)
    index=json.loads(args.sections.read_text(encoding='utf-8'));report=json.loads(args.report.read_text(encoding='utf-8'))
    assert report['schema']=='roadmodel.compile-single-report/v3'
    assert index['report_sha256']==digest(args.report)
    sections=index['sections']
    for s in sections:assert digest(Path(s['path']))==s['sha256']
    def call(arguments,name):
        proc=subprocess.run(CLI+arguments+['--as','user','--format','json'],capture_output=True)
        (root/(name+'.json')).write_bytes(proc.stdout)
        if proc.stderr:(root/(name+'.stderr.txt')).write_bytes(proc.stderr)
        data=json.loads((proc.stdout or proc.stderr).decode('utf-8-sig'))
        if proc.returncode or not data.get('ok'):raise RuntimeError('Lark call failed; inspect '+name)
        return data
    def parse(data):
        document=data['data']['document']
        return ET.fromstring('<root>'+document['content']+'</root>'),document['revision_id']
    def txt(node):return ''.join(node.itertext())
    def layout(node):
        # Ignore formatting IDs but compare all written text and table values.
        return ''.join(node.itertext())
    def chapter(nodes,title):
        starts=[i for i,n in enumerate(nodes) if n.tag=='h1' and txt(n)==title]
        if len(starts)!=1:raise ValueError('Cannot locate one chapter: '+title)
        start=starts[0];end=next((i for i in range(start+1,len(nodes)) if nodes[i].tag=='h1'),len(nodes))-1
        return nodes[start].get('id'),nodes[end].get('id')
    # Existing snapshot makes the complete patch reviewable before mutation.
    before_path=root/'before.json'
    if not before_path.is_file():
        call(['docs','+fetch','--doc',args.doc,'--detail','with-ids'],'before')
    before=json.loads(before_path.read_text(encoding='utf-8-sig'));old,_=parse(before)
    if old.find('title').get('id')!=args.doc:raise ValueError('Unexpected document identity')
    header=old.find('blockquote')
    if header is None or report['identity']['run_id'] not in txt(header):raise ValueError('Report/run does not match this document')
    titles=[txt(n) for n in old if n.tag=='h1']
    content={s['id']:Path(s['path']).read_text(encoding='utf-8') for s in sections}
    jobs=[]
    if titles==['评测信息','对齐部分（fp16）','统计说明与评测产物']:
        for name,title,keys in [
            ('configuration-coverage','评测信息',['configuration','coverage']),
            ('tensor-gt-findings','对齐部分（fp16）',['tensor','gt','findings']),
            ('definitions-artifacts','统计说明与评测产物',['definitions','artifacts'])]:
            start,end=chapter(list(old),title)
            jobs.append({'name':name,'command':'block_replace','start_block_id':start,'end_block_id':end,
                         'keys':keys,'content':'\n'.join(content[k] for k in keys)})
        jobs.append({'name':'overview','command':'block_insert_after','block_id':header.get('id'),
                     'keys':['overview'],'content':content['overview']})
    elif titles==[s['title'] for s in sections]:
        for s in sections:
            start,end=chapter(list(old),s['title'])
            jobs.append({'name':s['id'],'command':'block_replace','start_block_id':start,'end_block_id':end,
                         'keys':[s['id']],'content':content[s['id']]})
    else:raise ValueError('Document structure changed; inspect before preparing a patch')
    plan=[]
    for job in jobs:
        path=root/(job['name']+'.xml');path.write_text(job.pop('content'),encoding='utf-8')
        if path.stat().st_size>90000:
            raise ValueError('Split this publication patch into smaller sections: '+job['name'])
        plan.append(dict(job,path=str(path),sha256=digest(path)))
    (root/'patch-plan.json').write_text(json.dumps({'document_id':args.doc,'run_id':report['identity']['run_id'],
        'report_sha256':digest(args.report),'preserve':['document ID','title','version/run identity header'],
        'operations':plan},indent=2)+'\n')
    if not args.apply:
        print(json.dumps({'state':'prepared','operations':len(plan),'sections':len(sections),'document_id':args.doc}));return
    current=call(['docs','+fetch','--doc',args.doc,'--detail','with-ids'],'preflight')
    now,revision=parse(current)
    if ET.tostring(now)!=ET.tostring(old):raise ValueError('Document changed after snapshot; inspect new revision before retry')
    for job in plan:
        command=['docs','+update','--doc',args.doc,'--command',job['command'],'--revision-id',str(revision),
                 '--doc-format','xml','--content','@./'+Path(job['path']).as_posix()]
        for key in ['block_id','start_block_id','end_block_id']:
            if key in job:command+=['--'+key.replace('_','-'),job[key]]
        response=call(command,job['name']+'-response')
        if response['data'].get('result')!='success' or response['data'].get('warnings'):
            raise RuntimeError('Patch incomplete or warnings: '+job['name'])
        current=call(['docs','+fetch','--doc',args.doc,'--detail','with-ids'],job['name']+'-verified')
        now,revision=parse(current)
        nodes=list(now)
        for key in job['keys']:
            s=next(s for s in sections if s['id']==key)
            start,end=chapter(nodes,s['title']);a=next(i for i,n in enumerate(nodes) if n.get('id')==start);b=next(i for i,n in enumerate(nodes) if n.get('id')==end)
            expected=ET.fromstring('<root>'+content[key]+'</root>')
            actual=ET.Element('root');actual.extend(deepcopy(nodes[a:b+1]))
            # Attachment source nodes have no human text; tokens checked below.
            if ''.join(actual.itertext()).replace('\n','')!=''.join(expected.itertext()).replace('\n',''):
                raise RuntimeError('Written chapter text differs: '+key)
        print(json.dumps({'updated':job['name'],'revision_id':revision}),flush=True)
    assert txt(now.find('title'))==txt(old.find('title'))
    assert txt(now.find('blockquote'))==txt(header)
    assert [txt(n) for n in now if n.tag=='h1']==[s['title'] for s in sections]
    sources=list(now.iter('source'))
    assert len(sources)==2 and all(s.get('token') for s in sources)
    mapping=[]
    for s in sections:
        start,end=chapter(list(now),s['title'])
        mapping.append({'section_id':s['id'],'title':s['title'],'start_block_id':start,'end_block_id':end})
    receipt={'state':'verified','document_id':args.doc,'url':'https://li.feishu.cn/docx/'+args.doc,
             'revision_id':revision,'report_schema':report['schema'],'report_sha256':digest(args.report),
             'run_id':report['identity']['run_id'],'frames':report['coverage']['actual_frames'],
             'new_inference_frames':0,'section_blocks':mapping,
             'attachments':[dict(s.attrib) for s in sources]}
    (root/'publication-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'verified','document_id':args.doc,'revision_id':revision,'sections':len(mapping),'attachments':len(sources)}))

if __name__=='__main__':main()
