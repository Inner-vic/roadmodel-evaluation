"""Revision-bound update of business tables, preserving diagnostics/resources."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcm_eval.common import digest


def text(node):return ''.join(node.itertext())


def parse(response):
    doc=response['data']['document']
    return ET.fromstring('<root>'+doc['content']+'</root>'),doc['revision_id']


def chapter(root,title):
    nodes=list(root)
    starts=[i for i,n in enumerate(nodes) if n.tag=='h1' and text(n)==title]
    if len(starts)!=1:raise ValueError('Cannot uniquely locate chapter: '+title)
    start=starts[0];end=next((i for i in range(start+1,len(nodes)) if nodes[i].tag=='h1'),len(nodes))
    return nodes[start:end]


def resource_tokens(root):
    return {x.get('token') for x in root.iter() if x.tag in ('source','img') and x.get('token')}


def headers(table):return [text(x) for x in table.findall('./thead/tr/th')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['model_accuracy','compile_alignment'],required=True)
    p.add_argument('--doc',required=True);p.add_argument('--report-xml',type=Path,required=True)
    p.add_argument('--report-json',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=json.loads(args.report_json.read_text(encoding='utf-8'))
    if report.get('evaluation_type')!=args.kind or report['generation'].get('layout')!='business':
        raise ValueError('Matching business report required')
    draft=ET.fromstring('<root>'+args.report_xml.read_text(encoding='utf-8')+'</root>')
    def call(name,argv):
        proc=subprocess.run([shutil.which('lark-cli') or 'lark-cli',*argv,'--as','user'],capture_output=True,timeout=150)
        (args.output/(name+'.json')).write_bytes(proc.stdout)
        if proc.stderr:(args.output/(name+'.stderr.txt')).write_bytes(proc.stderr)
        data=json.loads(proc.stdout or proc.stderr)
        if proc.returncode or not data.get('ok'):raise RuntimeError('Lark call failed: '+name)
        return data
    before_response=call('before',['docs','+fetch','--doc',args.doc,'--detail','full'])
    before,revision=parse(before_response)
    expected_run=report.get('run_id') or report.get('identity',{}).get('run_id')
    if not expected_run or expected_run not in text(before):raise ValueError('Existing report belongs to another evaluation run')
    old_tokens=resource_tokens(before)
    diagnostics={n.get('id'):text(n) for n in before if n.tag=='h2' and text(n) in [
        '轨迹语义 0% 核查','同两帧 logits 与中间特征诊断','原训练任务的语义表现核查','训练与验证的水泥覆盖、GT 和预处理核查']}
    targets=[]
    if args.kind=='model_accuracy':
        targets.append({'kind':'chapter','title':'OCC 与柱状结果','nodes':chapter(draft,'OCC 与柱状结果')})
        new_vision=chapter(draft,'视觉分类结果')
        new_tables=[n for n in new_vision if n.tag=='table']
        core=next(n for n in new_tables if '参与子类数' in headers(n))
        detail=next(n for n in new_tables if '计入 mAP' in headers(n))
        targets.extend([{'kind':'vision-core','nodes':[core]},{'kind':'vision-detail','nodes':[detail]}])
    else:
        for title in ['张量一致性','相对GT的表现']:
            targets.append({'kind':'chapter','title':title,'nodes':chapter(draft,title)})
    plan=[]
    for i,target in enumerate(targets):
        payload=''.join(ET.tostring(n,encoding='unicode') for n in target['nodes'])
        path=args.output/f'patch-{i:02d}.xml';path.write_text(payload,encoding='utf-8')
        plan.append({'index':i,'kind':target['kind'],'title':target.get('title'),'path':str(path),'sha256':digest(path)})
    save={'document_id':args.doc,'run_id':expected_run,'initial_revision':revision,'operations':plan,
          'preserve_diagnostics':diagnostics,'preserve_attachment_tokens':sorted(old_tokens)}
    (args.output/'patch-plan.json').write_text(json.dumps(save,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not args.apply:
        print(json.dumps({'state':'prepared','operations':len(plan),'document_id':args.doc}));return
    current=before
    for job in plan:
        if job['kind']=='chapter':
            nodes=chapter(current,job['title'])
            if any(n.tag in ('source','img','figure') or list(n.iter('source')) or list(n.iter('img')) for n in nodes):
                raise ValueError('Unexpected resource in a replaceable business-table chapter')
            anchors=['--start-block-id',nodes[0].get('id'),'--end-block-id',nodes[-1].get('id')]
        else:
            tables=[n for n in chapter(current,'视觉分类结果') if n.tag=='table']
            if job['kind']=='vision-core':
                matches=[n for n in tables if any('Accuracy' in h for h in headers(n)) or '参与子类数' in headers(n)]
            else:
                matches=[n for n in tables if 'GT 类别 ID' in headers(n) or '计入 mAP' in headers(n)]
            if len(matches)!=1:raise ValueError('Ambiguous vision business table')
            anchors=['--block-id',matches[0].get('id')]
        response=call(f'patch-{job["index"]:02d}-response',['docs','+update','--doc',args.doc,'--command','block_replace',
            *anchors,'--revision-id',str(revision),'--doc-format','xml','--content','@./'+Path(job['path']).as_posix()])
        if response['data'].get('result')!='success' or response['data'].get('warnings'):
            raise RuntimeError('Document patch was partial or warned')
        current,revision=parse(call(f'patch-{job["index"]:02d}-verified',['docs','+fetch','--doc',args.doc,'--detail','full']))
        expected=ET.fromstring('<root>'+Path(job['path']).read_text(encoding='utf-8')+'</root>')
        expected_text=''.join(expected.itertext()).replace('\n','')
        if job['kind']=='chapter':
            actual_text=''.join(text(n) for n in chapter(current,job['title'])).replace('\n','')
            if expected_text!=actual_text:raise RuntimeError('Published chapter differs')
        else:
            if expected_text not in text(current).replace('\n',''):raise RuntimeError('Published business table differs')
        print(json.dumps({'updated':job['kind'],'revision_id':revision}),flush=True)
    footer=chapter(current,'明细与复现资料')[0]
    attachment_path=args.output/'attachments.xml'
    marker='业务单次模板交付：'+digest(args.report_json)
    attachment_path.write_text('<p>'+marker+'</p><source path="@./'+args.report_json.as_posix()+'" name="业务单次报告.json"/>'+
        '<source path="@./'+args.bundle.as_posix()+'" name="业务单次报告证据.zip"/>',encoding='utf-8')
    if marker not in text(current):
        response=call('attachments-response',['docs','+update','--doc',args.doc,'--command','block_insert_after',
            '--block-id',footer.get('id'),'--revision-id',str(revision),'--content','@./'+attachment_path.as_posix()])
        if response['data'].get('result')!='success' or response['data'].get('warnings'):raise RuntimeError('Attachment update incomplete')
        current,revision=parse(call('final-verified',['docs','+fetch','--doc',args.doc,'--detail','full']))
    if not old_tokens.issubset(resource_tokens(current)):raise RuntimeError('A historical resource was lost')
    live_ids={n.get('id'):text(n) for n in current if n.tag=='h2'}
    if any(live_ids.get(key)!=value for key,value in diagnostics.items()):raise RuntimeError('A diagnostic heading changed')
    receipt={'state':'verified','document_id':args.doc,'revision_id':revision,'run_id':expected_run,
             'report_sha256':digest(args.report_json),'preserved_resources':len(old_tokens),
             'preserved_diagnostic_sections':len(diagnostics),'new_inference_frames':0,
             'url':'https://li.feishu.cn/docx/'+args.doc}
    (args.output/'publication-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
