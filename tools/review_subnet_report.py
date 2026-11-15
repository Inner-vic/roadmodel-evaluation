"""Refine the same v3.2.2 report from retained data, preserving its first attachments."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,digest
from pcm_eval.runner import collect_remote
from pcm_eval.compile_report import write_compile_report
from pcm_eval.publication import cli_command,portable_import
from prepare_subnet_remote import remote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/compile-subnet-review-20261029'
RUN='pcm-v322-subnets-one-02'
ADAPTER='/mnt/data/developer/pcm-roadmodel-eval/subnet-adapter-20261029'
DOC='HxMTdf4nzo1jwtxoqcwcFzNCnXe'


def txt(node):return ''.join(node.itertext())


if __name__=='__main__':
    src=ROOT/'tools/subnet_chain_remote.py'
    subprocess.run(['scp.exe','-q','-o','BatchMode=yes',src.name,'ubuntu-workstation:'+ADAPTER+'/subnet_chain_remote.py'],cwd=src.parent,check=True,timeout=40)
    code=r'''import pathlib,sys,json
sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/subnet-adapter-20261029')
import subnet_chain_remote as adapter
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/pcm-v322-subnets-one-02')
record=json.loads((root/'execution.json').read_text());assert record['state']=='SUCCEEDED' and record['release']['state']=='RELEASED'
adapter.write_compatible_results(root,record)
print(json.dumps({'state':'metadata_collected','inference_frames':0}))'''
    print(remote(code))
    request=load_json(ROOT/'examples/compile_v322_subnets_small.json')
    request['execution']['driver_sha256']=digest(src)
    save_json(ROOT/'examples/compile_v322_subnets_small.json',request,overwrite=True)
    target=OUT/RUN/'reviewed-delivery'
    result=collect_remote('ubuntu-workstation','/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/'+RUN,
                          'compile_alignment',target/'collected',version_name='v3.2.2',expected_frames=1)
    frozen=load_json(OUT/'retained-executed-code.json');reported=deepcopy(request)
    reported['execution']['driver_sha256']=frozen['executed-adapter.py']
    reported['execution']['contract_sha256']=frozen['executed-contract.py']
    reported['source']['launch_mode']='retained explicit --run-id/--board CLI run; report request reconstructed only for artifact identity'
    reported['source']['executed_argv']=[reported['execution']['python'],ADAPTER+'/subnet_chain_remote.py','--run-id',RUN,'--board']
    save_json(target/'reported-request.json',reported)
    report=write_compile_report(target/'report/report.xml',result_file=target/'collected/result.json',request_file=target/'reported-request.json')
    pub=target/'publication';pub.mkdir()
    sequence=0
    def call(args):
        global sequence
        sequence+=1
        p=subprocess.run([*cli_command(),*args,'--as','user'],cwd=pub,capture_output=True,timeout=150)
        (pub/f'{sequence}.json').write_bytes(p.stdout or p.stderr)
        value=json.loads(p.stdout or p.stderr)
        if p.returncode or not value.get('ok'):raise RuntimeError('Lark operation failed; inspect receipt')
        return value['data']
    def read():
        value=call(['docs','+fetch','--doc',DOC,'--detail','full'])['document']
        return ET.fromstring('<root>'+value['content']+'</root>'),value['revision_id']
    before,revision=read()
    assert RUN in txt(before)
    original_tokens={n.get('token') for n in before.iter('source')}
    draft=ET.fromstring('<root>'+Path(report['xml']).read_text(encoding='utf-8')+'</root>')
    def chapter(root,title):
        nodes=list(root);start=next(i for i,n in enumerate(nodes) if n.tag=='h1' and txt(n)==title)
        end=next((i for i in range(start+1,len(nodes)) if nodes[i].tag=='h1'),len(nodes))
        return nodes[start:end]
    for title in ('评测对象与配置','张量一致性'):
        current,revision=read();old=chapter(current,title)
        fragment=pub/f'chapter-{sequence}.xml';fragment.write_text(''.join(ET.tostring(n,encoding='unicode') for n in chapter(draft,title)),encoding='utf-8')
        changed=call(['docs','+update','--doc',DOC,'--command','block_replace','--start-block-id',old[0].get('id'),
            '--end-block-id',old[-1].get('id'),'--content','@./'+fragment.name,'--doc-format','xml','--revision-id',str(revision)])
        assert changed['result']=='success' and not changed.get('warnings')
    appendix=ET.Element('root')
    note=ET.SubElement(appendix,'p');note.text='同一帧报告补全：以下补全版附件增加五分网产物身份、实际shape及整数一致率；保留首次附件供追溯，推理未重跑。'
    for node in chapter(draft,'明细与复现资料')[1:]:appendix.append(node)
    portable=portable_import(appendix,Path(report['xml']),pub)
    current,revision=read()
    changed=call(['docs','+update','--doc',DOC,'--command','append','--content','@./'+portable.name,'--doc-format','xml','--revision-id',str(revision)])
    assert changed['result']=='success' and not changed.get('warnings')
    after,revision=read();text=txt(after)
    for value in ('五份原始SubNetOps','52.625','90.625','0.983809'):
        assert value in text,value
    tokens={n.get('token') for n in after.iter('source')}
    assert original_tokens.issubset(tokens) and len(tokens)>=len(original_tokens)+2
    receipt={'state':'SUCCEEDED','document_id':DOC,'report_url':'https://li.feishu.cn/docx/'+DOC,'revision_id':revision,
        'report':report,'new_inference_frames':0,'original_attachment_tokens_preserved':sorted(original_tokens),
        'same_frame':RUN,'gt_metrics':'not_computed','exact_integer_ratio_displayed':True}
    save_json(OUT/'reviewed-publication.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
