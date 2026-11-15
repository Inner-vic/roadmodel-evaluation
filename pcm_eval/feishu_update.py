"""Update only the metric body of an existing Feishu report using bot CLI.

Revision-bound writes retain all existing attachments. Ambiguous writes stop for
reconciliation; they never restart inference or create another document.
"""
import gzip
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from .common import load_json,save_json,digest,utc_now

CLI='/mnt/data/bench-eval/service/roadmodel/lark-cli/node_modules/@larksuite/cli/bin/lark-cli'
CONFIG='/mnt/data/bench-eval/service/gateway/runtime/roadmodel/lark-cli-config'

def table_values(root):
    return [[[''.join(c.itertext()) for c in row] for row in t.findall('.//tr')] for t in root.iter('table')]


def update_report(xml_file,data_file,output,*,document_id,native_task_id,frozen_requests,scope,invoke=None):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(xml_file);data_file=Path(data_file)
    identity={'document_id':document_id,'native_task_id':native_task_id,'frozen_requests':frozen_requests,
              'scope':scope,'xml_sha256':digest(source),'data_sha256':digest(data_file)}
    journal_path=root/'journal.json'
    if journal_path.exists():
        old=load_json(journal_path)
        if old['identity']!=identity:raise ValueError('Publication input changed; preserve original journal')
        if old['state']=='VERIFIED':return load_json(root/'verification.json')
        raise ValueError('Previous publication needs reconciliation; no blind write retry')
    journal={'state':'PREPARING','identity':identity,'started_at':utc_now()}
    save_json(journal_path,journal)
    def checkpoint(state,**fields):
        journal.update(state=state,**fields);save_json(journal_path,journal,overwrite=True)
    sequence=0
    def call(args):
        nonlocal sequence
        if invoke:return invoke(args)
        sequence+=1
        env={**os.environ,'LARKSUITE_CLI_CONFIG_DIR':CONFIG,'LARKSUITE_CLI_NO_UPDATE_NOTIFIER':'1','LARKSUITE_CLI_NO_SKILLS_NOTIFIER':'1'}
        p=subprocess.run([CLI,*args,'--as','bot'],cwd=root,env=env,capture_output=True,timeout=180)
        (root/f'{sequence}.stdout.json').write_bytes(p.stdout);(root/f'{sequence}.stderr.txt').write_bytes(p.stderr)
        value=json.loads((p.stdout or p.stderr).decode('utf-8-sig'))
        if p.returncode or not value.get('ok'):raise ValueError('Bot report operation failed; inspect publication journal')
        value=value['data']
        if value.get('warnings') or value.get('result') in ('failed','partial_success'):raise ValueError('Partial Feishu operation; reconcile before proceeding')
        return value
    def read():
        d=call(['docs','+fetch','--doc',document_id,'--detail','full'])['document']
        return ET.fromstring('<root>'+d['content']+'</root>'),d['revision_id']
    try:
        expected=ET.fromstring('<root>'+source.read_text(encoding='utf8')+'</root>')
        before,revision=read();expected_tables=table_values(expected);before_tables=table_values(before)
        if len(expected_tables)!=5 or len(before_tables)!=5:raise ValueError('Reference must have exactly five metric tables')
        # Header and row-label drift requires explicit template review.
        for old,new in zip(before_tables,expected_tables):
            if old[0]!=new[0] or [r[0] for r in old[1:]]!=[r[0] for r in new[1:]]:
                raise ValueError('PCM table structure changed; preserve document for review')
        old_resources={n.get('id') for n in before.iter('source') if n.get('id')}
        old_nodes=list(before)
        start=next(i for i,n in enumerate(old_nodes) if n.tag!='title')
        end=next((i for i,n in enumerate(old_nodes[start:],start) if any(x.tag in ('source','img','whiteboard','sheet','bitable') for x in n.iter())),len(old_nodes))
        if end<=start or len(table_values(ET.fromstring('<root>'+''.join(ET.tostring(n,encoding='unicode') for n in old_nodes[start:end])+'</root>')))!=5:
            raise ValueError('Report body is interleaved with resources; do not overwrite')
        body=[n for n in expected if n.tag not in ('title','source') and not any(x.tag=='source' for x in n.iter())]
        (root/'body.xml').write_text(''.join(ET.tostring(n,encoding='unicode') for n in body),encoding='utf8')
        checkpoint('WRITING_BODY',start_revision=revision,original_source_blocks=sorted(old_resources))
        call(['docs','+update','--doc',document_id,'--command','block_replace',
              '--start-block-id',old_nodes[start].get('id'),'--end-block-id',old_nodes[end-1].get('id'),
              '--content','@./body.xml','--doc-format','xml','--revision-id',str(revision)])
        current,revision=read()
        if table_values(current)!=expected_tables:raise ValueError('Actual Feishu metric tables differ from real-result JSON')
        if not old_resources.issubset({n.get('id') for n in current.iter('source')}):raise ValueError('Original attachment disappeared')
        filename='evaluation-data-'+identity['data_sha256'][:12]+'.json.gz'
        with gzip.open(root/filename,'wb') as stream:stream.write(data_file.read_bytes())
        (root/'attachment.xml').write_text('<source path="@./'+filename+'" name="'+filename+'"/>',encoding='utf8')
        checkpoint('APPENDING_DATA',body_verified_revision=revision)
        call(['docs','+update','--doc',document_id,'--command','append','--content','@./attachment.xml','--doc-format','xml','--revision-id',str(revision)])
        after,revision=read()
        if table_values(after)!=expected_tables or not old_resources.issubset({n.get('id') for n in after.iter('source')}):
            raise ValueError('Final report readback or preserved attachments differ')
        if not any(n.get('name')==filename for n in after.iter('source')):raise ValueError('Current real-result attachment missing')
        value={'state':'VERIFIED','native_task_id':native_task_id,'frozen_requests':frozen_requests,'scope':scope,
               'five_tables_match':True,'report_url':'https://li.feishu.cn/docx/'+document_id,'document_id':document_id,
               'revision_id':revision,'source_json_sha256':identity['data_sha256'],'source_xml_sha256':identity['xml_sha256'],
               'original_sources_preserved':True,'new_inference_for_publication':0,'checked_at':utc_now()}
        if len(set(frozen_requests.values()))==1:value['frozen_request_sha256']=next(iter(frozen_requests.values()))
        save_json(root/'verification.json',value);checkpoint('VERIFIED',final_revision=revision)
        return value
    except Exception as error:
        checkpoint('NEEDS_REVIEW',error_type=type(error).__name__)
        raise
