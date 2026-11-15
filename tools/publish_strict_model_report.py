"""Replace the authorized report body with the original model template only."""
from pathlib import Path

from connect_active_entry import call


CODE = r'''import pathlib,json,hashlib,sys,os,subprocess,datetime,shutil,xml.etree.ElementTree as ET
base=pathlib.Path('/mnt/data/bench-eval/roadmodel')
out=base/'publication/model-strict-20261102-r2'
run=base/'formal/model-timing-v322-20261102-r1'
doc='DEMO00000000000000000000002'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
load=lambda path:json.loads(path.read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
if (out/'verification.json').exists():
 print(json.dumps(load(out/'verification.json'),ensure_ascii=False));sys.exit(0)
assert not out.exists(),'Reconcile previous strict-template attempt; do not repeat writes'
prior=base/'publication/model-full-20261102-r1'
verified=load(prior/'verification.json');data=load(prior/'model-data.json');proof=load(run/'report/verification.json')
assert verified['state']=='VERIFIED' and sha(prior/'model-data.json')==verified['published_data_sha256']
assert sha(run/'report/model-report.json')==verified['model_json_sha256']==proof['report_json_sha256']
assert all(sha(pathlib.Path(path))==h for path,h in proof['original_result_hashes'].items())
reference=pathlib.Path(data['template']['path']);assert sha(reference)==data['template']['sha256']
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/model-template-strict-20261102-r1')
original=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
if not release.exists():
 release.mkdir();shutil.copytree(original/'pcm_eval',release/'pcm_eval');shutil.copytree(original/'configs',release/'configs')
target=release/'pcm_eval/strict_model_template.py'
if target.exists():assert target.read_text()==MODULE,'Private renderer differs'
else:target.write_text(MODULE,encoding='utf8')
sys.path.insert(0,str(release))
from pcm_eval.strict_model_template import render
from pcm_eval.reference_report import display
ref=load(reference)['data']['document'];tables=data['tables']
body=render(ref['content'],tables,model_version=data['model']['version_name'],vision_samples=data['vision_frames'])
tree=ET.fromstring('<root>'+body+'</root>')
text=lambda n:''.join(n.itertext())
def values(root):return [[[text(c) for c in row] for row in t.iter('tr')] for t in root.iter('table')]
expected=[[t['headers']]+[[display(c) for c in row] for row in t['rows']] for t in tables]
assert values(tree)==expected and len(expected)==3
assert all(s not in body for s in ['model-timing-','完整验证集','数据集26-','编译精度评测','本次模型结果完成','报告更新：'])
assert [text(n).strip() for n in tree if n.tag in ('h1','h2')]==['开发空间模型评测','occ头','轨迹头','视觉头']
out.mkdir(parents=True)
(out/'body.xml').write_text(body,encoding='utf8')
journal={'state':'PREPARING','document_id':doc,'started_at':now(),'template_sha256':sha(reference),
         'renderer_sha256':sha(target),'model_json_sha256':verified['model_json_sha256'],'new_inference_frames':0}
def checkpoint(state,**fields):
 journal.update(state=state,**fields);tmp=out/'journal.tmp';tmp.write_text(json.dumps(journal,ensure_ascii=False,indent=2));tmp.replace(out/'journal.json')
checkpoint('PREPARING')
seq=0
def cli(args):
 global seq
 seq+=1
 env={**os.environ,'LARKSUITE_CLI_CONFIG_DIR':'/mnt/data/bench-eval/service/gateway/runtime/roadmodel/lark-cli-config',
      'LARKSUITE_CLI_NO_UPDATE_NOTIFIER':'1','LARKSUITE_CLI_NO_SKILLS_NOTIFIER':'1'}
 p=subprocess.run(['/mnt/data/bench-eval/service/roadmodel/lark-cli/node_modules/@larksuite/cli/bin/lark-cli',*args,'--as','bot'],
                  cwd=out,env=env,capture_output=True,timeout=180)
 (out/('%02d.stdout.json'%seq)).write_bytes(p.stdout);(out/('%02d.stderr.txt'%seq)).write_bytes(p.stderr)
 result=json.loads((p.stdout or p.stderr).decode('utf-8-sig'))
 assert p.returncode==0 and result.get('ok'),'CLI operation failed; inspect journal'
 value=result['data'];assert not value.get('warnings') and value.get('result') not in ('failed','partial_success'),'Partial write; reconcile'
 return value
def read():
 d=cli(['docs','+fetch','--doc',doc,'--detail','full'])['document']
 return d,ET.fromstring('<root>'+d['content']+'</root>')
def structure(root):
 # Tables are compared cell by cell separately. The server normalizes table
 # wrappers and list-item paragraphs; compare original block types/text/links.
 return [(n.tag,text(n).strip() if n.tag!='table' else '[table]',
          tuple(a.get('href') for a in n.iter('a'))) for n in root if n.tag!='title']
try:
 before,old=read();assert before['revision_id']==verified['revision_id'],'Document changed since verified model publication'
 assert values(old)[:3]==expected and len(values(old))==5
 # Keep all prior body/resource references on the workstation and in Feishu history.
 # Replacing this report's body removes old compile/diagnostic/attachment blocks,
 # without deleting their underlying files or any evaluation outputs.
 (out/'before-document.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
 nodes=[n for n in old if n.tag!='title'];assert nodes and all(n.get('id') for n in nodes)
 checkpoint('WRITING_TEMPLATE_BODY',start_revision=before['revision_id'],
            archived_resource_tokens=[dict(n.attrib) for n in old.iter('source')])
 cli(['docs','+update','--doc',doc,'--command','block_replace','--start-block-id',nodes[0].get('id'),
      '--end-block-id',nodes[-1].get('id'),'--content','@./body.xml','--doc-format','xml','--revision-id',str(before['revision_id'])])
 current,root=read();assert values(root)==expected,'Model table readback differs'
 assert structure(root)==structure(tree),'Non-table content differs from original model template'
 source_title=next(text(n) for n in ET.fromstring('<root>'+ref['content']+'</root>') if n.tag=='title')
 title=source_title.replace('发版报告v3.1.0','单次报告')
 checkpoint('RENAMING_MODEL_REPORT',body_verified_revision=current['revision_id'])
 renamed=cli(['drive','+update-title','--url','https://li.feishu.cn/docx/'+doc,'--title',title])
 assert renamed.get('updated') is True and renamed.get('title')==title
 after,root=read();assert values(root)==expected and structure(root)==structure(tree)
 assert next(text(n) for n in root if n.tag=='title')==title
 assert not list(root.iter('source')) and not list(root.iter('figure'))
 assert all(sha(pathlib.Path(path))==h for path,h in proof['original_result_hashes'].items())
 receipt={'state':'VERIFIED','report_url':'https://li.feishu.cn/docx/'+doc,'title':title,'revision_id':after['revision_id'],
          'checked_at':now(),'model_table_count':3,'all_model_table_cells_match':True,'non_table_body_matches_original_template':True,
          'only_original_model_headings':True,'cross_version_note_removed':True,'visual_sample_total':data['vision_frames'],
          'dataset_and_run_id_paragraphs_removed':True,'compile_body_removed':True,'non_template_attachment_blocks_removed':True,
          'previous_body_and_resource_references_archived':str(out/'before-document.json'),
          'native_results_unchanged':True,'new_inference_frames':0,'body_xml_sha256':sha(out/'body.xml')}
 (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));checkpoint('VERIFIED',final_revision=after['revision_id'])
 print(json.dumps(receipt,ensure_ascii=False))
except Exception as error:
 checkpoint('NEEDS_REVIEW',error_type=type(error).__name__,error=str(error)[:350]);raise
'''


def main():
    module = (Path(__file__).resolve().parents[1] / 'pcm_eval/strict_model_template.py').read_text(encoding='utf8')
    print(call('MODULE = ' + repr(module) + '\n' + CODE, timeout=950))


if __name__ == '__main__':main()
