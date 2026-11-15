"""Update the authorized report from the fresh model results, without inference.

Only the original PCM model tables are rendered. All business files, CLI output
and publication receipts stay on the workstation. Unknown writes are not retried.
"""
from connect_active_entry import call


CODE = r'''import pathlib,json,hashlib,sys,os,subprocess,datetime,gzip,xml.etree.ElementTree as ET
base=pathlib.Path('/mnt/data/bench-eval/roadmodel')
run=base/'formal/model-timing-v322-20261102-r1'
out=base/'publication/model-full-20261102-r1'
doc='DEMO00000000000000000000002'
consumer=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
sys.path.insert(0,str(consumer))
from pcm_eval.reference_report import organize,display
from pcm_eval.business_layout import table,p
load=lambda path:json.loads(path.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
text=lambda node:''.join(node.itertext())
def values(root):return [[[text(c) for c in row] for row in t.iter('tr')] for t in root.iter('table')]
if (out/'verification.json').exists():
 print(json.dumps(load(out/'verification.json'),ensure_ascii=False));sys.exit(0)
assert not out.exists(),'Publication already attempted; inspect journal before continuing'
model=load(run/'report/model-report.json');proof=load(run/'report/verification.json');request=load(run/'request.json')
assert proof['state']=='SUCCEEDED' and sha(run/'report/model-report.json')==proof['report_json_sha256']
assert all(sha(pathlib.Path(path))==h for path,h in proof['original_result_hashes'].items())
assert model['run']['selected_frames']==request['frames'] and len(request['frames'])==8337
vision=model['head_results']['vision'];vn=vision['provenance']['native_run']
assert vn['selected_frames']==request['vision']['frames'] and len(vn['selected_frames'])==11543
assert not model['trajectory_metrics'] and list(model['head_results'])==['vision']
reference=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/trial-20261030/retained/reference.json')
prior=load(base/'validation/entry-live-20261030-r3/result-report-r4/data.json')
organized=organize(model,prior['atomic_results']['compile'],load(reference))
tables=organized['tables'][:3];assert [t['id'] for t in tables]==['occ','vision_summary','vision_classes']
expected=[[t['headers']]+[[display(c) for c in row] for row in t['rows']] for t in tables]
compile_root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
compile_progress=load(compile_root/'compile/progress.json')
assert compile_progress['completed_frames']==2151 and compile_progress['expected_frames']==8337
out.mkdir(parents=True)
journal={'state':'PREPARING','document_id':doc,'native_task_id':run.name,'started_at':now(),
         'model_json_sha256':sha(run/'report/model-report.json'),'template_sha256':sha(reference),
         'new_inference_frames':0,'compile_execution_dispatched':False}
def checkpoint(state,**fields):
 journal.update(state=state,**fields);q=out/'journal.tmp';q.write_text(json.dumps(journal,ensure_ascii=False,indent=2));q.replace(out/'journal.json')
checkpoint('PREPARING')
seq=0
def cli(args):
 global seq
 seq+=1
 env={**os.environ,'LARKSUITE_CLI_CONFIG_DIR':'/mnt/data/bench-eval/service/gateway/runtime/roadmodel/lark-cli-config',
      'LARKSUITE_CLI_NO_UPDATE_NOTIFIER':'1','LARKSUITE_CLI_NO_SKILLS_NOTIFIER':'1'}
 proc=subprocess.run(['/mnt/data/bench-eval/service/roadmodel/lark-cli/node_modules/@larksuite/cli/bin/lark-cli',*args,'--as','bot'],
                     cwd=out,env=env,capture_output=True,timeout=160)
 (out/('%02d.stdout.json'%seq)).write_bytes(proc.stdout);(out/('%02d.stderr.txt'%seq)).write_bytes(proc.stderr)
 value=json.loads((proc.stdout or proc.stderr).decode('utf-8-sig'))
 assert proc.returncode==0 and value.get('ok'),'CLI failed; inspect workstation journal'
 data=value['data'];assert not data.get('warnings') and data.get('result') not in ('failed','partial_success'),'Partial operation; reconcile'
 return data
def read():
 d=cli(['docs','+fetch','--doc',doc,'--detail','full'])['document']
 return ET.fromstring('<root>'+d['content']+'</root>'),d['revision_id']
def shape(t):return t[0],[row[:2] if t[0][0]=='父类' else row[:1] for row in t[1:]]
try:
 before,rev=read();old=values(before);assert len(old)==5
 assert all(shape(a)==shape(b) for a,b in zip(old[:3],expected)),'Template headers or row labels changed'
 resources={n.get('id') for n in before.iter('source') if n.get('id')}
 nodes=list(before);end=next(i for i,n in enumerate(nodes) if n.tag=='h1' and text(n)=='编译精度评测')
 assert nodes[0].tag=='title' and len(list(ET.fromstring('<root>'+''.join(ET.tostring(n,encoding='unicode') for n in nodes[1:end])+'</root>').iter('table')))==3
 assert not any(n.tag in ('source','figure','img','whiteboard','sheet','bitable') for item in nodes[1:end] for n in item.iter())
 local=datetime.timezone(datetime.timedelta(hours=8))
 fmt=lambda s:datetime.datetime.fromisoformat(s).astimezone(local).strftime('%Y-%m-%d %H:%M:%S')
 updated=datetime.datetime.now(local).strftime('%Y-%m-%d %H:%M:%S')
 occ_scenes=len(set(f[0] for f in request['frames']));vision_scenes=len(set(f[0] for f in vn['selected_frames']))
 def h(level,title):return '<h%d seq="auto">%s</h%d>'%(level,title,level)
 def render_table(i):return table(tables[i]['headers'],[[display(c) for c in row] for row in tables[i]['rows']])
 content=[p('模型版本：v3.2.2'),
  '<p>模型交付：<a href="https://li.feishu.cn/record/PUBREF000000000000000000001">原报告所附模型记录</a></p>',
  p('本次模型结果完成：'+fmt(proof['all_evaluation_results_ready_at'])+'；报告更新：'+updated+'（北京时间）。'),
  p('模型精度：OCC、视觉分别覆盖各自完整验证集；轨迹本轮未执行。编译精度：下方保留历史1帧结果，全量累计2151/8337帧，因台架被他人占用等待接续，尚未生成全量编译报告。'),
  '<p>表格依据：<a href="https://li.feishu.cn/wiki/DEMO00000000000000000000004">PCM 发版报告V3.1.0</a>。保留原表指标、单位与顺序，仅去除跨版本列。</p>',
  h(1,'模型精度评测'),h(2,'occ头'),
  p('数据集26-10-20-1；%d场景 / 8337帧，完整验证集；运行%s。'%(occ_scenes,model['run_id'])),render_table(0),
  h(2,'轨迹头'),p('本轮未评测；未引用历史小场景分数。'),
  h(2,'视觉头'),p('数据集26-08-14-0；%d场景 / 11543帧，完整验证集；运行%s。'%(vision_scenes,vn['run_id'])),
  p('核心类 mAP 与准确率'),render_table(1),p('参与 mAP 的子类明细'),render_table(2)]
 body=''.join(content);tree=ET.fromstring('<root>'+body+'</root>');assert values(tree)==expected
 (out/'model-body.xml').write_text(body,encoding='utf8')
 data={'schema':'roadmodel.model-business-publication/v1','native_task_id':run.name,'generated_at':now(),
       'source_model_report':{'path':str(run/'report/model-report.json'),'sha256':sha(run/'report/model-report.json')},
       'source_verification':{'path':str(run/'report/verification.json'),'sha256':sha(run/'report/verification.json')},
       'template':{'path':str(reference),'sha256':sha(reference),'url':organized['template_source']['url']},
       'tables':tables,'model':model,'source_verification_receipt':proof,
       'trajectory_scope':'not_run_in_current_full_evaluation','compile_tables':'historical_one_frame_unchanged',
       'occ_scenes':occ_scenes,'occ_frames':8337,'vision_scenes':vision_scenes,'vision_frames':11543,
       'new_inference_for_publication':0}
 raw=json.dumps(data,ensure_ascii=False,indent=2).encode();(out/'model-data.json').write_bytes(raw)
 filename='model-full-20261102-'+hashlib.sha256(raw).hexdigest()[:12]+'.json.gz'
 with gzip.open(out/filename,'wb') as stream:stream.write(raw)
 checkpoint('WRITING_MODEL',start_revision=rev,original_source_blocks=sorted(resources))
 cli(['docs','+update','--doc',doc,'--command','block_replace','--start-block-id',nodes[1].get('id'),
      '--end-block-id',nodes[end-1].get('id'),'--content','@./model-body.xml','--doc-format','xml','--revision-id',str(rev)])
 current,rev=read();assert values(current)==expected+old[3:],'Report tables differ after update'
 assert resources.issubset({n.get('id') for n in current.iter('source')})
 assert '135场景 / 8337帧' in text(current) and '577场景 / 11543帧' in text(current)
 # Historical attachments stay in place, but their old caption must not imply current scope.
 captions=[n for n in current if n.tag=='p' and text(n).startswith('本次补覆盖更新的数据与核对材料如下')]
 if captions:
  assert len(captions)==1
  (out/'history-caption.xml').write_text(p('以下为历史小场景附件，供追溯，不代表本轮范围。本轮模型结果为2026-11-02的OCC8337帧、视觉11543帧，最新结构化结果见文末同名日期附件。'),encoding='utf8')
  checkpoint('LABELING_HISTORY',body_verified_revision=rev)
  cli(['docs','+update','--doc',doc,'--command','block_replace','--block-id',captions[0].get('id'),
       '--content','@./history-caption.xml','--doc-format','xml','--revision-id',str(rev)])
  current,rev=read();assert values(current)==expected+old[3:]
 (out/'attachment.xml').write_text('<source path="@./'+filename+'" name="'+filename+'"/>',encoding='utf8')
 checkpoint('APPENDING_CURRENT_MODEL_DATA',body_verified_revision=rev)
 cli(['docs','+update','--doc',doc,'--command','append','--content','@./attachment.xml','--doc-format','xml','--revision-id',str(rev)])
 after,rev=read();assert values(after)==expected+old[3:]
 assert resources.issubset({n.get('id') for n in after.iter('source')})
 assert any(n.get('name')==filename for n in after.iter('source'))
 assert all(sha(pathlib.Path(path))==h for path,h in proof['original_result_hashes'].items())
 receipt={'state':'VERIFIED','report_url':'https://li.feishu.cn/docx/'+doc,'revision_id':rev,'checked_at':now(),
          'native_task_id':run.name,'occ_scenes':occ_scenes,'occ_frames':8337,'vision_scenes':vision_scenes,'vision_frames':11543,
          'model_tables_match_current_results':True,'model_table_headers_and_row_labels_match_original_template':True,
          'compile_two_tables_unchanged':True,'original_sources_preserved':True,'new_inference_frames':0,
          'model_json_sha256':sha(run/'report/model-report.json'),'published_data_sha256':hashlib.sha256(raw).hexdigest(),
          'current_data_attachment':filename,'whole_evaluation_complete':False}
 (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));checkpoint('VERIFIED',final_revision=rev)
 print(json.dumps(receipt,ensure_ascii=False))
except Exception as error:
 checkpoint('NEEDS_REVIEW',error_type=type(error).__name__,error=str(error)[:350]);raise
'''


if __name__ == '__main__':
    # The historical body above included non-template explanations. The user's
    # strict-template instruction supersedes it; never re-publish that layout.
    from publish_strict_model_report import main
    main()
