"""Use the existing immutable report consumer; no Feishu write or inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE = r'''import pathlib,json,sys,sqlite3,hashlib,numpy as np
full=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
out=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-memory-gt-20261030-r1')
assert json.loads((out/'verification.json').read_text())['state']=='VERIFIED'
assert not (out/'report-consumer-verification.json').exists(),'Do not repeat completed report validation'
sys.path.insert(0,'/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
from pcm_eval.common import load_json,save_json,digest
from pcm_eval.compile_aggregate import aggregate_compile
from pcm_eval.compile_report import render_sections
from pcm_eval.business_layout import compile_business_sections
from pcm_eval.reference_report import organize
from pcm_eval.active_execution import process_alive
indices=[19,1001,2149,8333];frames=[];selected=[];request=load_json(full/'request.json')
counts={b+'_'+k:np.zeros(shape,np.int64) for b in ('board','onnx') for k,shape in (('confusion',(27,27)),('ds',(3,)))}
for index in indices:
 source=full/'compile/frames'/('frame_%06d'%index);frame=out/('frame-%06d'%index)
 for name in ('cosine_summary.json','input_validation.json','output_integrity.json'):
  assert not (frame/name).exists();(frame/name).write_bytes((source/name).read_bytes())
 with np.load(frame/'single-counts.npz',allow_pickle=False) as cp:
  for name in counts:counts[name]+=cp[name]
 frames.append(str(frame));selected.append(request['frames'][index])
checkpoint=out/'report-counts.npz'
with checkpoint.open('xb') as stream:np.savez(stream,completed=len(indices),**counts)
modelroot=pathlib.Path('/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01')
config=load_json(modelroot/'full-report-coordinator.json');execution=config['report']['compile_execution_request']
report=aggregate_compile(frames,selected,out/'report-consumer',request=execution,run_id='memory-GT-retained-validation-4',checkpoint=checkpoint)
sections=compile_business_sections(report,render_sections(report));xml='<document>'+''.join(s['xml'] for s in sections)+'</document>'
import xml.etree.ElementTree as ET;ET.fromstring(xml)
(out/'report-consumer/report.xml').write_text(xml)
cosine=load_json(out/'report-consumer/cosine_summary.json');tensor_records=sum(len(f['outputs']) for f in cosine['frames'])
assert tensor_records==72 and len(cosine['tensors'])==18 and report['result_validity']=='validated'
progress=load_json(full/'compile/progress.json');assert progress['state']=='STOPPED_AT_BOUNDARY' and progress['completed_frames']==2151
assert digest(full/'compile/checkpoint.npz')=='2c737cb749fb6268e4c1f8c630e86f60ebc83e9be490f19b6b8c1b4fca3f0bed'
assert not process_alive(progress['pid'],'formal_compile_remote.py')
assert not (full/'compile/controller.lock').exists() and (full/'STOP_AFTER_FRAME').exists()
dispatch=load_json(modelroot/'full-report-coordinator.dispatch.json')
proc=pathlib.Path('/proc')/str(dispatch['pid'])/'cmdline'
assert not proc.exists() or not proc.read_bytes(),'Old report coordinator unexpectedly running'
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/runtime.sqlite3') as db:
 rows=db.execute('SELECT kind,state,detail FROM chains WHERE task=?',('rm-a361dbe3df384cccb053',)).fetchall()
 assert len(rows)==2 and all(json.loads(row[2]).get('user_paused') for row in rows)
proof={'state':'VERIFIED','frames':4,'tensor_records':tensor_records,'result_validity':report['result_validity'],
 'report_generator':'existing immutable r4','business_XML_valid':True,'report_schema_unchanged':True,
 'production_prefix':2151,'production_checkpoint_unchanged':True,'full_evaluation_resumed':False,
 'Runtime_still_user_paused':True,'report_coordinator_still_stopped':True,'new_inference_frames':0,'Feishu_writes':0}
save_json(out/'report-consumer-verification.json',proof);print(json.dumps(proof))
'''

if __name__ == '__main__':print(call(CODE))
