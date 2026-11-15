"""Rehearse old work-order preparation migration on a copy; never resume it."""
import json
from pathlib import Path

from deploy_environment_flow import remote

CODE='''import pathlib,json,sys,shutil,numpy as np,hashlib,copy,xml.etree.ElementTree as ET
release=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-offline-20261107-r4')
sys.path.insert(0,str(release));sys.path.insert(0,str(release/'tools'))
from pcm_eval.common import digest,json_digest
from evaluation_worker import evaluate
from pcm_eval.combined_delivery import write_combined
root=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-997e5dda6e8546a4a935')
original=json.loads((root/'workflow.json').read_text());old=json.loads((root/'effective/workflow.json').read_text())
assert original['scope']=='validation' and original['jobs'][1]['plan']['expected_frames']==2
expected=copy.deepcopy(original);expected['jobs'][1]['ready']=False
expected['jobs'][1]['issues']=list(dict.fromkeys(expected['jobs'][1]['issues']+['台架暂不可用，编译材料准备结果已保存']))
expected.pop('workflow_sha256');expected['workflow_sha256']=json_digest(expected)
assert old==expected,'Old workflow differs beyond the known unavailable-bench gate'
assert not (root/'effective/compile_alignment/inference.json').exists()
assert not (root/'effective/workflow.lock').exists()
with np.load(root/'compile-scope/compile/checkpoint.npz',allow_pickle=False) as cp:assert int(cp['completed'])==0
source_files=[p for p in root.rglob('*') if p.is_file()]
assert sum(p.stat().st_size for p in source_files)<64*1024**2,'Only bounded metadata may be copied'
assert not any(p.suffix.lower() in ('.jpg','.png','.bin','.npy','.pth','.onnx') for p in source_files),'Raw data or model file found'
before={str(p.relative_to(root)):digest(p) for p in source_files}
target=stage/'legacy-compatibility-copy';assert not target.exists(),'Preserve the previous rehearsal'
shutil.copytree(root,target)
archive=target/'retained-r3-preparation';archive.mkdir()
shutil.move(target/'compile-scope',archive/'compile-scope')
shutil.move(target/'effective/workflow.json',archive/'effective-workflow.json')
(target/'effective/workflow.json').write_text(json.dumps(original))
cfg=json.loads((release/'configs/evaluation.json').read_text());assert cfg['bench_available'] is False
def no_inference(*args,**kwargs):raise AssertionError('Legacy rehearsal must reuse model and keep board disabled')
workflow,result=evaluate(copy.deepcopy(original),target,cfg,executor=no_inference)
assert next(c for c in result['chains'] if c['kind']=='compile_alignment')['state']=='WAITING_BENCH'
new=write_combined(workflow,result,target/'compatibility-report',reference_file=cfg['reference_file'],template_file=release/'configs/report_template.json')
tables=lambda p:[ET.tostring(n,encoding='unicode') for n in ET.fromstring('<root>'+p.read_text()+'</root>').iter('table')]
prior=list((root/'combined/generation').glob('*/report.xml'));assert len(prior)==1
assert tables(pathlib.Path(new['report_xml']))==tables(prior[0]),'Rehearsal changed business metric tables'
assert before=={str(p.relative_to(root)):digest(p) for p in source_files},'Live work order changed'
proof={'state':'OFFLINE_COMPATIBILITY_VERIFIED','task':original['run_id'],'scope':original['scope'],
       'expected_compile_frames':2,'allowed_workflow_changes':['compile.ready','compile.issues','internal.workflow_sha256'],
       'model_result_reused':True,'new_inference_frames':0,'bench_contacted':False,'dataset_copied':False,
       'live_work_order_unchanged':True,'five_business_tables_unchanged':True,
       'old_preparation_retained_on_copy':True,'new_preparation_bound_to_original_input':True,
       'physical_resume_started':False}
(stage/'legacy-compatibility-verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

if __name__=='__main__':
    result=remote(CODE,timeout=100)
    Path('evidence/compile-offline-20261107/legacy-compatibility-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
