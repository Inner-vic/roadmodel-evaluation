"""Read real delivery content and reuse real small-scene artifacts; no inference."""
import json
from pathlib import Path
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1]
CODE='''import pathlib,json,sys,copy,hashlib
root=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-content-parser-20261109-r9');sys.path.insert(0,str(root))
from pcm_eval.delivery_source import LarkSource,inspect_record
from pcm_eval.delivery_workflow import material_identity
from pcm_eval.compile_preparation import prepare_compile
cfg=json.loads((root/'configs/evaluation.json').read_text());b=cfg['workstation_lark']
url='https://li.feishu.cn/record/PUBREF000000000000000000001'
source=LarkSource(executable=b['executable'],profile=b['profile']);raw=source.query(['base','+url-resolve','--url',url]);original=inspect_record(raw,url)
renamed=copy.deepcopy(raw)
for i,f in enumerate(renamed['data']['hint']['fields']['fields']):
 if f.get('type')!='not_support':f['name']='任意字段'+str(i)
parsed=inspect_record(renamed,url)
assert material_identity(original)==material_identity(parsed)==cfg['native_profile']['materials']
conflict=copy.deepcopy(renamed);hint=conflict['data']['hint'];hint['fields']['fields'].append({'id':'ambiguous','name':'其他内容','type':'text'})
hint['record_data']['ambiguous']='https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/model/detail/other?name=old'
assert len(inspect_record(conflict,url)['model_assets'])==3
run=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-37b5434cca814890bd9d')
w=json.loads((run/'workflow.json').read_text());result=json.loads((run/'progress.json').read_text())
hashes={}
for c in result['chains']:
 e=c['execution'];path=pathlib.Path(e['result_file']);actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==e['result_sha256'];hashes[c['kind']]=actual
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/content-parser-20261109');stage.mkdir(parents=True,exist_ok=True)
job=next(j for j in w['jobs'] if j['kind']=='compile_alignment');prep=prepare_compile(job,stage/'compile-content')
assert prep['state']=='PREPARED'
print(json.dumps({'state':'VERIFIED','live_record_read':True,'all_readable_fields_renamed':True,'material_identity_unchanged':True,
 'conflicting_model_preserved':True,'model_versions':[m['version'] for m in parsed['model_assets']],
 'compile_content_validation':prep['state'],'small_scene_result_hashes':hashes,'small_scene_frames':2,
 'new_inference_frames':0,'report_url':result['combined_report']['report_url']}))
'''
if __name__=='__main__':
    value=remote(CODE,timeout=70)
    out=ROOT/'evidence/content-parser-20261109/live-verification.json';out.write_text(json.dumps(value,indent=2))
    print(json.dumps(value))
