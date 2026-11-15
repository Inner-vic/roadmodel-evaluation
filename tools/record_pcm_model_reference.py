"""Correct the model-input inventory using the reference already supplied."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import utc_now

root=Path('evidence/environment/model-input-review-20261027')
doc=json.loads((root/'release-reference-current.json').read_text(encoding='utf-8-sig'))['data']['document']
assert 'PUBREF000000000000000000001' in doc['content']
error=json.loads((root/'release-model-base-resolve.stderr.txt').read_text(encoding='utf-8-sig'))
assert error['error']['code']==800004011
browser=(root/'release-model-browser-access.txt').read_text(encoding='utf-8')
assert '暂无访问权限' in browser
record={'schema':'pcm.provided-model-reference/v1','checked_at':utc_now(),
 'source_report':'https://li.feishu.cn/wiki/DEMO00000000000000000000004',
 'source_document_id':doc['document_id'],'source_revision':doc['revision_id'],
 'reference_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'source_report_version_label':'v3.2.2','resource_type':'shared_base_record',
 'provided_by_user':True,'access_state':'read_access_denied',
 'browser_observation':'暂无访问权限','base_api_error_code':800004011,
 'underlying_checkpoint':'not_yet_read','paired_configuration':'not_yet_read',
 'eligibility_for_model_evaluation':'pending_content_inspection; not determined unusable',
 'correction':'The model reference was already provided in the original PCM report. Do not ask for a new model/version before resolving it.',
 'needed_action':'Read access to the existing record, or its actual model artifact link/path; no need to reassemble all model information',
 'permission_request_sent':False,'new_evaluation_started':False}
(root/'provided-model-reference.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readiness_path=root/'readiness.json'
readiness=json.loads(readiness_path.read_text(encoding='utf-8'))
readiness['provided_model_reference']=record
readiness['requires_user'][0]={'id':'access_existing_model_reference',
 'provide':'Read access to the model record already supplied, or copy the actual model asset link/path from that record',
 'reason':'Model-reference content cannot currently be read; this is not an absence of a provided model reference'}
readiness['checked_at']=record['checked_at']
readiness_path.write_text(json.dumps(readiness,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assets_path=Path('configs/known_assets.json')
assets=json.loads(assets_path.read_text(encoding='utf-8'))
assets['model_accuracy_readiness']=str(readiness_path)
assets['provided_pcm_model_reference']={k:record[k] for k in ['reference_url','source_report_version_label','access_state','provided_by_user']}
assets_path.write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'model_reference_provided':True,'access_state':record['access_state'],'new_model_not_requested':True,'permission_request_sent':False}))
