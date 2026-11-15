"""Record completed input discovery and small-sample evidence without running evaluation."""
import json,datetime,hashlib
from pathlib import Path
r=Path('evidence/model-evaluation/remaining-heads-20261027');report=Path('evidence/reports/model-heads-v2-20261027')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
results={kind:load(r/'collected'/folder/'normalized_result-reviewed.json') for kind,folder in [('vision','vision-01'),('trajectory','trajectory-01-recovered')]}
receipt={'schema':'roadmodel.model-head-readiness/v1','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model_version':'v3.2.2','state':'small_scene_validation_completed','missing_input_fields':[],
    'input_receipt':str(r/'input-receipt.json'),'additional_input_files':19,'additional_input_bytes':20110304,
    'heads':{kind:{'state':'small_scene_result_available','run_id':d['provenance']['native_run']['run_id'],
        'dataset_version':d['provenance']['native_run']['dataset_version'],'selected_frames':d['provenance']['native_run']['selected_frames'],
        'normalized_result':str(r/'collected'/('vision-01' if kind=='vision' else 'trajectory-01-recovered')/'normalized_result-reviewed.json'),
        'metric_records':len(d['metrics']),'collection':d['collection']} for kind,d in results.items()},
    'coverage_limits':['Vision: only one GT class per core task; mAP degeneracy','Trajectory: straight front-wheel paths only; rear wheels masked by native model','Trajectory: no GT front events; related metrics unavailable'],
    'quality_findings':{'vision_bump_type_accuracy_percent':50.,'trajectory_label_accuracy_percent':0.,'trajectory_height_mae_cm':.7220078861012178},
    'quality_assessment':'not_assessed','new_inference_frames_in_postprocessing':0,
    'independent_report':str(report/'report.json'),'publication':load(report/'publication/publication-receipt.json'),
    'validation':{'command':'python tools/verify_gt_changes.py','passed':80}}
(r/'completion-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':receipt['state'],'missing_input_fields':[],'head_metrics':{k:len(v['metrics']) for k,v in results.items()},'report_revision':receipt['publication']['revision_id']}))
