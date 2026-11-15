"""Bind publication receipts to the generated independent task deliveries."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'evidence/batch-upgrade-20261029'
task_path=base/'reports-01/task.json'
task=json.loads(task_path.read_text(encoding='utf-8'))
publications={}
for job,label in zip(task['jobs'],['model','compile']):
    path=base/('publication-'+label)/'publication-receipt.json'
    publication=json.loads(path.read_text(encoding='utf-8'))
    if publication['state']!='verified':raise ValueError('Publication not verified')
    job['states']['publication']='verified'
    job['publication']={k:publication[k] for k in ['document_id','revision_id','url','report_sha256']}
    manifest_path=Path(job['delivery']['manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['states']['publication']='verified'
    manifest['publication']=job['publication']
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    publications[label]=job['publication']
task['publication']='verified'
task_path.write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
space=json.loads((base/'placement/space-status.json').read_text())
if not any(x['value']=='stopped' for x in space['status_fields']):raise ValueError('Space not restored to stopped')
summary={'schema':'roadmodel.batch-upgrade/v1','new_inference_frames':0,
         'implemented':['registered_source_resolution','preflight','result_contract_validation','portable_report_delivery',
                        'business_layout','batch_report_generation','revision_bound_publication','project_skill',
                        'location_neutral_compile_driver','locked_workstation_gateway_adapter'],
         'publications':publications,'skill':json.loads((base/'skill-installation.json').read_text()),
         'migration':{'state':'environment_prerequisites_missing','space_restored':'stopped',
                      'direct_board_route':False,'workstation_tcp_route':True,
                      'missing':['26-10-20-1 dataset mount/index','paramiko in the target runtime',
                                 'verified gateway SSH authentication and runtime deployment'],
                      'new_two_frame_migration_run':False},
         'limits':['New unregistered model/dataset references need explicit bindings; no model substitution.',
                   'Unified model execution example is OCC-only; vision/trajectory retain their native scoped execution adapters.',
                   'No full evaluation, training, version comparison or release quality decision.',
                   'Publication content verified through native document API; final browser screenshot capture was unavailable.']}
(base/'completion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
