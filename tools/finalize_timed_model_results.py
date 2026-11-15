"""Finish the fresh model run from validated results; never imports a launcher."""
from connect_active_entry import call
from prepare_timed_model_rerun import TASK


def main():
    code='''import pathlib,json,sys,hashlib,time,datetime,copy,xml.etree.ElementTree as ET
r=pathlib.Path(%r);consumer=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4')
sys.path.insert(0,str(consumer))
from pcm_eval.model_aggregate import aggregate_occ
from pcm_eval.model_report import build_model_report,render_model_xml
from pcm_eval.business_layout import render_model_business
from pcm_eval.validation import require_result
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=load(r/'run-config.json');req=load(r/'request.json');start=load(r/'timing-start.json');old=load(r/'timing-summary.json')
assert not (r/'report/verification.json').exists(),'Report already verified; read existing receipt'
assert old['state']=='NEEDS_REVIEW' and load(r/'model/progress.json')['state']=='OCC_SUCCEEDED' and load(r/'vision/progress.json')['state']=='SUCCEEDED'
assert start['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
files=[r/'model'/f'batch-{i:04d}-a02/normalized-result.json' for i in range(84)]
sources=[load(p) for p in files];vision=load(r/'vision/normalized-result.json');vn=vision['provenance']['native_run']
assert all(d['provenance']['native_run']['run_id']==r.name+f'-occ-{i:04d}' for i,d in enumerate(sources))
assert vn['run_id']==r.name+'-vision' and vn['selected_frames']==req['vision']['frames']
assert len(req['frames'])==8337 and len(req['vision']['frames'])==11543
before={str(p):sha(p) for p in files+[r/'vision/normalized-result.json',r/'request.json',r/'timing-events.jsonl']}
assert all(sha(pathlib.Path(cfg['compile_root'])/n)==h for n,h in cfg['compile_pause_hashes'].items())
events=[json.loads(s) for s in (r/'timing-events.jsonl').read_text().splitlines()]
head_names={e['token']:e['head'] for e in events if e['stage']=='head_end_to_end' and e['state']=='STARTED'}
heads=[dict(e,head=head_names[e['token']]) for e in events if e['stage']=='head_end_to_end' and e['state']=='SUCCEEDED'];assert len(heads)==2
ready=max(heads,key=lambda e:e['monotonic_ns']);first=next(e for e in events if e['stage']=='head_end_to_end' and e['state']=='STARTED')
repair_start=time.monotonic_ns();now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
if (r/'report/occ/normalized-result.json').exists():
 proof=load(r/'report/occ/counter-merge.json');assert proof['selected_frames']==req['frames']
 assert proof['batches']==[{'path':str(p),'sha256':sha(p)} for p in files]
 model=load(r/'report/occ/normalized-result.json');require_result(model,verify_artifacts=True)
 assert model['provenance']['native_run']['run_id']==r.name+'-occ'
else:model=aggregate_occ(files,req['frames'],r/'report/occ',run_id=r.name+'-occ')
# The original aggregator copied batch zero's date range; report the complete range.
native=model['provenance']['native_run'];native['started_at']=min(d['provenance']['native_run']['started_at'] for d in sources);native['finished_at']=max(d['provenance']['native_run']['finished_at'] for d in sources)
native['timing_scope']='all native OCC batches; duration_seconds is eval-process sum, not elapsed wall time'
require_result(model,verify_artifacts=False)
(r/'report/occ/normalized-result.json').write_text(json.dumps(model,ensure_ascii=False,indent=2))
baseline=load(pathlib.Path(cfg['baseline_model_report']));binding=copy.deepcopy(baseline['binding']);binding['dataset'].update(selected_frames=req['frames'],version='26-10-20-1')
report=build_model_report(model,binding=binding,head_results=[vision]);report['timing_run_id']=r.name
timing={'first_task_started_at':start['started_at'],'all_evaluation_results_ready_at':ready['utc'],
        'evaluation_elapsed_seconds':(ready['monotonic_ns']-start['started_monotonic_ns'])/1e9,
        'evaluation_pipeline_seconds':(ready['monotonic_ns']-first['monotonic_ns'])/1e9,
        'heads':{h['head']:{'completed_at':h['utc'],'elapsed_seconds':h['duration_seconds']} for h in heads},
        'OCC_native_eval_process_sum_seconds':sum(d['provenance']['native_run']['duration_seconds'] for d in sources),
        'vision_native_eval_process_seconds':vn['duration_seconds'],
        'report_repair_started_at':now(),'report_repair_wait_seconds':(repair_start-ready['monotonic_ns'])/1e9,
        'includes_setup_failures_and_waiting':True,'Feishu_publication_included':False}
report['timing']=timing
(r/'report/model-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
xml=render_model_business(report,render_model_xml(report,json_attachment='model-report.json'));ET.fromstring('<root>'+xml+'</root>')
(r/'report/model-report.xml').write_text(xml,encoding='utf8')
assert all(sha(pathlib.Path(p))==h for p,h in before.items()),'Original native evidence changed'
assert all(sha(pathlib.Path(cfg['compile_root'])/n)==h for n,h in cfg['compile_pause_hashes'].items())
timing.update(report_generated_at=now(),report_generation_seconds=(time.monotonic_ns()-repair_start)/1e9,
              elapsed_through_report_seconds=(time.monotonic_ns()-start['started_monotonic_ns'])/1e9)
(r/'report/timing.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2))
history=r/'timing-history';history.mkdir(exist_ok=True);(history/'report-import-failure.json').write_text(json.dumps(old,indent=2))
proof={'state':'SUCCEEDED','task_id':r.name,'verified_OCC_frames':8337,'verified_vision_frames':11543,
       'old_results_imported':0,'new_or_repeated_inference_during_repair':0,'original_result_hashes':before,
       'consumer_module':str(consumer/'pcm_eval/model_aggregate.py'),'consumer_sha256':sha(consumer/'pcm_eval/model_aggregate.py'),
       'report_json_sha256':sha(r/'report/model-report.json'),'report_xml_sha256':sha(r/'report/model-report.xml'),
       'xml_valid':True,'compile_paused_and_unchanged':True,**timing}
(r/'report/verification.json').write_text(json.dumps(proof,indent=2))
summary={**old,'state':'SUCCEEDED','observed_at':timing['report_generated_at'],
         'elapsed_seconds':timing['elapsed_through_report_seconds'],'evaluation_elapsed_seconds':timing['evaluation_elapsed_seconds'],
         'evaluation_results_ready_at':ready['utc'],'report_recovered_without_inference':True,'report_verification':str(r/'report/verification.json'),
         'Feishu_published':False}
for name in ['timing-summary.json','progress.json']:
 p=r/name;q=p.with_suffix('.final.tmp');q.write_text(json.dumps(summary,ensure_ascii=False,indent=2));q.replace(p)
print(json.dumps({k:v for k,v in proof.items() if k!='original_result_hashes'}))
''' % TASK
    print(call(code,timeout=110))


if __name__=='__main__':main()
