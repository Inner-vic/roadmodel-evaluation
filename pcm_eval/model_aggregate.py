"""Merge native OCC batches by counts, with exact frame and model provenance.

No inference, publication or arithmetic averaging of per-batch IoU is performed.
Vision AP remains the output of one independent full native evaluation.
"""
from copy import deepcopy
import json
from pathlib import Path
from .common import Result,digest,load_json,save_json,utc_now
from .validation import require_result


def sum_counters(results):
    totals={};expected_groups=None
    for result in results:
        groups={}
        for metric in result['metrics']:
            prefix,_,field=metric['name'].rpartition('.')
            if prefix not in ('occ','drivable') or field not in ('seen','correct','positive'):continue
            key=(prefix,json.dumps(metric['dimensions'],sort_keys=True))
            row=groups.setdefault(key,{})
            if field in row:raise ValueError('Duplicate native class counter')
            value=metric['value']
            if type(value)!=int or value<0:raise ValueError('Native counters must be nonnegative integers')
            row[field]=value
        if not groups or any(set(row)!={'seen','correct','positive'} for row in groups.values()):
            raise ValueError('Native counter set is incomplete')
        if expected_groups is None:expected_groups=set(groups)
        if set(groups)!=expected_groups:raise ValueError('Native class spaces differ between batches')
        for key,row in groups.items():
            if row['correct']>min(row['seen'],row['positive']):raise ValueError('Impossible native counters')
            target=totals.setdefault(key,dict.fromkeys(row,0))
            for field,value in row.items():target[field]+=value
    return [{'prefix':p,'dimensions':json.loads(d),**counts} for (p,d),counts in totals.items()]


def aggregate_occ(files,selected_frames,output,*,run_id,require_complete=True):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new aggregation directory')
    results=[load_json(p) for p in files]
    if not results:raise ValueError('No completed native batches')
    accepted=[];identity=None;duration=0
    for result in results:
        require_result(result,verify_artifacts=False)
        native=result['provenance']['native_run']
        if result['execution']['state']!='succeeded' or result['collection']['status']!='complete' or native.get('state')!='succeeded':
            raise ValueError('Incomplete or failed native batch cannot be aggregated')
        frames=native['selected_frames']
        if result['coverage']['observed']['model_samples']!=len(frames):raise ValueError('Batch sample count differs')
        # Input directory and available physical GPU vary; model settings cannot vary.
        env={k:v for k,v in native['environment'].items() if k not in ('DATASET_PATH','DATA_LINK_FILE','CUDA_VISIBLE_DEVICES')}
        current=(result['version_name'],native['source_commit'],native['checkpoint_sha256'],env)
        if identity is None:identity=current
        if current!=identity:raise ValueError('Batch weights, source or model configuration differ')
        accepted.extend(frames);duration+=native.get('duration_seconds',0)
    if len(set(map(tuple,accepted)))!=len(accepted):raise ValueError('Duplicate frames between native batches')
    if accepted!=selected_frames[:len(accepted)] or (require_complete and accepted!=selected_frames):
        raise ValueError('Native batches do not cover the exact frozen frame sequence')
    counters=sum_counters(results)
    proof={'schema':'roadmodel.occ-counter-merge/v1','method':'sum native seen/correct/positive then compute ratios',
           'completed_frames':len(accepted),'requested_frames':len(selected_frames),'selected_frames':accepted,
           'scope':'full' if accepted==selected_frames else 'partial',
           'batches':[{'path':str(Path(p).resolve()),'sha256':digest(Path(p))} for p in files],
           'counters':counters}
    source=save_json(output/'counter-merge.json',proof)
    merged=Result('model_accuracy',identity[0],output,len(accepted))
    merged.artifact(source,'native_counter_aggregation')
    for p in files:merged.artifact(Path(p),'completed_native_batch_result')
    for i,row in enumerate(counters):
        prefix=row['prefix'];dims=row['dimensions'];s,c,p=[row[k] for k in ('seen','correct','positive')]
        values={k:row[k] for k in ('seen','correct','positive')}
        if prefix=='occ':
            # Native metric_util.py convention: absent GT IoU/recall=100, precision=0 if no prediction.
            values.update(iou=c/(s+p-c)*100 if s else 100.,precision=c/p*100 if p else 0.,recall=c/s*100 if s else 100.)
        else:values['iou']=c/max(1,s+p-c)*100
        for name,value in values.items():merged.metric(prefix+'.'+name,value,'count' if name in ('seen','correct','positive') else 'percent',source,f'/counters/{i}',**dims)
    native=deepcopy(results[0]['provenance']['native_run'])
    native.update(run_id=run_id,selected_frames=accepted,expected_frames=len(accepted),duration_seconds=duration,
                  aggregation_scope=proof['scope'],dataset_version=results[0]['provenance']['native_run'].get('dataset_version'))
    merged.observe('model_samples',len(accepted))
    merged.value['execution']={'state':'succeeded','run_id':run_id,'source':'verified_native_batches'}
    merged.value['provenance']={'native_run':native,'counter_aggregation':str(source),
                               'requested_frames':len(selected_frames),'scope':proof['scope']}
    result=merged.finish();require_result(result)
    save_json(output/'normalized-result.json',result)
    return result
