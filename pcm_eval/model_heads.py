"""Collect trajectory-head evidence independently from execution and publication."""
from pathlib import Path
from .common import Result,digest


def trajectory_unavailable(data):
    """Native helpers guard empty denominators with 1; expose those as unavailable."""
    m=data['metrics'];c=data['raw_counts'];out={}
    for key in m:
        if key.startswith('front_'):
            if key in ['front_gt_events','front_pred_events','front_matched_events']:continue
            denominator=m['front_gt_events'] if key=='front_miss_rate' else m['front_pred_events'] if key=='front_false_event_rate' else m['front_matched_events']
            if not denominator:out[key]='no_gt_events' if key=='front_miss_rate' else 'no_pred_events' if key=='front_false_event_rate' else 'no_matched_events'
        elif key.startswith('boundary_') and key!='boundary_matched_wheels':
            denominator=c.get('boundary_gt_positive',0) if key=='boundary_miss_rate' else c.get('boundary_gt_negative',0) if key=='boundary_false_event_rate' else m['boundary_matched_wheels']
            if not denominator:out[key]='no_boundary_denominator'
        elif key.startswith('z_float_') and not m['float_valid_points']:out[key]='no_valid_height_points'
    return out


def collect_trajectory(run_dir,version_name,expected_frames):
    root=Path(run_dir).resolve();r=Result('model_accuracy',version_name,root,expected_frames)
    data=r.read(root/'trajectory_metrics.json','native_trajectory',required=True)
    receipt=r.read(root/'execution_receipt.json','execution_receipt',required=True)
    pre=r.read(root/'model-preflight.json','model_preflight',required=True)
    r.read(root/'source_inference_receipt.json','source_inference_receipt')
    r.read(root/'launch.json','head_launch',required=True)
    if not data or data.get('schema')!='roadmodel.native-trajectory/v1':raise ValueError('Native trajectory metrics required')
    if receipt.get('state')!='succeeded' or pre.get('state')!='passed':raise ValueError('Successful postprocessing and input preflight required')
    frames=data['frames'];selected=[[x['scene'],x['frame_index']] for x in frames]
    if len(frames)!=expected_frames or len(set(map(tuple,selected)))!=expected_frames or selected!=receipt['selected_frames']:
        raise ValueError('Trajectory sample scope mismatch')
    if data['run_id']!=receipt['run_id'] or data['source_commit']!=receipt['source_commit'] or data['gt_source']!='mesh':
        raise ValueError('Trajectory provenance mismatch')
    if sum(f['valid_points'] for f in frames)!=data['metrics']['valid_points'] or data['raw_counts']['label_total']!=data['metrics']['valid_points']:
        raise ValueError('Valid-point counters disagree')
    for frame in frames:
        path=root/frame['prediction_file']
        if not path.resolve().is_relative_to(root) or digest(path)!=frame['sha256']:raise ValueError('Retained trajectory prediction mismatch')
        r.artifact(path,'retained_trajectory_prediction')
    r.observe('model_samples',expected_frames)
    unavailable=trajectory_unavailable(data)
    counts={'valid_points','float_valid_points','boundary_matched_wheels','front_gt_events','front_pred_events','front_matched_events'}
    ratios={'smooth_ratio_d1','smooth_ratio_d2','shape_ratio_error','front_within10','front_within20','front_miss_rate','front_false_event_rate','boundary_miss_rate','boundary_false_event_rate'}
    for key,value in data['metrics'].items():
        if key in unavailable or value is None:continue
        if key=='per_class_recall':
            for cid,recall in value.items():
                r.metric('trajectory.class.recall',recall,'ratio',root/'trajectory_metrics.json','/metrics/per_class_recall/'+str(cid),class_id=str(cid))
            continue
        unit='count' if key in counts else 'centimeter' if key.endswith('_cm') else 'percent' if key.endswith('_acc') else 'bin' if key.endswith('_bins') else 'ratio' if key in ratios else 'source_defined'
        r.metric('trajectory.'+key,value,unit,root/'trajectory_metrics.json','/metrics/'+key)
    r.value['execution']={k:receipt[k] for k in ['state','exit_code','source']}
    r.value['provenance'].update(native_run=receipt,model_preflight=pre,native_trajectory=data,
                                unavailable_metrics=unavailable)
    return r.finish()
