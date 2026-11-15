"""Analyze retained logits/features and exact semantic-head decision margins."""
import json,csv,hashlib,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np

ROOT=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01')
RUN=ROOT/'trajectory-diagnostic-01'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def quantiles(a):return {str(q):float(np.quantile(a,q)) for q in [0,.05,.5,.95,1]}
def hist(a):
    v,n=np.unique(a,return_counts=True);return {str(int(k)):int(c) for k,c in zip(v,n)}
def softmax(a):
    e=np.exp(a-a.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
receipt=load(RUN/'diagnostic-receipt.json');pre=load(RUN/'preflight.json')
assert receipt['new_forward_frames']==2
with np.load(RUN/'semantic-weights.npz',allow_pickle=False) as a:w=a['weight'].reshape(3,27,96)[pre['center_offset_index']];bias=a['bias'].reshape(3,27)[pre['center_offset_index']]
diff=w[1]-w[2];bias_diff=float(bias[1]-bias[2])
frames=[];csv_rows=[]
for i,row in enumerate(receipt['frames']):
    with np.load(RUN/f'frame-{i}-native.npz',allow_pickle=False) as data:
        a={k:data[k] for k in data.files}
    mask=a['result_traj_valid_mask'].reshape(-1).astype(bool)&(a['result_dp3_gt_traj_label'].reshape(-1)!=26)
    logits=a['result_pred_traj_label_logits'].reshape(1600,27);prob=a['result_traj_probabilities'].reshape(1600,27)
    pred=a['result_traj_label'].reshape(-1);gt=a['result_dp3_gt_traj_label'].reshape(-1)
    xy=a['result_traj_xy_used'].reshape(1600,2);f=a['surface_cls_head_input0'].reshape(800,96)
    assert np.array_equal(logits.argmax(-1)[mask],pred[mask])
    assert np.allclose(softmax(logits)[mask],prob[mask],atol=2e-7)
    linear=f@w.T+bias
    reconstruction=float(np.max(np.abs(linear-logits[:800])))
    assert reconstruction<5e-4,('Linear layer reconstruction',reconstruction)
    fm=mask[:800];margin=logits[mask,1]-logits[mask,2]
    feature_contribution=f[fm]@diff
    assert np.allclose(feature_contribution+bias_diff,margin,atol=5e-4)
    stage=[]
    for name in ['surface_semantic_context_input0']+['surface_semantic_context.blocks.'+str(j) for j in range(4)]:
        feature=a[name].reshape(800,96);projected=feature@w.T+bias
        stage.append({'stage':name,'diagnostic_projection_predictions':hist(projected.argmax(-1)[fm]),
            'mean_asphalt_minus_cement_margin':float((projected[fm,1]-projected[fm,2]).mean()),
            'valid_feature_rms':float(np.sqrt(np.mean(feature[fm]**2))),
            'median_spatial_channel_std':float(np.median(feature[fm].std(axis=0)))})
    branches={}
    for name in ['surface_position_pack','surface_lidar_pack','surface_gaussian_attention']:
        feature=a[name].reshape(800,96)[fm]
        branches[name]={'rms':float(np.sqrt(np.mean(feature**2))),'mean_abs':float(np.abs(feature).mean()),
            'median_spatial_channel_std':float(np.median(feature.std(axis=0))),'nonzero_fraction':float(np.mean(feature!=0))}
    replay_file=RUN/f'frame-{i}-semantic-interventions.npz';interventions={}
    if replay_file.is_file():
        with np.load(replay_file,allow_pickle=False) as data:
            for name in data.files:
                if not name.endswith('_logits'):continue
                v=data[name].reshape(800,27)[fm];pp=softmax(v)
                interventions[name.removesuffix('_logits')]={'prediction_counts':hist(v.argmax(-1)),
                    'mean_asphalt_probability':float(pp[:,1].mean()),'mean_cement_probability':float(pp[:,2].mean()),
                    'asphalt_minus_cement_margin':quantiles(v[:,1]-v[:,2])}
    rank_gt=1+(logits[mask]>logits[mask,2,None]).sum(axis=1)
    channels=(f[fm]*diff).mean(axis=0);order=np.argsort(np.abs(channels))[::-1][:12]
    fmprob=prob[mask]
    info={'frame':row['frame_index'],'valid_points':int(mask.sum()),'correct':int((pred[mask]==gt[mask]).sum()),
        'prediction_counts':hist(pred[mask]),'asphalt_probability':quantiles(fmprob[:,1]),'cement_probability':quantiles(fmprob[:,2]),
        'cement_rank_counts':hist(rank_gt),'top2_class_counts':hist(np.argsort(logits[mask],axis=1)[:,-2]),
        'asphalt_minus_cement_logit_margin':quantiles(margin),'mean_logit_margin':float(margin.mean()),
        'mean_feature_margin_contribution':float(feature_contribution.mean()),'bias_margin_contribution':bias_diff,
        'native_linear_reconstruction_max_error':reconstruction,'branch_features':branches,'semantic_stages':stage,
        'largest_mean_margin_channels':[{'channel':int(c),'mean_margin_contribution':float(channels[c])} for c in order],
        'semantic_interventions':interventions,
        'natural_lidar_hit_count':int(a['surface_lidar_conditioner_1'].astype(bool).sum()),
        'natural_lidar_query_count_including_padding':int(a['surface_lidar_conditioner_1'].size),
        'valid_surface_queries':int(a['result_surface_valid_mask'].astype(bool).sum()),
        'valid_surface_lidar_hits':int((a['surface_lidar_conditioner_1'].astype(bool)&a['result_surface_valid_mask'].astype(bool)).sum()),
        'original_prediction_exact_match':row['prior_output_exact_match']}
    component=RUN/'semantic-component-replay-02'
    if (component/f'frame-{i}.npz').is_file():
        info['gaussian_component_interventions']={}
        with np.load(component/f'frame-{i}.npz',allow_pickle=False) as archive:
            for name in archive.files:
                if not name.endswith('_logits'):continue
                v=archive[name].reshape(800,27)[fm];pp=softmax(v)
                info['gaussian_component_interventions'][name.removesuffix('_logits')]={'prediction_counts':hist(v.argmax(-1)),
                    'mean_asphalt_probability':float(pp[:,1].mean()),'mean_cement_probability':float(pp[:,2].mean()),
                    'mean_asphalt_minus_cement_margin':float((v[:,1]-v[:,2]).mean())}
    frames.append(info)
    for j in np.flatnonzero(mask):csv_rows.append([row['frame_index'],j//400,j%400,float(xy[j,0]),float(xy[j,1]),int(gt[j]),int(pred[j]),float(prob[j,1]),float(prob[j,2]),float(logits[j,1]-logits[j,2])])
with (ROOT/'pointwise-semantic-decisions.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['frame','wheel','slot','x_m','y_m','GT','prediction','p_asphalt','p_cement','logit_asphalt_minus_cement']);writer.writerows(csv_rows)
analysis={'schema':'roadmodel.trajectory-semantic-diagnostic-analysis/v1','run_id':receipt['run_id'],
    'selected_frames':receipt['selected_frames'],'new_forward_frames':receipt['new_forward_frames'],'execution_state':receipt['state'],
    'frames':frames,'linear_bias_only':{'prediction':int(bias.argmax()),'probabilities':softmax(bias).tolist(),'asphalt_minus_cement_margin':bias_diff},
    'diagnostic_interventions_are_quality_scores':False,'pointwise_csv':str(ROOT/'pointwise-semantic-decisions.csv')}
if (RUN/'semantic-component-replay-02/receipt.json').is_file():analysis['component_replay']=load(RUN/'semantic-component-replay-02/receipt.json')
amp=ROOT/'trajectory-amp-control-01/receipt.json'
if amp.is_file():
    analysis['full_model_amp_control']=load(amp)
    assert analysis['full_model_amp_control']['state']=='succeeded'
    assert analysis['full_model_amp_control']['selected_frames']==analysis['selected_frames']
    assert all(x['same_trajectory_xy'] and x['same_GT'] for x in analysis['full_model_amp_control']['frames'])
    analysis['total_new_full_model_frame_passes']=analysis['new_forward_frames']+analysis['full_model_amp_control']['new_full_model_forward_frames']
    analysis['unique_frames']=len(analysis['selected_frames'])
(ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
print(json.dumps({'state':receipt['state'],'unique_frames':len(analysis['selected_frames']),
    'total_new_full_model_frame_passes':analysis.get('total_new_full_model_frame_passes'),
    'frames':[{'frame':x['frame'],'correct':x['correct'],'valid_points':x['valid_points'],
        'asphalt_probability_median':x['asphalt_probability']['0.5'],'logit_margin_mean':x['mean_logit_margin'],
        'valid_surface_queries':x['valid_surface_queries'],'valid_surface_lidar_hits':x['valid_surface_lidar_hits']} for x in frames],
    'AMP_label_changes':[x['changed_labels_vs_FP32'] for x in analysis.get('full_model_amp_control',{}).get('frames',[])]}))
