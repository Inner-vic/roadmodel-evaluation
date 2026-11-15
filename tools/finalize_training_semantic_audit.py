"""Join original logs, validated TensorBoard scalars and checkpoint metadata."""
import csv,hashlib,json,statistics
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
a=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
tb=json.loads((root/'tensorboard-scalars.json').read_text(encoding='utf-8'));s=tb['scalar_series']
def val(tag,e):
    values=[x['value'] for x in s[tag] if x['step']==e];assert len(values)==1;return values[0]
for r in a['epochs']:
    e=r['log_epoch'];assert r['checkpoint_epoch']==e+1
    for name,cls in r['classes'].items():
        recall=val('Validation/Traj_cls_'+name,e)
        assert abs(recall-cls['recall_percent'])<=.05001
        assert val('Validation/Traj_cls_count_'+name,e)==cls['support']
        cls['tensorboard_recall_percent']=recall
    acc=val('Validation/Traj_label_acc',e);assert abs(acc-r['validation_accuracy_percent'])<=.00501
    assert val('Validation/Traj_valid_points',e)==r['validation_valid_points']
    r['tensorboard_validation_accuracy_percent']=acc
    r['sampled_training_scalars']={}
    for name in ['label_ce','label_acc']:
        vs=[x['value'] for x in s['surface_patch/'+name] if e*523<=x['step']<(e+1)*523]
        assert len(vs) in [52,53]
        r['sampled_training_scalars'][name]={'n':len(vs),'mean':statistics.mean(vs),'min':min(vs),'max':max(vs),'sampling':'master process, every 10 iterations; unweighted mean of retained batch scalars'}
    r['historical_validation_loss']=val('Validation/Loss',e);assert r['historical_validation_loss']==0
    cement=r['classes']['cement']
    r['cement_share_of_errors_percent']=(cement['support']*(1-cement['tensorboard_recall_percent']/100))/(r['validation_valid_points']*(1-acc/100))*100
meta=a['checkpoint'];assert meta['epoch']==44 and meta['global_iter']==44*523
best=meta['best_traj'];assert best['epoch']==27
assert abs(best['label_acc']-val('Validation/Traj_label_acc',27))<.0001
assert abs(best['per_class_recall']['2']*100-val('Validation/Traj_cls_cement',27))<.0001
selected=a['epochs'][43]
a.update(tensorboard={'tags':len(tb['tags']),'records':tb['record_count'],'all_record_checksums_verified':tb['all_tfrecord_checksums_verified'],'semantic_train_scalar_samples':len(s['surface_patch/label_ce']),'validation_epochs':60,'training_cement_histogram_present':False,'validation_cement_ce_present':False,'full_confusion_matrix_present':False},
    findings={'cement_recall_min_percent':min(r['classes']['cement']['tensorboard_recall_percent'] for r in a['epochs']),'cement_recall_max_percent':max(r['classes']['cement']['tensorboard_recall_percent'] for r in a['epochs']),'release_epoch':44,'release_cement_recall_percent':selected['classes']['cement']['tensorboard_recall_percent'],'release_asphalt_recall_percent':selected['classes']['asphalt']['tensorboard_recall_percent'],'cement_share_of_validation_errors_percent':selected['cement_share_of_errors_percent'],'validation_semantic_loss_status':'not_computed; inference omits surface_pred/target, loss forward returns zero before CE','training_semantic_loss_status':'sampled aggregate across all valid lateral surface tokens; no cement-specific decomposition','selection_semantics':'front geometry/stability gates primary; aggregate label accuracy only last tie-break; no cement-recall gate'})
(root/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (root/'epoch-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['checkpoint_epoch','log_epoch','val_label_acc_percent','asphalt_recall_percent','cement_recall_percent','cement_support','training_sampled_batches','training_sampled_ce_mean','training_sampled_acc_percent','validation_ce_status'])
    for r in a['epochs']:
        tr=r['sampled_training_scalars'];w.writerow([r['checkpoint_epoch'],r['log_epoch'],r['tensorboard_validation_accuracy_percent'],r['classes']['asphalt']['tensorboard_recall_percent'],r['classes']['cement']['tensorboard_recall_percent'],r['classes']['cement']['support'],tr['label_ce']['n'],tr['label_ce']['mean'],tr['label_acc']['mean']*100,'not_computed'])
base=Path('inspection/release-model-complete/9e16e29351d17a726ce6beca6f821de79b7ade40')
specs=[('loss/surface_patch_ddpm_loss.py',753,758),('loss/surface_patch_ddpm_loss.py',795,848),('loss/surface_patch_ddpm_loss.py',1040,1069),('train.py',280,432),('train.py',1642,1675),('model/head/traj_head/traj_surface_front_dit_ddpm_head.py',1244,1295),('model/head/traj_head/traj_surface_dense_ddpm_head.py',1556,1566),('misc/traj_eval_utils.py',450,501)]
snippets=[]
for p,start,end in specs:
    raw=(base/p).read_bytes();text=raw.decode('utf-8').splitlines()
    snippets.append({'path':p,'sha256':hashlib.sha256(raw).hexdigest(),'start':start,'end':end,'code':'\n'.join(f'{i}: {text[i-1]}' for i in range(start,end+1))})
(root/'source-evidence.json').write_text(json.dumps(snippets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'findings':a['findings'],'release_epoch_train':selected['sampled_training_scalars'],'checks':'60 epochs logs/scalars consistent; checkpoint iteration and best-traj metadata cross-checked'},ensure_ascii=False,indent=2))
