"""Offline extraction from retained original-task logs; no model execution."""
import ast,csv,io,json,re,zipfile,statistics
from pathlib import Path
ROOT=Path('evidence/model-evaluation/training-semantic-audit-20261027')
lines=(ROOT/'semantic.log').read_text(encoding='utf-8').splitlines()
# Retain algorithm logging only; exclude platform initialization/configuration payloads.
lines=[x for x in lines if re.match(r'\d\d/\d\d \d\d:\d\d:\d\d - mmengine - ',x)]
(ROOT/'algorithm.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
epochs={};epoch=None;iteration=None;phase=None
for n,line in enumerate(lines,1):
    m=re.search(r'\[(TRAIN|EVAL)\] Epoch (\d+) Iter\s+(\d+)',line)
    if m:phase,epoch,iteration=m[1],int(m[2]),int(m[3])
    if epoch is None:continue
    row=epochs.setdefault(epoch,{'log_epoch':epoch,'checkpoint_epoch':epoch+1,'train_samples':[]})
    if '/surface_label_ce:' in line:
        stats={k:float(v) for k,v in re.findall(r'SurfacePatchDDPMLoss/(\w+): ([-+\deE.]+)',line)}
        row['train_samples'].append({'iteration':iteration,'log_line':n,**stats})
    if '[Traj-DP3] per-class:' in line:
        row['classes']={name:{'id':int(cid),'recall_percent':float(acc),'support':int(count)} for name,cid,acc,count in re.findall(r'(\w+)\((\d+)\): ([\d.]+)%\((\d+)\)',line)}
    if '[Traj-DP3] label_acc=' in line:
        row['validation_accuracy_percent']=float(re.search(r'label_acc=([\d.]+)%',line)[1])
        row['validation_valid_points']=int(re.search(r'\(n=(\d+)\)',line)[1])
    if '[Traj-Best] candidate' in line:row['selection']=line.split('INFO - ',1)[1]
for row in epochs.values():
    values=row['train_samples']
    row['logged_train_batch_count']=len(values)
    for k in ['surface_label_ce','surface_label_acc']:
        vals=[v[k] for v in values if k in v]
        if vals:row[k]={'min':min(vals),'max':max(vals),'mean':statistics.mean(vals),'median':statistics.median(vals)}
    if 'classes' in row:
        total=row['validation_valid_points'];row['validation_class_share_percent']={k:100*v['support']/total for k,v in row['classes'].items()}
assert sorted(epochs)==list(range(60))
assert all('cement' in x['classes'] for x in epochs.values())
src=Path('tools/inspect_checkpoint_metadata_only.py').read_text(encoding='utf-8');tree=ast.parse(src)
tree.body=tree.body[:next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='root' for t in n.targets))]
scope={};exec(compile(tree,'metadata_definitions','exec'),scope)
pth=Path('evidence/model-evaluation/model-reference-20261027/weights/26-09-08-2/epoch_44.pth')
with zipfile.ZipFile(pth) as z:
    pkl=next(n for n in z.namelist() if n.endswith('/data.pkl'))
    ckpt=scope['MetadataOnlyUnpickler'](io.BytesIO(z.read(pkl))).load()
binding={k:ckpt[k] for k in ['epoch','global_iter','best_traj']}
result={'schema':'roadmodel.training-semantic-audit/v1','task_uid':'8d226ad5-e769-411b-9c72-57f9dd822c0e','checkpoint':binding,'epochs':list(epochs.values()),'new_training':False,'new_inference_frames':0,'support_unit':'historical distributed validation trajectory points, not scenes or unique frames'}
(ROOT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'epoch-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['checkpoint_epoch','val_label_acc_percent','asphalt_recall_percent','cement_recall_percent','cement_support','logged_train_batches','logged_train_ce_mean','logged_train_acc_mean'])
    for r in epochs.values():w.writerow([r['checkpoint_epoch'],r['validation_accuracy_percent'],r['classes']['asphalt']['recall_percent'],r['classes']['cement']['recall_percent'],r['classes']['cement']['support'],r['logged_train_batch_count'],r['surface_label_ce']['mean'],r['surface_label_acc']['mean']])
print(json.dumps({'checkpoint':binding,'rows':[{k:v for k,v in epochs[i].items() if k!='train_samples'} for i in [0,19,29,43,59]],'cement_recall_range':[min(x['classes']['cement']['recall_percent'] for x in epochs.values()),max(x['classes']['cement']['recall_percent'] for x in epochs.values())]},ensure_ascii=False,indent=2))
