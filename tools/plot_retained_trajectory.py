"""Plot retained pointwise curves without smoothing, interpolation or inference."""
import json,hashlib,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('evidence/model-evaluation/remaining-heads-20261027/collected/trajectory-01-recovered')
out=Path('evidence/model-evaluation/remaining-heads-20261027/figures');out.mkdir(exist_ok=True)
meta=json.loads((root/'trajectory_metrics.json').read_text());zmin=meta['pc_range'][2];dz=meta['voxel_size'][2];bins=meta['z_bins']
raw=[];distribution=[]
for i in range(meta['num_frames']):
    with np.load(root/f'frame-{i}.npz',allow_pickle=False) as a:
        x=a['traj_xy_used'].reshape(-1,2)[:,0];pred=a['traj_z'].reshape(-1);gt=a['dp3_gt_traj_z_norm'].reshape(-1)*(bins-1)
        gl=a['dp3_gt_traj_label'].reshape(-1);pl=a['traj_label'].reshape(-1)
        valid=a['traj_valid_mask'].reshape(-1).astype(bool)&(gl!=26)
        assert len(x)%4==0
        pred_cm=(zmin+pred*dz)*100;gt_cm=(zmin+gt*dz)*100
        raw.append((x,np.where(valid,gt_cm,np.nan),np.where(valid,pred_cm,np.nan),valid))
        labels,counts=np.unique(pl[valid],return_counts=True)
        distribution.append({'frame':meta['frames'][i]['frame_index'],'valid_points':int(valid.sum()),
            'valid_points_per_wheel':valid.reshape(4,-1).sum(axis=1).astype(int).tolist(),
            'predicted_label_counts':dict(zip(labels.astype(int).tolist(),counts.tolist())),
            'gt_label_counts':dict(zip(*[v.tolist() for v in np.unique(gl[valid],return_counts=True)]))})
with (out/'trajectory-curves.csv').open('w',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['frame','wheel','slot','x_m','gt_z_cm','pred_z_cm','valid'])
    for i,(x,gt,pred,mask) in enumerate(raw):
        slots=len(x)//4
        for j in range(len(x)):writer.writerow([meta['frames'][i]['frame_index'],j//slots,j%slots,float(x[j]),float(gt[j]) if mask[j] else '',float(pred[j]) if mask[j] else '',int(mask[j])])
with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'}):
    fig,axes=plt.subplots(2,2,figsize=(12,6.5),sharex=True,sharey=True,layout='constrained')
    for col,(x,gt,pred,mask) in enumerate(raw):
        n=len(x)//4
        for wheel,label in enumerate(['Front-left','Front-right']):
            ax=axes[wheel,col];sl=slice(wheel*n,(wheel+1)*n);xx=np.where(mask[sl],x[sl],np.nan)
            ax.plot(xx,gt[sl],color='#222222',linestyle='-',linewidth=1.8,label='Mesh GT')
            ax.plot(xx,pred[sl],color='#0072B2',linestyle='--',linewidth=1.4,label='Model prediction')
            ax.set_title(f'Frame {meta["frames"][col]["frame_index"]} / {label}')
            ax.grid(alpha=.22);ax.set_xlim(5,25)
            if col==0:ax.set_ylabel('VCS height (cm)')
            if wheel==1:ax.set_xlabel('Forward distance x (m)')
            if wheel==0:ax.legend(loc='best',frameon=False)
    fig.suptitle('Trajectory head: front-wheel height predictions vs Mesh GT\nRear wheels are masked by the native front-only head and are not scored',fontsize=13)
    fig.savefig(out/'trajectory-height-curves.png',dpi=180,facecolor='white',transparent=False);plt.close(fig)
receipt={'source_run_id':meta['source_inference_run_id'],'frames':distribution,
    'source_files':[{'path':str(root/f'frame-{i}.npz'),'sha256':hashlib.sha256((root/f'frame-{i}.npz').read_bytes()).hexdigest()} for i in range(meta['num_frames'])],
    'transformations':['height_cm=(z_min+z_bin*0.02)*100','GT z_bin=gt_z_norm*(100-1)',
        'invalid points retained as gaps; no smoothing or interpolation','front wheels only; native front-only head masks rear wheels','shared y limits across all wheel/frame panels'],
    'source_data':str(out/'trajectory-curves.csv'),'image':str(out/'trajectory-height-curves.png'),
    'matplotlib_version':matplotlib.__version__,'new_inference_frames':0}
(out/'figure-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
