"""Explicit V161 sparse OCC -> compiled ROI adaptation and count-based metrics.

NumPy is only required on the evaluation worker, not for collecting reports.
"""
import numpy as np

LABELS = ('ignore asphalt cement epoxy_floor brick_road gravel cobbles grass dirt '
          'hard_sand soft_sand water ice snow other_surface road_noise bumper '
          'manhole_cover steel_plate drain road_beam railway step road_joint '
          'moving_obstacle fixed_obstacle empty').split()
CLASS_IDS = [i for i in range(26) if i not in (0, 15)]
DRIVABLE_IDS = list(range(1, 15)) + list(range(16, 24))


def adapt_sparse(sparse, source_min, voxel_size, target_min, target_shape):
    a = np.asarray(sparse)
    if a.ndim != 2 or a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError('Expected finite V161 sparse [ix,iy,iz,class,drivable]')
    if not np.equal(a, np.rint(a)).all():
        raise ValueError('Sparse indices and labels must be integers')
    a = a.astype(np.int64)
    if (a[:, :3] < 0).any() or not np.isin(a[:, 3], range(26)).all() or not np.isin(a[:, 4], [0, 1]).all():
        raise ValueError('Invalid GT index, class or drivable flag')
    shift = (np.asarray(source_min) - np.asarray(target_min)) / np.asarray(voxel_size)
    if not np.allclose(shift, np.rint(shift), atol=1e-7):
        raise ValueError('Grid origins must align to integer voxels; resampling is not implicit')
    idx = a[:, :3] + np.rint(shift).astype(np.int64)
    keep = ((idx >= 0) & (idx < np.asarray(target_shape))).all(axis=1)
    idx, labels = idx[keep], a[keep, 3:]
    if len(np.unique(idx, axis=0)) != len(idx):
        raise ValueError('Duplicate GT voxels require an explicit conflict policy')
    gt = np.full(target_shape, 26, dtype=np.int32)
    drive = np.zeros(target_shape, dtype=np.uint8)
    gt[tuple(idx.T)] = labels[:, 0]
    drive[tuple(idx.T)] = labels[:, 1]
    mask = ~np.isin(gt, [0, 15])
    return gt, mask, drive, {'source_rows':len(a), 'kept_rows':int(keep.sum()),
                            'cropped_rows':int((~keep).sum()), 'index_shift':np.rint(shift).astype(int).tolist()}


def check_height_geometry(sparse, height, source_min, voxel_size):
    """Independent physical-coordinate cross-check, not derived from predictions."""
    a, h = np.asarray(sparse), np.asarray(height)
    if h.ndim != 2 or h.shape[1] < 3 or not len(h) or not np.isfinite(h[:, :3]).all():
        raise ValueError('Height-field geometry evidence is missing or invalid')
    hi = np.floor((h[:, :3] - np.asarray(source_min)) / np.asarray(voxel_size) + 1e-5).astype(np.int64)
    # Source is 800 x 48 x 200. Encoding is unambiguous, including neighbor z.
    shape = np.asarray([800,48,200]); inside = ((hi >= 0) & (hi < shape)).all(axis=1)
    if inside.mean() < .99:
        raise ValueError('Physical GT points fall outside the source grid')
    ai = a[:, :3].astype(np.int64)
    if not ((ai >= 0) & (ai < shape)).all():
        raise ValueError('Sparse GT indices exceed the declared source geometry')
    keys = (ai[:,0]*48 + ai[:,1])*200 + ai[:,2]
    base = (hi[:,0]*48 + hi[:,1])*200
    exact = inside & np.isin(base+hi[:,2], keys)
    near = exact.copy()
    for delta in (-1,1):
        near |= inside & (hi[:,2]+delta >= 0) & (hi[:,2]+delta < 200) & np.isin(base+hi[:,2]+delta, keys)
    fraction = float(near.mean())
    if fraction < .98:
        raise ValueError(f'Height/OCC physical alignment failed: {fraction:.6f}')
    return {'physical_points':len(h),'exact_fraction':float(exact.mean()),
            'within_one_z_voxel_fraction':fraction,'required_fraction':.98}


class Counts:
    def __init__(self):
        self.confusion = np.zeros((27,27), dtype=np.int64)
        self.ds = np.zeros(3, dtype=np.int64)  # seen, correct, positive
        self.num_frames = 0

    def update(self, pred, gt, mask, drive):
        pred = np.asarray(pred)
        if pred.shape != gt.shape or not np.isfinite(pred).all() or not np.equal(pred,np.rint(pred)).all() or not np.isin(pred,range(27)).all():
            raise ValueError('Prediction shape, class IDs or finite values are invalid')
        pred = pred.astype(np.int64)
        self.confusion += np.bincount(gt[mask].astype(np.int64)*27+pred[mask], minlength=729).reshape(27,27)
        # Exact metric_util.MeanIoU.single_layer_ds_count(probs=None) branch.
        # Counts all predicted drivable voxels, selects the first drivable Z for TP.
        pd = np.isin(pred,DRIVABLE_IDS).reshape(-1,gt.shape[-1])
        gd = (drive == 1).reshape(pd.shape)
        z = pd.argmax(axis=-1); xy = np.arange(len(pd))
        self.ds += [int(gd.sum()),int((pd[xy,z]&gd[xy,z]).sum()),int(pd.sum())]
        self.num_frames += 1

    def summary(self):
        seen=self.confusion.sum(1); positive=self.confusion.sum(0); correct=self.confusion.diagonal()
        rows=[]
        for c in CLASS_IDS:
            den=int(seen[c]+positive[c]-correct[c])
            rows.append({'class_id':c,'name':LABELS[c], 'seen':int(seen[c]),'positive':int(positive[c]),
                         'correct':int(correct[c]),'iou':float(correct[c]/den*100) if seen[c]>0 else None,
                         'status':'available' if seen[c]>0 else 'no_gt_samples'})
        present=[r for r in rows if r['seen']>0]
        ne_seen=int(seen[:26].sum());ne_positive=int(positive[:26].sum());ne_correct=int(self.confusion[:26,:26].sum())
        s,c,p=map(int,self.ds)
        return {'num_frames':self.num_frames,'present_miou':float(np.mean([r['iou'] for r in present])) if present else None,
                'occ_iou':ne_correct/max(1,ne_seen+ne_positive-ne_correct)*100,
                'drivable_iou':c/max(1,s+p-c)*100 if s else None,
                'drivable_counts':{'seen':s,'correct':c,'positive':p},
                'drivable_definition':'MeanIoU.single_layer_ds_count(probs=None); no probability tensor retained',
                'present_class_ious':present,'class_support':rows,'HWD':[400,48,100],'empty_label':26}
