"""Bounded native-model GT preparation, independent of report generation."""
import numpy as np
from .gt_adapter import adapt_sparse


def native_sparse_gt(sparse, contract):
    """Keep physical voxels unchanged while shifting indices into the model ROI."""
    a=np.asarray(sparse)
    _,_,_,receipt=adapt_sparse(a,contract['source_min'],contract['voxel_size'],
                              contract['target_min'],contract['target_shape'])
    source_shape=np.asarray(contract['source_shape'])
    if not ((a[:,:3]>=0)&(a[:,:3]<source_shape)).all():
        raise ValueError('GT indices exceed the declared source grid')
    shifted=a[:,:3].astype(np.int64)+np.asarray(receipt['index_shift'])
    keep=((shifted>=0)&(shifted<np.asarray(contract['target_shape']))).all(1)
    out=a[keep].copy()
    out[:,:3]=shifted[keep]
    source_xyz=np.asarray(contract['source_min'])+a[keep,:3]*np.asarray(contract['voxel_size'])
    target_xyz=np.asarray(contract['target_min'])+out[:,:3]*np.asarray(contract['voxel_size'])
    if not np.allclose(source_xyz,target_xyz,atol=1e-7,rtol=0):
        raise ValueError('Physical GT positions changed during adaptation')
    return out,dict(receipt,physical_positions_preserved=True,labels_and_drivable_preserved=True)
