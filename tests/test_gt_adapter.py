import unittest
import numpy as np
from pcm_eval.gt_adapter import Counts, adapt_sparse
from pcm_eval.single_report import gt_cell, display


class GTAdapterTests(unittest.TestCase):
    def test_physical_crop_shift_boundary_and_fifth_column(self):
        sparse=np.array([[0,0,49,1,1],[0,0,50,16,1],[399,47,149,25,0],
                         [400,0,70,1,1],[2,1,150,2,0],[1,1,51,15,0]])
        gt,mask,ds,receipt=adapt_sparse(sparse,[5,-3.6,-2],[.05,.15,.02],[5,-3.6,-1],(400,48,100))
        self.assertEqual(receipt['index_shift'],[0,0,-50])
        self.assertEqual(receipt['kept_rows'],3)
        self.assertEqual(gt[0,0,0],16)
        self.assertEqual(gt[399,47,99],25)
        self.assertEqual(ds[0,0,0],1)
        self.assertFalse(mask[1,1,1])
        self.assertEqual(gt[1,0,0],26)
        self.assertTrue(mask[1,0,0])

    def test_metrics_against_hand_counts_and_drivable_is_not_nonempty(self):
        gt=np.array([[[1,26,26],[24,26,26]]])
        pred=np.array([[[1,2,26],[24,26,26]]])
        drive=np.array([[[1,0,0],[0,0,0]]])
        c=Counts();c.update(pred,gt,np.ones_like(gt,dtype=bool),drive)
        summary=c.summary()
        self.assertAlmostEqual(summary['drivable_iou'],50)
        self.assertAlmostEqual(summary['occ_iou'],200/3)
        self.assertAlmostEqual(summary['present_miou'],100)
        self.assertEqual(summary['drivable_counts'],{'seen':1,'correct':1,'positive':2})
        self.assertEqual(next(r for r in summary['class_support'] if r['name']=='cement')['status'],'no_gt_samples')

    def test_fail_closed_for_duplicate_indices_noninteger_and_misaligned_grid(self):
        for sparse,origin in [(np.array([[0,0,0,1,1],[0,0,0,2,0]]),[0,0,0]),
                              (np.array([[0,0,.5,1,1]]),[0,0,0]),
                              (np.array([[0,0,0,1,1]]),[.5,0,0])]:
            with self.assertRaises(ValueError):
                adapt_sparse(sparse,origin,[1,1,1],[0,0,0],(2,2,2))

    def test_missing_class_is_distinct_from_uncomputed_and_zero_score(self):
        result={'provenance':{'gt_class_support':{'board':[{'name':'water','seen':0,'correct':0,'positive':12,'status':'no_gt_samples'}]}}}
        value=gt_cell(result,'board','water')
        self.assertEqual(value['status'],'no_gt_samples')
        self.assertEqual(value['support']['positive'],12)
        self.assertEqual(display(value),'无GT样本')
        self.assertEqual(gt_cell(result,'board','asphalt')['status'],'missing')

if __name__=='__main__':unittest.main()
