import unittest
import numpy as np
from pcm_eval.model_dataset import native_sparse_gt

CONTRACT={'source_min':[5,-3.6,-2],'source_shape':[800,48,200],
          'voxel_size':[.05,.15,.02],'target_min':[5,-3.6,-1],'target_shape':[400,48,100]}

class NativeGTTests(unittest.TestCase):
 def test_physical_crop_and_task_labels(self):
  a=np.array([[0,0,50,1,1],[399,47,149,25,0],[400,0,50,1,1],
              [0,0,49,2,1],[0,0,150,2,1]],dtype=np.int32)
  out,receipt=native_sparse_gt(a,CONTRACT)
  np.testing.assert_array_equal(out,[[0,0,0,1,1],[399,47,99,25,0]])
  self.assertEqual(receipt['cropped_rows'],3)
  self.assertTrue(receipt['physical_positions_preserved'])
  np.testing.assert_array_equal(a[:,2],[50,149,50,49,150])
 def test_source_grid_violation_is_rejected(self):
  with self.assertRaises(ValueError):native_sparse_gt([[800,0,50,1,1]],CONTRACT)
 def test_invalid_label_is_rejected(self):
  with self.assertRaises(ValueError):native_sparse_gt([[0,0,50,27,1]],CONTRACT)
