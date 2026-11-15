import unittest
from pcm_eval.batch_schedule import groups


class ScheduleTests(unittest.TestCase):
    def test_tail_and_retained_frames_are_never_reinferred(self):
        value=list(groups(range(11),4,retained=[2,7]))
        self.assertEqual([i for g in value for i in g['indices']],list(range(11)))
        self.assertEqual([g['indices'] for g in value if g['mode']=='adopt'],[[2],[7]])
        self.assertTrue(all(len(g['indices']) in (1,2,4) for g in value))
        self.assertFalse(any(i in (2,7) for g in value if g['mode']=='infer' for i in g['indices']))

    def test_duplicate_or_unordered_indices_are_rejected(self):
        for bad in ([1,1],[2,1],[-1],[True]):
            with self.assertRaises(ValueError):list(groups(bad,2))

    def test_eight_plus_odd_tail(self):
        self.assertEqual([len(g['indices']) for g in groups(range(15),8)],[8,4,2,1])


if __name__=='__main__':unittest.main()
