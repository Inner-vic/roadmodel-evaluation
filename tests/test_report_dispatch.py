from pathlib import Path
from types import SimpleNamespace
import sys,threading
import unittest
from pcm_eval.common import save_json,digest
from pcm_eval.report_dispatch import start_report_pipeline
from test_retained_runtime import directory

class ReportDispatchTests(unittest.TestCase):
    def binding(self,d):
        p=d/'pipeline.py';p.write_text('print("report")\n')
        c=save_json(d/'pipeline.json',{'report':{'native_task_id':'native','frozen_requests':{'model_accuracy':'frozen'}}})
        return {'kind':'model_accuracy','native_task_id':'native','frozen_request':{'sha256':'frozen'},'report_pipeline':{
            'config':str(c),'config_sha256':digest(c),'script':str(p),'script_sha256':digest(p),'dispatch_receipt':str(d/'dispatch.json'),
            'python':sys.executable,'cwd':str(d)}}

    def test_two_atomic_chains_start_one_report_coordinator(self):
        with directory() as d:
            b=self.binding(d);seen=[];answers=[]
            def spawn(*a,**kw):seen.append(a);return SimpleNamespace(pid=123)
            threads=[threading.Thread(target=lambda:answers.append(start_report_pipeline(b,spawn=spawn))) for _ in range(2)]
            for t in threads:t.start()
            for t in threads:t.join()
            self.assertEqual(1,len(seen));self.assertEqual(2,len(answers))

    def test_different_scope_cannot_reuse_current_publisher(self):
        with directory() as d:
            b=self.binding(d);b['frozen_request']['sha256']='different'
            with self.assertRaises(ValueError):start_report_pipeline(b,spawn=lambda *a,**kw:self.fail('must not launch'))

    def test_unknown_start_never_automatically_restarts_publisher(self):
        with directory() as d:
            b=self.binding(d)
            def fail(*a,**kw):raise OSError('unknown')
            with self.assertRaises(OSError):start_report_pipeline(b,spawn=fail)
            self.assertEqual('NEEDS_REVIEW',start_report_pipeline(b,spawn=lambda *a,**kw:self.fail('no retry'))['state'])

if __name__=='__main__':unittest.main()
