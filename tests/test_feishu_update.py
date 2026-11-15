from pathlib import Path
import unittest
from pcm_eval.common import save_json
from pcm_eval.feishu_update import update_report
from test_retained_runtime import directory

def xml(value,resource=True):
    return '<title>T</title><p id="p0">scope</p>'+''.join('<table id="t'+str(i)+'"><tr><th>H</th></tr><tr><td>'+value+'</td></tr></table>' for i in range(5))+('<source id="old-file" name="history.zip"/>' if resource else '')

class PublishTests(unittest.TestCase):
    def test_table_drift_stops_before_any_write(self):
        with directory() as d:
            (d/'report.xml').write_text(xml('new'));save_json(d/'data.json',{})
            writes=[]
            def call(args):
                if '+fetch' in args:return {'document':{'content':xml('user-changed'),'revision_id':8}}
                writes.append(args);raise AssertionError('No writes allowed')
            with self.assertRaises(ValueError):update_report(d/'report.xml',d/'data.json',d/'pub',document_id='doc',native_task_id='job',frozen_requests={},scope='full',invoke=call)
            self.assertEqual([],writes)

    def test_unknown_write_is_not_repeated_by_retry(self):
        with directory() as d:
            (d/'report.xml').write_text(xml('same'));save_json(d/'data.json',{})
            calls=[]
            def call(args):
                calls.append(args)
                if '+fetch' in args:return {'document':{'content':xml('same'),'revision_id':8}}
                raise TimeoutError('write outcome unknown')
            with self.assertRaises(TimeoutError):update_report(d/'report.xml',d/'data.json',d/'pub',document_id='doc',native_task_id='job',frozen_requests={},scope='full',invoke=call)
            before=len(calls)
            with self.assertRaises(ValueError):update_report(d/'report.xml',d/'data.json',d/'pub',document_id='doc',native_task_id='job',frozen_requests={},scope='full',invoke=call)
            self.assertEqual(before,len(calls))

    def test_preserved_resources_and_exact_tables_are_required(self):
        with directory() as d:
            (d/'report.xml').write_text(xml('same'));save_json(d/'data.json',{})
            count=0
            def call(args):
                nonlocal count
                if '+fetch' in args:
                    count+=1
                    return {'document':{'content':xml('same',resource=count==1),'revision_id':8+count}}
                return {'result':'success'}
            with self.assertRaisesRegex(ValueError,'attachment disappeared'):
                update_report(d/'report.xml',d/'data.json',d/'pub',document_id='doc',native_task_id='job',frozen_requests={},scope='full',invoke=call)

if __name__=='__main__':unittest.main()
