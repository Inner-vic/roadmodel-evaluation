import unittest
from pathlib import Path
from tests.test_two_link_runtime import directory
from pcm_eval.publication import publish_report


class PublicationTests(unittest.TestCase):
    def test_create_readback_and_repeat_does_not_recreate(self):
        with directory() as path:
            xml=path/'report.xml';xml.write_text('<title>模型单次报告</title><h1>精度结果</h1><p>本次2帧</p>',encoding='utf-8')
            calls=[]
            def invoke(args):
                calls.append(args)
                if args[1]=='+create':return {'document':{'document_id':'doc_test','url':'https://li.feishu.cn/docx/doc_test'}}
                return {'document':{'content':xml.read_text(encoding='utf-8')}}
            receipt=publish_report(xml,path/'publication',invoke=invoke)
            self.assertEqual('SUCCEEDED',receipt['state'])
            self.assertEqual(receipt,publish_report(xml,path/'publication',invoke=invoke))
            self.assertEqual(2,len(calls))

    def test_unknown_create_outcome_is_not_retried(self):
        with directory() as path:
            xml=path/'report.xml';xml.write_text('<title>报告</title><p>2帧</p>',encoding='utf-8')
            def fail(args):raise TimeoutError('unknown after submission')
            with self.assertRaises(TimeoutError):publish_report(xml,path/'publication',invoke=fail)
            with self.assertRaises(ValueError):publish_report(xml,path/'publication',invoke=lambda *_:self.fail('must not create again'))

    def test_resource_paths_are_made_portable_before_create(self):
        with directory() as path:
            asset=path/'data.json';asset.write_text('{"frames":2}')
            xml=path/'report.xml';xml.write_text('<title>报告</title><source path="@./data.json" name="原始结果"/>',encoding='utf-8')
            def invoke(args):
                if args[1]=='+create':
                    self.assertTrue(list((path/'publication/assets').glob('*-data.json')))
                    self.assertTrue((path/'publication/import.xml').exists())
                    return {'document':{'document_id':'doc_test'}}
                return {'document':{'content':'<title>报告</title><source token="asset_token"/>'}}
            self.assertEqual('SUCCEEDED',publish_report(xml,path/'publication',invoke=invoke)['state'])

    def test_missing_metric_table_is_not_reported_as_published_success(self):
        with directory() as path:
            xml=path/'report.xml';xml.write_text('<title>报告</title><table><tbody><tr><td>0.5</td></tr></tbody></table>',encoding='utf-8')
            def invoke(args):
                return {'document':{'document_id':'doc_test'}} if args[1]=='+create' else {'document':{'content':'<title>报告</title>'}}
            with self.assertRaises(ValueError):publish_report(xml,path/'publication',invoke=invoke)
