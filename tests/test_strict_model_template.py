import unittest
import xml.etree.ElementTree as ET

from pcm_eval.strict_model_template import render


class StrictModelTemplateTest(unittest.TestCase):
    def setUp(self):
        self.reference = ('<title>PCM</title><blockquote><p>模型版本：v3.2.2</p></blockquote>'
            '<h1>开发空间模型评测</h1><h2>occ头</h2>'
            '<table><tr><th>类别</th><th>IoU</th></tr><tr><td/><td>旧版/新版</td></tr>'
            '<tr><td>沥青</td><td>旧值</td></tr></table>'
            '<blockquote><p>v3.0.4训练与v3.1.0训练对比</p></blockquote>'
            '<h2>轨迹头</h2><blockquote><p>该版本默认外发occ头</p></blockquote>'
            '<h2>视觉头</h2><table><tr><th>类别</th><th>mAP</th></tr>'
            '<tr><td>雨天</td><td>旧值</td></tr></table>'
            '<ul><li>参与 mAP 的子类明细（major 子类，Total samples = 11550）</li></ul>'
            '<table><tr><th>父类</th><th>子类</th><th>AP</th></tr>'
            '<tr><td>雨天</td><td>无雨</td><td>旧值</td></tr></table>'
            '<h1>板端产物与fp16</h1><p>历史编译结果</p>')
        def c(value):return {'value': value, 'state': 'measured', 'unit': None}
        self.tables = [
            {'id': 'occ', 'headers': ['类别', 'IoU'], 'rows': [[c('沥青'), c(42)]]},
            {'id': 'vision_summary', 'headers': ['类别', 'mAP'], 'rows': [[c('雨天'), c(.9)]]},
            {'id': 'vision_classes', 'headers': ['父类', '子类', 'AP'], 'rows': [[c('雨天'), c('无雨'), c(.9)]]},
        ]

    def test_only_original_model_blocks_and_current_values(self):
        xml = render(self.reference, self.tables, model_version='v3.2.2', vision_samples=11543)
        root = ET.fromstring('<root>'+xml+'</root>')
        self.assertEqual([n.tag for n in root], ['blockquote','h1','h2','table','h2','blockquote','h2','table','ul','table'])
        self.assertIn('Total samples = 11543', xml)
        self.assertNotIn('v3.0.4', xml)
        self.assertNotIn('历史编译结果', xml)
        self.assertNotIn('运行', xml)
        self.assertNotIn('数据集', xml)
        self.assertIn('<p>42</p>', xml)

    def test_reject_template_drift_instead_of_guessing(self):
        self.tables[0]['headers'][1] = '自定义评分'
        with self.assertRaises(ValueError):
            render(self.reference, self.tables, model_version='v3.2.2', vision_samples=11543)


if __name__ == '__main__':unittest.main()
