import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pcm_eval.common import json_digest, save_json
from pcm_eval.delivery_demo import DemoRuntime
from pcm_eval.delivery_demo_entry import render
from pcm_eval.planning import ssh_command, validate_ssh_options
from tools.delivery_demo_plugin import card_route

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://li.feishu.cn/record/PUBREF000000000000000000001'
DATA = 'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a?version=v1'


class DemoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = self.root/'config.json'
        save_json(self.cfg, {'agent_id': 'agent', 'owner': 'ou_owner', 'python': '/python'})
        self.rt = DemoRuntime(self.root/'state', config_file=self.cfg)
        self.owner = 'ou_owner'
        self.session = 'feishu:oc_test:ou_owner'

    def tearDown(self):
        self.temp.cleanup()

    def submit(self):
        task = self.rt.open(self.owner, self.session)
        self.rt.bind_message(task['id'], self.owner, 'om_card', self.session)
        event = {'action': 'roadmodel_submit', 'event_id': 'real-callback', 'message_id': 'om_card',
                 'operator_id': self.owner, 'chat_id': 'oc_test', 'value': {'task_id': task['id'], 'nonce': task['nonce']},
                 'form_value': {'model_delivery_url': URL, 'dataset_url': DATA}}
        self.rt.submit(self.owner, self.session, event)
        return task

    def workflow(self, task):
        value = {'schema': 'roadmodel.delivery-workflow/v1', 'scope': 'validation',
                 'inputs': {'model_delivery_url': URL, 'dataset_ref': DATA},
                 'jobs': [{'kind': k, 'ready': True, 'issues': []} for k in ('model_accuracy', 'compile_alignment')]}
        value['workflow_sha256'] = json_digest(value)
        return value

    def test_demo_card_is_explicit_about_scope_and_has_only_two_inputs(self):
        task = self.rt.open(self.owner, self.session)
        card = render(task)
        content = json.dumps(card, ensure_ascii=False)
        self.assertIn('模型OCC 2帧', content)
        self.assertIn('编译张量对齐1帧', content)
        self.assertIn('delivery-demo/v1', content)
        form = next(n for n in card['body']['elements'] if n.get('tag') == 'form')
        self.assertEqual([n['name'] for n in form['elements'] if n.get('tag') == 'input'], ['model_delivery_url', 'dataset_url'])

    def test_authenticated_callback_only_queues_material_reads(self):
        task = self.submit()
        self.rt.tick(spawn=lambda *a, **kw: self.fail('No inference before fresh material inspection'))
        self.assertEqual(self.rt.get(task['id'], self.owner)['state'], 'PREPARING')
        row = self.rt.claim_materials('agent', self.owner)
        self.assertEqual(row['task'], task['id'])
        self.assertEqual(json.loads(row['request']), {'model_delivery_url': URL, 'dataset_ref': DATA})

    def test_connector_without_entry_schema_routes_by_bound_card(self):
        root = self.root / 'gateway-state'
        demo = DemoRuntime(root / 'roadmodel' / 'delivery-demo', config_file=self.cfg)
        task = demo.open(self.owner, self.session)
        demo.bind_message(task['id'], self.owner, 'om_demo', self.session)
        event = {'operator_id': self.owner, 'message_id': 'om_demo',
                 'value': {'task_id': task['id'], 'nonce': task['nonce']}}
        self.assertEqual(card_route(root, self.session, event),
                         ('delivery_demo_entry', 'delivery-demo/v1'))
        self.assertIsNone(card_route(root, self.session, {**event, 'message_id': 'om_other'}))
        self.assertIsNone(card_route(root, 'feishu:oc_other:ou_owner', event))
        wrong_schema = {**event['value'], 'entry_schema': 'model-ticket/v2'}
        self.assertIsNone(card_route(root, self.session, {**event, 'value': wrong_schema}))

    def test_other_identity_cannot_claim_or_complete_material_reads(self):
        with self.assertRaises(ValueError):
            self.rt.claim_materials('other', self.owner)

    def test_source_input_cannot_be_changed_by_preparation(self):
        task = self.submit()
        self.rt.tick()
        row = self.rt.claim_materials('agent', self.owner)
        w = self.workflow(task)
        w['inputs']['dataset_ref'] = '/another'
        with self.assertRaises(ValueError):
            self.rt.complete_materials('agent', task['id'], row['digest'], {'workflow': w})

    def test_parse_failure_is_visible_and_never_dispatches(self):
        task = self.submit()
        self.rt.tick()
        row = self.rt.claim_materials('agent', self.owner)
        self.rt.complete_materials('agent', task['id'], row['digest'], {'error': '材料权限不足'})
        self.rt.tick(spawn=lambda *a, **kw: self.fail('No dispatch'))
        self.assertEqual(self.rt.get(task['id'], self.owner)['state'], 'BLOCKED')

    def test_prepared_workflow_dispatches_once_and_keeps_original_message(self):
        task = self.submit()
        self.rt.tick()
        row = self.rt.claim_materials('agent', self.owner)
        self.rt.complete_materials('agent', task['id'], row['digest'], {'workflow': self.workflow(task)})
        calls = []
        self.rt.tick(spawn=lambda *a, **kw: (calls.append(a) or SimpleNamespace(pid=999999999)))
        self.rt.tick(spawn=lambda *a, **kw: self.fail('No automatic duplicate'))
        current = self.rt.get(task['id'], self.owner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(current['message_id'], 'om_card')
        self.assertEqual(current['state'], 'NEEDS_REVIEW')

    def test_fixed_ssh_user_is_not_a_shell_fragment(self):
        validate_ssh_options({'user': 'root', 'port': 31017})
        for name in ('root; echo x', 'root@host', '-oProxyCommand=evil'):
            with self.assertRaises(ValueError):
                validate_ssh_options({'user': name})


if __name__ == '__main__':
    unittest.main()
