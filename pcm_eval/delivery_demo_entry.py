"""Small-scope Demo card, delivered through the existing authenticated connection."""
import hashlib
import json
from pathlib import Path
import time

from .delivery_demo import DemoRuntime
from .feishu_entry import render as render_base

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {'RoadModel Demo', 'RoadModel演示', 'RoadModel评测', 'RoadModel 评测'}


def runtime(cfg):
    return DemoRuntime(Path(cfg['state_root'])/'roadmodel/delivery-demo')


def render(task):
    display = dict(task)
    display['chains'] = [{**c, 'detail': {k: v for k, v in c['detail'].items() if k not in ('report_url', 'publication')}} for c in task['chains']]
    if task['state'] in ('PREPARING', 'READY'):
        display['state'] = 'RUNNING'
    card = render_base(display)
    card['header']['title']['content'] = 'RoadModel评测'
    text = ('自动识别交付材料，执行模型OCC 2帧、编译张量对齐1帧，汇总到同一份评测报告。'
            '当前先验证小样本，结果不代表完整验证集。')
    if task['state'] == 'PREPARING':
        text += '\n正在读取线上交付记录与核验数据。'
    card['body']['elements'][0]['content'] = text
    if task['state'] == 'DRAFT':
        form = next(e for e in card['body']['elements'] if e.get('tag') == 'form')
        form['elements'][-1]['text']['content'] = '开始评测'
        form['elements'][-1]['behaviors'][0]['value']['entry_schema'] = 'delivery-demo/v1'
    if task.get('report_url'):
        card['body']['elements'].append({'tag': 'button', 'type': 'primary_filled', 'width': 'fill',
            'text': {'tag': 'plain_text', 'content': '查看评测报告'},
            'behaviors': [{'type': 'open_url', 'default_url': task['report_url']}]})
    return card


def handle(service, session, text, event=None):
    from bench_gateway.service import identity
    from bench_gateway.card_entities import send, update
    from bench_gateway.task_cards import _client
    owner = identity(session)
    rt = runtime(service.cfg)
    if event:
        if event.get('value', {}).get('entry_schema') != 'delivery-demo/v1':
            raise ValueError('Demo card schema differs')
        tid = rt.submit(owner, session, event)
        task = rt.get(tid, owner)
        try:
            update(service.store, _client(service.cfg), task['message_id'], render(task), lambda: 8)
        except Exception:
            return 'Demo提交已接收，请查看原卡；不要重复提交。'
    else:
        task = rt.open(owner, session)
        key = hashlib.sha256(task['id'].encode()).hexdigest()[:32]
        mid = send(service.store, _client(service.cfg), session.split(':')[1], render(task), key, lambda: 8)
        rt.bind_message(task['id'], owner, mid, session)
    return 'NO_REPLY'


def tick(cfg):
    runtime(cfg).tick()


def recover(cfg):
    # Native workers outlive the Bot; tick observes their receipts and identities.
    runtime(cfg)


def deliver(cfg, store):
    from bench_gateway.card_entities import update
    from bench_gateway.task_cards import _client
    rt = runtime(cfg)
    deadline = cfg.get('_notification_deadline', time.monotonic()+8)
    with rt.connect() as c:
        rows = c.execute("select id,owner,sent_digest from tasks where message_id<>'' and state<>'DRAFT'").fetchall()
    for row in rows:
        if time.monotonic() >= deadline:
            return
        task = rt.get(row['id'], row['owner'])
        card = render(task)
        signature = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
        if signature == row['sent_digest']:
            continue
        try:
            update(store, _client(cfg), task['message_id'], card, lambda: max(.1, deadline-time.monotonic()))
        except Exception:
            return
        with rt.connect() as c:
            c.execute('update tasks set sent_digest=? where id=?', (signature, row['id']))
        return
