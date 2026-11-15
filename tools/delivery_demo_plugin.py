"""Demo routing alongside existing work orders on the single registered Bot."""
from contextlib import closing
import importlib
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path('/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r1')


def card_route(state_root, session, event):
    """Use the original card binding when the connector drops entry_schema."""
    root = Path(state_root) / 'roadmodel'
    value = event['value']
    bindings = (
        ('evaluation/runtime.sqlite3', 'tasks', 'evaluation_entry', 'roadmodel-evaluation/v1'),
        ('delivery-demo/runtime.sqlite3', 'tasks', 'delivery_demo_entry', 'delivery-demo/v1'),
        ('model-tickets/tickets.sqlite3', 'tickets', 'model_ticket_entry', 'model-ticket/v2'),
        ('runtime.sqlite3', 'tasks', 'feishu_entry', None),
    )
    matches = []
    for name, table, handler, schema in bindings:
        path = root / name
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            row = db.execute(
                f'SELECT 1 FROM {table} WHERE id=? AND message_id=? AND owner=? AND session=?',
                (value.get('task_id'), event['message_id'], event['operator_id'], session),
            ).fetchone()
        if row:
            matches.append((handler, schema))
    if len(matches) != 1:
        return None
    handler, schema = matches[0]
    claimed_schema = value.get('entry_schema')
    if schema and claimed_schema not in (None, schema):
        return None
    return handler, schema


def backend(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.import_module('pcm_eval.'+name)
    if ROOT not in Path(module.__file__).resolve().parents:
        raise ValueError('Unexpected RoadModel module release')
    return module


def handle(service, session, text):
    entry = backend('evaluation_entry')
    if text.strip() in entry.COMMANDS:
        return entry.handle(service, session, text)
    demo = backend('delivery_demo_entry')
    if text.strip() in demo.COMMANDS:
        return demo.handle(service, session, text)
    from .card_ui import PREFIX, decode
    if text.startswith(PREFIX):
        event = decode(service.cfg, session, text)
        if event.get('action') == 'roadmodel_submit':
            route = card_route(service.cfg['state_root'], session, event)
            if route is None:
                return '找不到这张卡绑定的工单。请发送 RoadModel评测 打开当前入口。'
            handler, schema = route
            if schema is None:
                return backend(handler).handle(service, session, text)
            routed = {**event, 'value': {**event['value'], 'entry_schema': schema}}
            return backend(handler).handle(service, session, text, routed)
    if text.strip() in backend('model_ticket_entry').COMMANDS:
        return backend('model_ticket_entry').handle(service, session, text)
    return None


def recover(cfg):
    for name in ('evaluation_entry', 'delivery_demo_entry', 'feishu_entry'):
        backend(name).recover(cfg)


def tick(cfg):
    for name in ('evaluation_entry', 'delivery_demo_entry', 'model_ticket_entry', 'feishu_entry'):
        backend(name).tick(cfg)


def deliver(cfg, store, deadline=None):
    if deadline:
        cfg = {**cfg, '_notification_deadline': min(deadline, cfg.get('_notification_deadline', deadline))}
    for name in ('evaluation_entry', 'delivery_demo_entry', 'model_ticket_entry', 'feishu_entry'):
        if time.monotonic() >= cfg.get('_notification_deadline', time.monotonic()+8):
            return
        backend(name).deliver(cfg, store)
