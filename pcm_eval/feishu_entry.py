"""Plugin for the existing authenticated CC Connect bridge, no new connection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
OPEN_COMMANDS = {'RoadModel', 'RoadModel评测', 'RoadModel 评测', 'roadmodel'}
LABELS = {'model_accuracy': '模型精度', 'compile_alignment': '编译精度'}
STATES = {'DRAFT': '填写材料', 'QUEUED': '已提交', 'RUNNING': '评测中', 'SUCCEEDED': '已完成',
          'BLOCKED': '待补充材料', 'PARTIAL': '部分完成', 'NEEDS_REVIEW': '待核验', 'PENDING': '待检查',
          'FAILED': '失败'}


def runtime(cfg):
    # A separate RoadModel database; no shared hardware job or material state mutation.
    return Runtime(Path(cfg['state_root']) / 'roadmodel', catalog_file=ROOT / 'configs/two_link_catalog.json')


def render(task):
    elements = [{'tag': 'markdown', 'content': '**' + STATES[task['state']] + '**\n模型精度与编译精度分别执行。相同输入已有在途任务时关联并继续执行；各自显示原生进度，报告生成与回读完成后返回报告。'}]
    if task['state'] == 'DRAFT':
        value = {'action': 'bench:roadmodel_submit', 'task_id': task['id'], 'nonce': task['nonce']}
        inputs = []
        for name, label, prompt in [('model_delivery_url', '模型交付链接', '粘贴飞书模型交付记录链接'),
                                    ('dataset_url', '数据集链接', '粘贴本次数据集对应的LPAI任务或数据资产链接')]:
            inputs.append({'tag': 'input', 'name': name, 'required': True, 'input_type': 'text',
                           'width': 'fill', 'max_length': 1000, 'label_position': 'top',
                           'label': {'tag': 'plain_text', 'content': label},
                           'placeholder': {'tag': 'plain_text', 'content': prompt}})
        inputs.append({'tag': 'button', 'name': 'roadmodel_submit', 'type': 'primary_filled', 'width': 'fill',
                       'text': {'tag': 'plain_text', 'content': '确认执行'}, 'form_action_type': 'submit',
                       'behaviors': [{'type': 'callback', 'value': value}]})
        elements.append({'tag': 'form', 'name': 'roadmodel_form', 'element_id': 'roadmodel_form', 'elements': inputs})
    else:
        for kind in ('model_accuracy', 'compile_alignment'):
            chain = next((x for x in task['chains'] if x['kind'] == kind), {'state': 'PENDING', 'detail': {}})
            detail = chain['detail']
            status='报告整理中' if chain['state']=='RUNNING' and detail.get('native_finished') else STATES.get(chain['state'],'待核验')
            text = '**' + LABELS[kind] + '** · ' + status
            for issue in detail.get('issues', [])[:4]: text += '\n' + issue
            if detail.get('report_url') and detail.get('mode')!='active_native_execution': text += '\n[查看单次报告](' + detail['report_url'] + ')'
            elements.append({'tag': 'column_set', 'flex_mode': 'none', 'background_style': 'grey',
                             'columns': [{'tag': 'column', 'width': 'weighted', 'weight': 1,
                                          'elements': [{'tag': 'markdown', 'content': text}]}]})
        if task.get('notice'): elements.append({'tag': 'markdown', 'content': task['notice']})
    return {'schema': '2.0', 'config': {'update_multi': True, 'enable_forward': False, 'width_mode': 'default'},
            'header': {'template': 'blue', 'title': {'tag': 'plain_text', 'content': 'RoadModel 评测'}},
            'body': {'elements': elements}}


def handle(service, session, text):
    """Return None for every non-RoadModel message; called before bench intake."""
    from bench_gateway.card_ui import PREFIX, decode
    from bench_gateway.service import identity
    from bench_gateway.card_entities import send, update
    from bench_gateway.task_cards import _client
    event = None
    if text.startswith(PREFIX):
        event = decode(service.cfg, session, text)
        if event.get('action') != 'roadmodel_submit': return None
    elif text.strip() not in OPEN_COMMANDS:
        return None
    owner = identity(session)
    rt = runtime(service.cfg)
    if event:
        tid = rt.submit(owner, session, event)
        task = rt.get(tid, owner)
        try: update(service.store, _client(service.cfg), task['message_id'], render(task), lambda: 8)
        except Exception: return 'RoadModel已接收本次提交，将在原卡显示结果；不会重复执行。'
    else:
        task = rt.open(owner, session)
        key = hashlib.sha256(task['id'].encode()).hexdigest()[:32]
        mid = send(service.store, _client(service.cfg), session.split(':')[1], render(task), key, lambda: 8)
        rt.bind_message(task['id'], owner, mid, session)
    return 'NO_REPLY'


def recover(cfg):
    runtime(cfg).recover()


def tick(cfg):
    return runtime(cfg).tick()


def deliver(cfg, store):
    from bench_gateway.card_entities import update
    from bench_gateway.task_cards import _client
    deadline = cfg.get('_notification_deadline', time.monotonic() + 8)
    if time.monotonic() >= deadline: return
    rt = runtime(cfg)
    with rt.connect() as c:
        rows = c.execute("SELECT id,owner,sent_digest FROM tasks WHERE message_id<>'' AND state<>'DRAFT'").fetchall()
    for row in rows:
        if time.monotonic() >= deadline: return
        task = rt.get(row['id'], row['owner'])
        card = render(task)
        signature = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
        if signature == row['sent_digest']: continue
        try: update(store, _client(cfg), task['message_id'], card, lambda: max(.1, deadline - time.monotonic()))
        except Exception: return  # Sending cannot trigger another evaluation.
        with rt.connect() as c:
            c.execute('UPDATE tasks SET sent_digest=? WHERE id=?', (signature, task['id']))
        return  # Bound one card per notifier pass; preserve existing notification budgets.
