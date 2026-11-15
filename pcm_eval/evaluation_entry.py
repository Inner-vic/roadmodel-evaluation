"""Three-field RoadModel work order with explicit per-head dataset sources."""
import hashlib
import json
from pathlib import Path
import time

from .evaluation import EvaluationRuntime

COMMANDS = {'RoadModel', 'RoadModel评测', 'RoadModel 评测', 'roadmodel'}
SCHEMA = 'roadmodel-evaluation/v1'
STATES = {'DRAFT': '填写材料', 'QUEUED': '已提交', 'PREPARING': '核验材料',
          'READY': '等待执行', 'RUNNING': '评测中', 'WAITING_REPORT': '报告发布中',
          'WAITING_BENCH': '等待台架', 'WAITING_SPACE': '等待开发空间',
          'BLOCKED': '待补充材料', 'NEEDS_REVIEW': '待核验', 'PARTIAL': '部分完成',
          'SUCCEEDED': '已完成', 'PENDING': '待检查', 'FAILED': '失败'}


def runtime(cfg):
    return EvaluationRuntime(Path(cfg['state_root'])/'roadmodel/evaluation')


def render(task):
    def plain(text):
        return {'tag': 'plain_text', 'content': text}

    def box(text):
        return {'tag': 'column_set', 'flex_mode': 'none', 'background_style': 'grey',
                'columns': [{'tag': 'column', 'width': 'weighted', 'weight': 1,
                             'elements': [{'tag': 'markdown', 'content': text}]}]}

    elements = [box('**'+STATES.get(task['state'], '待核验')+'**\n模型精度 · 编译精度 · 飞书报告')]
    if task['state'] == 'DRAFT':
        fields = []
        for name, label, prompt in (
                ('model_delivery_url', '模型交付链接', '飞书交付记录或模型资料文件夹链接'),
                ('dataset_url', '评测数据集链接', '模型精度与编译精度共用的LPAI数据集版本链接')):
            fields.append({'tag': 'input', 'name': name, 'required': True, 'input_type': 'text',
                           'label_position': 'top', 'width': 'fill', 'max_length': 1000,
                           'label': plain(label), 'placeholder': plain(prompt)})
        fields.append({'tag': 'button', 'name': 'roadmodel_submit', 'type': 'primary_filled', 'width': 'fill',
                       'text': plain('开始评测'), 'form_action_type': 'submit',
                       'behaviors': [{'type': 'callback', 'value': {'action': 'bench:roadmodel_submit',
                           'entry_schema': SCHEMA, 'task_id': task['id'], 'nonce': task['nonce']}}]})
        elements.append({'tag': 'form', 'name': 'roadmodel_form', 'element_id': 'roadmodel_form', 'elements': fields})
        options = task.get('options', {'scope': 'full'})
        if options['scope'] == 'validation':
            selection = options['selection']
            scope = f"当前试运行范围为每个可评测头{selection['scene_count']}场景、每场景{selection['frames_per_scene']}帧。全部评测项均检查，缺GT项会明确标记。"
        else:
            scope = '两条评测链路使用同一份数据集，冻结完整验证范围并检查全部评测项；缺少可靠GT的项目如实标注，不自动换用其他数据。'
        elements.append({'tag': 'markdown', 'content': scope+'提交后在本卡查看两条链路进度与报告。'})
    else:
        for kind, title in (('model_accuracy', '模型精度'), ('compile_alignment', '编译精度')):
            chain = next((c for c in task['chains'] if c['kind'] == kind), {'state': 'PENDING', 'detail': {}})
            detail = chain['detail']
            text = '**'+title+'** · '+STATES.get(chain['state'], '待核验')
            if detail.get('stage'):
                text += '\n'+detail['stage']
            for issue in detail.get('issues', [])[:4]:
                text += '\n'+issue
            elements.append(box(text))
        if task.get('notice'):
            elements.append({'tag': 'markdown', 'content': task['notice']})
        if task.get('report_url'):
            elements.append({'tag': 'button', 'type': 'primary_filled', 'width': 'fill',
                             'text': plain('查看评测报告'), 'behaviors': [{'type': 'open_url', 'default_url': task['report_url']}]})
    return {'schema': '2.0', 'config': {'update_multi': True, 'enable_forward': False, 'width_mode': 'default'},
            'header': {'template': 'blue', 'title': plain('RoadModel 评测工单')}, 'body': {'elements': elements}}


def handle(service, session, text, event=None):
    from bench_gateway.service import identity
    from bench_gateway.card_entities import send, update
    from bench_gateway.task_cards import _client
    owner = identity(session)
    rt = runtime(service.cfg)
    if event:
        if event.get('value', {}).get('entry_schema') != SCHEMA:
            raise ValueError('评测工单类型不符')
        tid = rt.submit(owner, session, event)
        task = rt.get(tid, owner)
        try:
            update(service.store, _client(service.cfg), task['message_id'], render(task), lambda: 8)
        except Exception:
            return 'RoadModel工单已接收，将在原卡更新进度。'
    else:
        task = rt.open(owner, session)
        mid = send(service.store, _client(service.cfg), session.split(':')[1], render(task),
                   hashlib.sha256(task['id'].encode()).hexdigest()[:32], lambda: 8)
        rt.bind_message(task['id'], owner, mid, session)
    return 'NO_REPLY'


def recover(cfg):
    runtime(cfg)


def tick(cfg):
    runtime(cfg).tick()


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
        card = render(rt.get(row['id'], row['owner']))
        signature = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
        if signature == row['sent_digest']:
            continue
        try:
            update(store, _client(cfg), rt.get(row['id'], row['owner'])['message_id'], card,
                   lambda: max(.1, deadline-time.monotonic()))
        except Exception:
            return
        with rt.connect() as c:
            c.execute('update tasks set sent_digest=? where id=?', (signature, row['id']))
        return
