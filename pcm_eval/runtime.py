"""Durable dual-chain Runtime. Unknown execution is never automatically retried."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .common import load_json, save_json
from .two_link import resolve_pair, validate_inputs
from .planning import KINDS


class Runtime:
    def __init__(self, root, *, catalog_file=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_file = catalog_file
        with self.connect() as c:
            c.executescript('''
              CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,owner TEXT NOT NULL,session TEXT NOT NULL,
                message_id TEXT NOT NULL,nonce TEXT NOT NULL UNIQUE,state TEXT NOT NULL,inputs TEXT NOT NULL,
                event_id TEXT UNIQUE,created REAL NOT NULL,updated REAL NOT NULL,notice TEXT NOT NULL DEFAULT '',
                sent_digest TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS chains(task TEXT NOT NULL,kind TEXT NOT NULL,state TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '{}',PRIMARY KEY(task,kind));
            ''')

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.root / 'runtime.sqlite3', timeout=15)
        c.row_factory = sqlite3.Row
        try:
            with c: yield c
        finally:
            c.close()

    def open(self, owner, session):
        if not re.fullmatch(r'ou_[A-Za-z0-9_-]{3,100}', owner) or session.split(':')[-1] != owner:
            raise ValueError('需要本人飞书私聊身份')
        task = 'rm-' + uuid.uuid4().hex[:20]
        now = time.time()
        with self.connect() as c:
            c.execute('INSERT INTO tasks(id,owner,session,message_id,nonce,state,inputs,created,updated) VALUES(?,?,?,?,?,?,?,?,?)',
                      (task, owner, session, '', uuid.uuid4().hex, 'DRAFT', '{}', now, now))
        return self.get(task, owner)

    def bind_message(self, task, owner, message_id, session):
        with self.connect() as c:
            if c.execute("UPDATE tasks SET message_id=?,session=? WHERE id=? AND owner=? AND state='DRAFT' AND message_id=''",
                         (message_id, session, task, owner)).rowcount != 1:
                raise ValueError('卡片身份已绑定，不能重新设置')

    def get(self, task, owner):
        with self.connect() as c:
            row = c.execute('SELECT * FROM tasks WHERE id=? AND owner=?', (task, owner)).fetchone()
            if not row: raise ValueError('任务不存在或不属于本人')
            value = dict(row)
            value['inputs'] = json.loads(value['inputs'])
            value['chains'] = [dict(x) for x in c.execute('SELECT * FROM chains WHERE task=? ORDER BY kind', (task,))]
            for x in value['chains']: x['detail'] = json.loads(x['detail'])
            return value

    def validate_submission(self, form):
        return validate_inputs(form)

    def submit(self, owner, session, event):
        inputs = self.validate_submission(event.get('form_value'))
        value = event['value']
        if event.get('action') != 'roadmodel_submit' or not isinstance(event.get('event_id'), str) or not event['event_id']:
            raise ValueError('不是有效的RoadModel表单提交')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT * FROM tasks WHERE id=? AND owner=?', (value.get('task_id'), owner)).fetchone()
            if (not row or row['session'] != session or row['message_id'] != event.get('message_id')
                    or row['nonce'] != value.get('nonce') or event.get('operator_id') != owner
                    or event.get('chat_id') != session.split(':')[1]):
                raise ValueError('卡片、本人私聊或操作人核验失败')
            if row['state'] != 'DRAFT':
                if row['event_id'] == event['event_id'] and json.loads(row['inputs']) == inputs:
                    return row['id']
                raise ValueError('本卡已提交，请重新打开入口；不会重复执行')
            if c.execute('SELECT 1 FROM tasks WHERE event_id=?', (event['event_id'],)).fetchone():
                raise ValueError('事件已经用于另一张卡片')
            c.execute("UPDATE tasks SET state='QUEUED',inputs=?,event_id=?,updated=? WHERE id=?",
                      (json.dumps(inputs), event['event_id'], time.time(), row['id']))
            for kind in KINDS:
                c.execute("INSERT INTO chains(task,kind,state) VALUES(?,?,'PENDING')", (row['id'], kind))
            return row['id']

    def recover(self):
        with self.connect() as c:
            c.execute("UPDATE tasks SET state='NEEDS_REVIEW',notice='调度曾中断，需核对原执行收据；不会自动重跑' WHERE state='RUNNING' AND id NOT IN (SELECT task FROM chains WHERE json_extract(detail,'$.mode')='active_native_execution')")
            c.execute("UPDATE chains SET state='NEEDS_REVIEW' WHERE state='RUNNING' AND COALESCE(json_extract(detail,'$.mode'),'')<>'active_native_execution'")

    def refresh_active(self):
        from .active_execution import poll
        with self.connect() as c:
            rows=c.execute("SELECT task,kind,detail FROM chains WHERE json_extract(detail,'$.mode')='active_native_execution' AND state='RUNNING'").fetchall()
        for row in rows:
            old=json.loads(row['detail'])
            try:detail=poll(old['binding'])
            except (ValueError,OSError,KeyError,json.JSONDecodeError):
                # A transient read cannot finish a task or dispatch more inference.
                continue
            with self.connect() as c:
                c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',(detail['state'],json.dumps(detail,ensure_ascii=False),row['task'],row['kind']))
                states=[r[0] for r in c.execute('SELECT state FROM chains WHERE task=?',(row['task'],))]
                final='SUCCEEDED' if all(s=='SUCCEEDED' for s in states) else 'NEEDS_REVIEW' if 'NEEDS_REVIEW' in states else 'RUNNING'
                notice='已关联相同输入的在途正式评测；继续未完成范围，不重复已完成帧。两条评测及报告回读全部完成后才标记已完成。'
                if final=='SUCCEEDED':
                    details=[json.loads(r[0]) for r in c.execute('SELECT detail FROM chains WHERE task=?',(row['task'],))]
                    urls={d.get('report_url') for d in details if d.get('report_url')}
                    if len(urls)==1:notice='两条评测、报告生成及飞书回读已完成。\n[查看合并评测报告]('+next(iter(urls))+')'
                c.execute('UPDATE tasks SET state=?,notice=?,updated=? WHERE id=?',(final,notice,time.time(),row['task']))

    def tick(self, executor=None):
        self.refresh_active()
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute("SELECT * FROM tasks WHERE state='QUEUED' ORDER BY created LIMIT 1").fetchone()
            if not row: return None
            c.execute("UPDATE tasks SET state='RUNNING',updated=? WHERE id=?", (time.time(), row['id']))
        task = dict(row)
        run = self.root / 'runs' / task['id']
        try:
            run.mkdir(parents=True, exist_ok=False)
            resolved = resolve_pair(json.loads(task['inputs']), task['id'], catalog_file=self.catalog_file)
            save_json(run / 'resolution.json', resolved)
            def chain(job):
                kind = job['kind']
                if not job['ready']:
                    state, detail = 'BLOCKED', {'issues': job['issues']}
                else:
                    with self.connect() as c:
                        c.execute("UPDATE chains SET state='RUNNING' WHERE task=? AND kind=?", (task['id'], kind))
                    try:
                        detail = (executor or execute_atomic)(job, run / kind)
                        state = detail['state']
                        if state not in ('SUCCEEDED', 'PARTIAL', 'FAILED', 'BLOCKED', 'NEEDS_REVIEW','RUNNING'):
                            raise ValueError('原子执行器返回无效状态')
                    except Exception as error:
                        state, detail = 'NEEDS_REVIEW', {'issues': ['执行结果待核验: ' + type(error).__name__]}
                with self.connect() as c:
                    c.execute('UPDATE chains SET state=?,detail=? WHERE task=? AND kind=?',
                              (state, json.dumps(detail, ensure_ascii=False), task['id'], kind))
                return state
            with ThreadPoolExecutor(max_workers=2) as pool:
                states = list(pool.map(chain, resolved['jobs']))
            final = ('SUCCEEDED' if all(x == 'SUCCEEDED' for x in states) else 'NEEDS_REVIEW' if 'NEEDS_REVIEW' in states else 'RUNNING' if 'RUNNING' in states
                     else 'PARTIAL' if any(x in ('SUCCEEDED', 'PARTIAL') for x in states) else 'BLOCKED')
            notice = ''
            if any(j.get('active_execution') for j in resolved['jobs']):
                notice='已关联相同输入的在途正式评测；继续未完成范围，不重复已完成帧。两条评测及报告回读全部完成后才标记已完成。'
            if all(x == 'SUCCEEDED' for x in states) and all(j.get('retained_result') for j in resolved['jobs']):
                from .retained import compose
                jobs = resolved['jobs']
                if len({(j['material_root'], json.dumps(j['delivery_record'],sort_keys=True)) for j in jobs}) != 1:
                    raise ValueError('Two retained chains have different delivery bindings')
                current = self.get(task['id'],task['owner'])
                delivery = compose(run,{x['kind']:x['detail'] for x in current['chains']},
                                   root=jobs[0]['material_root'],delivery_record=jobs[0]['delivery_record'])
                notice = '复用已验证小场景结果，新增推理0帧；全量评测独立继续。\n[查看合并评测报告]('+delivery['report_url']+')'
        except Exception as error:
            final, notice = 'NEEDS_REVIEW', '入口预检中断: ' + type(error).__name__
        with self.connect() as c:
            c.execute('UPDATE tasks SET state=?,notice=?,updated=? WHERE id=?', (final, notice, time.time(), task['id']))
        return self.get(task['id'], task['owner'])


def execute_atomic(job, target):
    """Invoke existing execution and report abilities, never a second agent loop."""
    if job.get('active_execution'):
        from .active_execution import execute as execute_active
        return execute_active(job,target)
    if job.get('retained_result'):
        from .retained import execute as execute_retained
        return execute_retained(job, target)
    from .runner import execute
    from .model_report import write_model_report
    from .compile_report import write_compile_report
    target.mkdir(parents=True, exist_ok=False)
    save_json(target / 'request.json', job['request'])
    receipt = execute(job['plan'], target / 'execution')
    if receipt['state'] != 'succeeded' or not receipt.get('result_file'):
        if receipt.get('steps') and receipt['steps'][-1]['name']=='preflight' and receipt['steps'][-1]['exit_code']:
            return {'state':'BLOCKED','execution':receipt,'issues':['执行前环境或输入检查失败，尚未启动推理；请核对预检日志。']}
        return {'state': 'NEEDS_REVIEW', 'execution': receipt, 'issues': ['核查原子执行收据后再续收，不自动重跑']}
    result = load_json(receipt['result_file'])
    if result['collection']['status'] != 'complete':
        return {'state': 'PARTIAL', 'execution': receipt, 'issues': ['已有推理结果，但指标收集尚不完整']}
    if job['kind'] == 'model_accuracy':
        from .atomic_result import prepare_model_result
        bound = prepare_model_result(job,receipt,target/'normalized-result.json')
        report = write_model_report(target / 'report', result_file=bound,
                                    binding_file=job.get('binding_file'),gt_preparation_file=job.get('gt_preparation_file'),
                                    supporting_artifacts=[target/'request.json',target/'execution/plan.json',target/'execution/execution_receipt.json'])
    else:
        report = write_compile_report(target / 'report/report.xml', result_file=receipt['result_file'],
                                      request_file=target / 'request.json')
    # Publication is an independently recorded stage, not inferred from a local file.
    from .publication import publish_report
    try:
        publication = publish_report(report.get('report_xml') or report['xml'], target / 'publication')
    except Exception as error:
        return {'state': 'PARTIAL', 'execution': receipt, 'report': report, 'publication': 'NEEDS_REVIEW',
                'issues': ['评测报告已生成，飞书发布需核对原收据: ' + type(error).__name__]}
    return {'state': 'PARTIAL' if job.get('warnings') else 'SUCCEEDED', 'execution': receipt, 'report': report, 'publication': publication,
            'report_url': publication['report_url'], 'issues': job.get('warnings', [])}
