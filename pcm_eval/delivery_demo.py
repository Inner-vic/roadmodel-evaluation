"""Demo queue on the existing Bot; authenticated source reads stay on the PC."""
import json
import os
from pathlib import Path
import subprocess
import time

from .common import digest, json_digest, load_json, save_json
from .runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


class DemoRuntime(Runtime):
    scope = 'validation'
    worker = 'delivery_demo_worker.py'
    ready_notice = 'Demo范围：模型OCC 2帧，编译张量对齐1帧。'

    def __init__(self, root, *, config_file=None):
        super().__init__(root)
        self.config_file = Path(config_file or ROOT/'configs/delivery_demo.json')
        with self.connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS materials(
                task TEXT PRIMARY KEY,request TEXT NOT NULL,digest TEXT NOT NULL,
                state TEXT NOT NULL,detail TEXT NOT NULL DEFAULT '{}',updated REAL NOT NULL)''')
            c.execute('CREATE TABLE IF NOT EXISTS source_agent(id TEXT PRIMARY KEY,heartbeat REAL)')

    def get(self, task, owner):
        value = super().get(task, owner)
        progress = self.root/'runs'/task/'progress.json'
        if progress.exists():
            current = load_json(progress)
            publication = current.get('combined_report', {})
            if publication.get('state') == 'VERIFIED':
                value['report_url'] = publication['report_url']
        return value

    def claim_materials(self, agent, owner):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id'] or owner != cfg['owner']:
            raise ValueError('Source agent identity differs')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            c.execute('insert or replace into source_agent values(?,?)', (agent, time.time()))
            row = c.execute("select * from materials where state='READING' order by updated limit 1").fetchone()
            if row:
                return dict(row)
            row = c.execute("select * from materials where state='QUEUED' order by updated limit 1").fetchone()
            if not row:
                return None
            c.execute("update materials set state='READING',updated=? where task=?", (time.time(), row['task']))
            return dict(row)

    def complete_materials(self, agent, task, request_digest, detail):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id']:
            raise ValueError('Source agent identity differs')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('select * from materials where task=?', (task,)).fetchone()
            if not row or row['digest'] != request_digest or row['state'] != 'READING':
                raise ValueError('Source request changed or is no longer owned')
            if detail.get('workflow'):
                w = detail['workflow']
                if w['inputs'] != json.loads(row['request']) or w['scope'] != self.scope:
                    raise ValueError('Prepared workflow differs from confirmed Demo inputs')
            c.execute("update materials set state='PREPARED',detail=?,updated=? where task=?",
                      (json.dumps(detail, ensure_ascii=False), time.time(), task))

    def material_request(self, inputs):
        return {'model_delivery_url': inputs['model_delivery_url'], 'dataset_ref': inputs['dataset_url']}

    def tick(self, spawn=None):
        cfg = load_json(self.config_file)
        with self.connect() as c:
            rows = [dict(r) for r in c.execute("select * from tasks where state='QUEUED'")]
        for row in rows:
            inputs = json.loads(row['inputs'])
            # Preserve the two-input form, with a source-agent-friendly dataset name.
            request = self.material_request(inputs)
            if row['owner'] != cfg['owner']:
                detail = {'issues': ['Demo尚未配置该用户的开发空间与报告交付']}
                with self.connect() as c:
                    c.execute("update tasks set state='BLOCKED',notice=? where id=?", (detail['issues'][0], row['id']))
                continue
            with self.connect() as c:
                c.execute('insert or ignore into materials values(?,?,?,?,?,?)',
                          (row['id'], json.dumps(request), json_digest(request), 'QUEUED', '{}', time.time()))
                c.execute("update tasks set state='PREPARING',notice=? where id=?", ('正在读取线上交付材料并核验两条链路。', row['id']))
        with self.connect() as c:
            rows = [dict(r) for r in c.execute("select m.* from materials m join tasks t on t.id=m.task where m.state='PREPARED' and t.state='PREPARING'")]
        for row in rows:
            detail = json.loads(row['detail'])
            workflow = detail.get('workflow')
            if not workflow:
                failed_environment = detail.get('failure_kind') == 'preparation_environment'
                state = 'NEEDS_REVIEW' if failed_environment else 'BLOCKED'
                notice = detail.get('error', '材料读取待核验')
                if failed_environment:
                    notice += '；此为后台准备异常，不需要重新填写交付材料。'
                with self.connect() as c:
                    c.execute("update tasks set state=?,notice=? where id=?", (state, notice, row['task']))
                    c.execute("update chains set state=?,detail=? where task=?", (state,json.dumps({'stage':detail.get('stage'), 'issues':[notice]}),row['task']))
                continue
            run = self.root/'runs'/row['task']
            run.mkdir(parents=True, exist_ok=True)
            path = run/'workflow.json'
            if not path.exists():
                save_json(path, workflow)
            elif load_json(path) != workflow:
                raise ValueError('Prepared Demo workflow changed')
            with self.connect() as c:
                for job in workflow['jobs']:
                    c.execute('update chains set state=?,detail=? where task=? and kind=?',
                              ('PENDING' if job['ready'] else 'BLOCKED', json.dumps({'issues': job['issues']}), row['task'], job['kind']))
                c.execute("update tasks set state='READY',notice=? where id=?", (self.ready_notice, row['task']))
        with self.connect() as c:
            running = [dict(r) for r in c.execute("select * from tasks where state='RUNNING'")]
        for row in running:
            run = self.root/'runs'/row['id']
            final = run/'progress.json'
            if final.exists():
                value = load_json(final)
                state = value['state']
                with self.connect() as c:
                    for chain in value['chains']:
                        c.execute('update chains set state=?,detail=? where task=? and kind=?',
                                  (chain['state'], json.dumps(chain, ensure_ascii=False), row['id'], chain['kind']))
                    c.execute('update tasks set state=?,updated=? where id=?', (state, time.time(), row['id']))
                continue
            for kind in ('model_accuracy', 'compile_alignment'):
                path = run/'effective'/kind/'chain.json'
                if path.exists():
                    chain = load_json(path)
                    with self.connect() as c:
                        c.execute('update chains set state=?,detail=? where task=? and kind=?',
                                  (chain['state'], json.dumps(chain, ensure_ascii=False), row['id'], kind))
            dispatch = load_json(run/'dispatch.json')
            pid = dispatch.get('pid')
            alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    alive = str(run).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
                except OSError:
                    pass
            if not alive:
                with self.connect() as c:
                    c.execute("update tasks set state='NEEDS_REVIEW',notice=? where id=?", ('Demo执行进程已退出，需要核验原收据；不会重复推理。', row['id']))
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute("select 1 from tasks where state='RUNNING'").fetchone():
                return
            row = c.execute("select * from tasks where state='READY' order by created limit 1").fetchone()
            if not row:
                return
            c.execute("update tasks set state='RUNNING' where id=?", (row['id'],))
        run = self.root/'runs'/row['id']
        dispatch = {'state': 'DISPATCHING', 'workflow_sha256': digest(run/'workflow.json')}
        save_json(run/'dispatch.json', dispatch)
        try:
            with (run/'worker.log').open('xb') as f:
                child = (spawn or subprocess.Popen)([cfg['python'], str(ROOT/'tools'/self.worker), str(run)],
                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            dispatch.update(state='STARTED', pid=child.pid)
            save_json(run/'dispatch.json', dispatch, overwrite=True)
        except Exception:
            with self.connect() as c:
                c.execute("update tasks set state='NEEDS_REVIEW' where id=?", (row['id'],))
            raise
