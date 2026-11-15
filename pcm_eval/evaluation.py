"""Full-scope work orders and user-owned publication on the existing Bot."""
import json
from pathlib import Path
import time

from .common import json_digest, load_json
from .delivery_demo import DemoRuntime
from .delivery_workflow import _write_atomic
from .model_ticket import inputs as ticket_inputs

ROOT = Path(__file__).resolve().parents[1]


class EvaluationRuntime(DemoRuntime):
    scope = 'full'
    worker = 'evaluation_worker.py'
    ready_notice = '已冻结本次材料与完整验证范围，分别推进模型精度和编译精度。'

    def __init__(self, root, *, config_file=None):
        super().__init__(root, config_file=config_file or ROOT/'configs/evaluation.json')
        with self.connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS publications(
                task TEXT PRIMARY KEY,digest TEXT NOT NULL,payload TEXT NOT NULL,
                state TEXT NOT NULL,detail TEXT NOT NULL DEFAULT '{}',updated REAL NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS space_actions(
                id TEXT PRIMARY KEY,task TEXT NOT NULL,action TEXT NOT NULL,context TEXT NOT NULL,
                state TEXT NOT NULL,detail TEXT NOT NULL DEFAULT '{}',updated REAL NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS space_observations(
                name TEXT PRIMARY KEY,detail TEXT NOT NULL,updated REAL NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS task_options(
                task TEXT PRIMARY KEY,detail TEXT NOT NULL)''')

    def open(self, owner, session):
        task = super().open(owner, session)
        cfg = load_json(self.config_file)
        options = {'scope': cfg.get('execution_scope', 'full'), 'selection': cfg.get('validation_selection')}
        if options['scope'] not in ('full', 'validation') or (options['scope'] == 'full' and options['selection']):
            raise ValueError('Invalid installed evaluation scope')
        with self.connect() as c:
            c.execute('insert into task_options values(?,?)', (task['id'], json.dumps(options)))
        return self.get(task['id'], owner)

    def get(self, task, owner):
        value = super().get(task, owner)
        with self.connect() as c:
            row = c.execute('select detail from task_options where task=?', (task,)).fetchone()
        value['options'] = json.loads(row['detail']) if row else {'scope': 'full', 'selection': None}
        return value

    def claim_materials(self, agent, owner):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id'] or owner != cfg['owner']:
            raise ValueError('材料读取器身份不同')
        with self.connect() as c:
            c.execute("update materials set state='QUEUED' where state='WAITING_SPACE' and updated<?", (time.time()-10,))
        row = super().claim_materials(agent, owner)
        if row:
            row['options'] = self.get(row['task'], owner)['options']
        return row

    def request_space_action(self, task, action, context):
        if action not in ('observe', 'create', 'stop', 'start', 'configure_and_start'):
            raise ValueError('Unsupported space action')
        cfg = load_json(self.config_file)
        current = self.get(task, cfg['owner'])
        if context['dataset_ref'] not in self.dataset_refs(current['inputs']):
            raise ValueError('空间动作对应其他数据输入')
        if context['profile'] != cfg['profile']:
            raise ValueError('空间动作环境模板与后台登记不同')
        identity = {'task': task, 'action': action, 'context': context}
        if action == 'observe':
            identity['observation_minute'] = int(time.time()//60)
        key = json_digest(identity)
        with self.connect() as c:
            c.execute('insert or ignore into space_actions values(?,?,?,?,?,?,?)',
                      (key, task, action, json.dumps(context), 'QUEUED', '{}', time.time()))
        return {'state': 'QUEUED', 'action_id': key}

    def claim_space_action(self, agent, owner):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id'] or owner != cfg['owner']:
            raise ValueError('空间执行器身份不符')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute("select * from space_actions where state='QUEUED' order by updated limit 1").fetchone()
            if not row:
                return None
            c.execute("update space_actions set state='EXECUTING',updated=? where id=?", (time.time(), row['id']))
            return dict(row)

    def complete_space_action(self, agent, action_id, detail):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id']:
            raise ValueError('空间执行器身份不符')
        if detail.get('observation'):
            self._validate_space_observation(cfg, detail['observation'])
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('select * from space_actions where id=?', (action_id,)).fetchone()
            if not row or row['state'] != 'EXECUTING':
                raise ValueError('空间动作已经处理或没有领取')
            api = (cfg.get('space_backend') == 'official_cli'
                   and detail.get('provider') == 'LPAI_OFFICIAL_API'
                   and detail.get('real_api_call_completed') is True)
            api_waiting = api and detail.get('state') == 'WAITING_APPROVAL'
            pending = api_waiting
            observed_phase = (detail.get('observation', {}).get('space') or {}).get('status', {}).get('phase')
            confirmed_phases = {'start':('pending','starting','ready'), 'stop':('stopping','stopped')}
            if pending and observed_phase in confirmed_phases.get(row['action'], ()):
                pending = False
            verified = ((detail.get('state') in ('OBSERVED', 'ACTION_RETURNED')
                         and (detail.get('real_browser_call_completed') is True or api)) or api_waiting)
            c.execute('update space_actions set state=?,detail=?,updated=? where id=?',
                      ('WAITING_APPROVAL' if pending else 'OBSERVED' if verified else 'NEEDS_REVIEW', json.dumps(detail), time.time(), action_id))
        if verified and detail.get('observation'):
            self.record_space_observation(detail['observation'])
            if pending:
                with self.connect() as c:
                    c.execute('update tasks set notice=?,sent_digest=? where id=?',
                              ('启动请求已提交，正在等待LPAI实际状态回读；不重复发送', '', row['task']))
        elif verified:
            with self.connect() as c:
                c.execute('delete from space_observations where name=?', (cfg['profile']['space_name'],))
            current = self.get(row['task'], cfg['owner'])
            self.request_space_action(row['task'], 'observe', {'profile': cfg['profile'],
                'dataset_ref': json.loads(row['context'])['dataset_ref'], 'after_action': action_id})
        else:
            reason = detail.get('reason') or detail.get('reply', {}).get('problem') or '空间动作结果需核验，不重复提交'
            with self.connect() as c:
                c.execute('update chains set state=?,detail=? where task=? and kind=?',
                          ('NEEDS_REVIEW', json.dumps({'stage': '开发空间准备待核验', 'issues': [reason],
                           'space_action_id': action_id}), row['task'], 'model_accuracy'))
                c.execute('update tasks set notice=?,sent_digest=?,updated=? where id=?',
                          (reason, '', time.time(), row['task']))

    def space_observation(self):
        cfg = load_json(self.config_file)
        with self.connect() as c:
            row = c.execute('select detail,updated from space_observations where name=?', (cfg['profile']['space_name'],)).fetchone()
        return json.loads(row['detail']) if row and time.time()-row['updated'] <= 300 else None

    @staticmethod
    def _validate_space_observation(cfg, observation):
        official = observation.get('source') == 'LPAI_OFFICIAL_API'
        if official and (cfg.get('space_backend') != 'official_cli'
                         or not observation.get('space')
                         or observation['space'].get('metadata', {}).get('uid') != cfg['profile']['space_uid']
                         or observation['space'].get('verification', {}).get('source') != 'LPAI_OFFICIAL_API'):
            raise ValueError('官方空间回读缺少已启用的后端或固定UID')
        if (observation.get('source') not in ('LPAI_UI', 'LPAI_OFFICIAL_API') or observation.get('space_name') != cfg['profile']['space_name']
                or observation.get('namespace') != cfg['profile']['namespace']
                or time.time()-observation.get('observed_at', 0) > 300):
            raise ValueError('页面观察记录缺少对应空间或已过期')
        access = (observation.get('space') or {}).get('runtime_access')
        if access and (access.get('host') != cfg['profile']['ssh_host']
                       or access.get('credential_name') != cfg['profile']['environment_template']['accesses']['ssh_credential']
                       or type(access.get('port')) is not int or not 1 <= access['port'] <= 65535):
            raise ValueError('页面SSH入口与已审核集群或现有凭证不同')

    def record_space_observation(self, observation):
        cfg = load_json(self.config_file)
        self._validate_space_observation(cfg, observation)
        with self.connect() as c:
            c.execute('insert or replace into space_observations values(?,?,?)',
                      (observation['space_name'], json.dumps(observation), time.time()))
            if observation.get('source') == 'LPAI_OFFICIAL_API':
                phase = observation['space']['status']['phase']
                for row in c.execute("select id,action,context,detail from space_actions where state='WAITING_APPROVAL'").fetchall():
                    context = json.loads(row['context'])
                    expected = {'start':('pending','starting','ready'),'stop':('stopping','stopped')}.get(row['action'], ())
                    if phase in expected and context.get('profile', {}).get('space_uid') == cfg['profile']['space_uid']:
                        detail = json.loads(row['detail'])
                        detail['platform_request_reconciled'] = {'source':'LPAI_OFFICIAL_API','phase':phase,
                                                                'space_uid':cfg['profile']['space_uid']}
                        c.execute("update space_actions set state='OBSERVED',detail=?,updated=? where id=?",
                                  (json.dumps(detail),time.time(),row['id']))
            c.execute("update materials set state='QUEUED' where state='WAITING_SPACE'")

    @staticmethod
    def dataset_refs(inputs):
        return {inputs[k] for k in ('dataset_url', 'occ_dataset_url', 'vision_dataset_url') if inputs.get(k)}

    def space_journal_path(self, task, dataset_ref=None):
        cfg = load_json(self.config_file)
        current = self.get(task, cfg['owner'])
        if dataset_ref is not None and dataset_ref not in self.dataset_refs(current['inputs']):
            raise ValueError('空间准备日志对应其他数据输入')
        name = 'space-preparation.json' if dataset_ref is None else 'space-preparation-'+json_digest(dataset_ref)[:20]+'.json'
        return self.root/'runs'/task/name

    def save_space_journal(self, task, journal, dataset_ref=None):
        cfg = load_json(self.config_file)
        self.get(task, cfg['owner'])
        if dataset_ref is not None and journal.get('source_ref') != dataset_ref:
            raise ValueError('空间日志来源与对应输入不同')
        _write_atomic(self.space_journal_path(task, dataset_ref), journal)

    def validate_submission(self, form):
        from .evaluation_inputs import validate_inputs
        return validate_inputs(form)

    def material_request(self, inputs):
        if 'occ_dataset_url' not in inputs:
            return super().material_request(inputs)  # Previously confirmed work orders retain their identity.
        return {'model_delivery_url': inputs['model_delivery_url'], 'dataset_ref': inputs['occ_dataset_url'],
                'vision_dataset_ref': inputs['vision_dataset_url']}

    def retry_pre_execution_materials(self, task, owner):
        """Explicit maintenance retry after fixing preparation, never inference."""
        value = self.get(task, owner)
        run = self.root/'runs'/task
        if value['state'] not in ('BLOCKED','NEEDS_REVIEW') or (run/'workflow.json').exists():
            raise ValueError('Only an unexecuted material-preparation failure can be retried')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('select * from materials where task=?',(task,)).fetchone()
            if not row or row['state']!='PREPARED' or json.loads(row['detail']).get('workflow'):
                raise ValueError('Preparation is already claimed or contains an execution plan')
            if db.execute('select 1 from publications where task=?',(task,)).fetchone():
                raise ValueError('An existing report must be reconciled separately')
            _write_atomic(run/'material-preparation-attempts'/(json_digest(dict(row))+'.json'),dict(row))
            db.execute("update materials set state='QUEUED',detail='{}',updated=? where task=?",(time.time(),task))
            db.execute("update tasks set state='PREPARING',notice=?,sent_digest='',updated=? where id=?",
                       ('评测环境已修复，正在接续原工单的材料准备；无需重新提交',time.time(),task))
            db.execute("update chains set state='PENDING',detail='{}' where task=?",(task,))
        return self.get(task, owner)

    def complete_materials(self, agent, task, request_digest, detail):
        if detail.get('state') in ('WAITING_UI', 'PREPARING', 'WAITING_SPACE'):
            cfg = load_json(self.config_file)
            if agent != cfg['agent_id']:
                raise ValueError('Source agent identity differs')
            with self.connect() as c:
                c.execute('BEGIN IMMEDIATE')
                row = c.execute('select * from materials where task=?', (task,)).fetchone()
                if not row or row['digest'] != request_digest or row['state'] != 'READING':
                    raise ValueError('空间准备请求发生变化')
                if (cfg.get('space_backend') == 'official_cli'
                        and c.execute("select 1 from space_actions where task=? and state='WAITING_APPROVAL'", (task,)).fetchone()):
                    detail = {**detail, 'stage':'等待LPAI处理原空间请求；状态核验后自动接续，不重复启动',
                              'issues':['空间就绪前不派发模型评测；后续以实际状态和挂载核验为准']}
                c.execute("update materials set state='WAITING_SPACE',detail=?,updated=? where task=?", (json.dumps(detail), time.time(), task))
                c.execute('update chains set state=?,detail=? where task=? and kind=?',
                          ('WAITING_SPACE', json.dumps({'issues': detail.get('issues', []), 'stage': detail.get('stage', '准备LPAI空间')}), task, 'model_accuracy'))
            return
        workflow = detail.get('workflow')
        if workflow:
            expected = json_digest({k: v for k, v in workflow.items() if k != 'workflow_sha256'})
            if (workflow.get('schema') != 'roadmodel.delivery-workflow/v1'
                    or workflow.get('run_id') != task or workflow.get('workflow_sha256') != expected
                    or len(workflow.get('jobs', [])) != 2
                    or {j['kind'] for j in workflow['jobs']} != {'model_accuracy', 'compile_alignment'}):
                raise ValueError('评测计划身份、完整性或哈希核验失败')
            expected_scope = self.get(task, load_json(self.config_file)['owner'])['options']['scope']
            if workflow.get('scope') != expected_scope:
                raise ValueError('评测范围与原卡片冻结范围不同')
        original_scope = self.scope
        self.scope = workflow['scope'] if workflow else original_scope
        try:
            super().complete_materials(agent, task, request_digest, detail)
        finally:
            self.scope = original_scope

    def queue_publication(self, task, report, result):
        run = self.root/'runs'/task
        workflow = load_json(run/'workflow.json')
        payload = {'task': task, 'workflow_sha256': workflow['workflow_sha256'],
                   'xml': Path(report['report_xml']).read_text(encoding='utf-8'),
                   'data': load_json(report['report_json']), 'source_sha256': report['source_sha256']}
        if payload['data']['workflow_sha256'] != workflow['workflow_sha256']:
            raise ValueError('报告与原工单输入不一致')
        value = {**result, 'state': 'WAITING_REPORT', 'evaluation_state': result['state']}
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            old = c.execute('select * from publications where task=?', (task,)).fetchone()
            signature = json_digest(payload)
            if old and old['digest'] == signature:
                return
            if old and old['digest'] != signature:
                if old['state'] not in ('QUEUED', 'VERIFIED'):
                    raise ValueError('已有报告请求内容不同，需要核对发布收据')
                previous = json.loads(old['payload'])
                if (previous['workflow_sha256'] != payload['workflow_sha256']
                        or previous['data']['task_sha256'] != payload['data']['task_sha256']):
                    raise ValueError('报告更新不属于原工单和冻结计划')
                _write_atomic(run/'superseded-publications'/(old['digest']+'.json'), dict(old))
                c.execute("update publications set digest=?,payload=?,state='QUEUED',detail='{}',updated=? where task=?",
                          (signature, json.dumps(payload, ensure_ascii=False), time.time(), task))
            c.execute('insert or ignore into publications values(?,?,?,?,?,?)',
                      (task, signature, json.dumps(payload, ensure_ascii=False), 'QUEUED', '{}', time.time()))
            if old and old['state'] == 'VERIFIED':
                value['combined_report'] = json.loads(old['detail'])
            elif (run/'progress.json').exists():
                previous = load_json(run/'progress.json').get('combined_report')
                if previous and previous.get('state') == 'VERIFIED':
                    value['combined_report'] = previous
            _write_atomic(run/'progress.json', value)
            for chain in result['chains']:
                c.execute('update chains set state=?,detail=? where task=? and kind=?',
                          (chain['state'], json.dumps(chain, ensure_ascii=False), task, chain['kind']))
            c.execute("update tasks set state='WAITING_REPORT',notice=?,updated=? where id=?",
                      ('评测状态已保存，正在发布飞书报告并核验可读性', time.time(), task))

    def claim_publication(self, agent, owner):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id'] or owner != cfg['owner']:
            raise ValueError('报告发布器身份不符')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute("select p.* from publications p join tasks t on t.id=p.task "
                            "where p.state='QUEUED' and t.owner=? order by p.updated limit 1", (owner,)).fetchone()
            if not row:
                return None
            c.execute("update publications set state='WRITING',updated=? where task=?", (time.time(), row['task']))
            return dict(row)

    def complete_publication(self, agent, task, signature, detail):
        cfg = load_json(self.config_file)
        if agent != cfg['agent_id']:
            raise ValueError('报告发布器身份不符')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('select * from publications where task=?', (task,)).fetchone()
            if not row or row['digest'] != signature:
                raise ValueError('报告发布请求身份不一致')
            if row['state'] == 'VERIFIED' and json.loads(row['detail']) == detail:
                return
            if row['state'] != 'WRITING':
                raise ValueError('报告请求已经处理，需核对原收据')
            payload = json.loads(row['payload'])
            verified = (detail.get('state') == 'VERIFIED' and detail.get('table_count') == 5
                        and detail.get('all_metric_cells_match') is True
                        and detail.get('task_sha256') == payload['data']['task_sha256']
                        and detail.get('workflow_sha256') == payload['workflow_sha256']
                        and detail.get('requester_read_verified') is True
                        and detail.get('report_url') == 'https://li.feishu.cn/docx/'+str(detail.get('document_id')))
            run = self.root/'runs'/task
            result = load_json(run/'progress.json')
            if verified:
                result.update(state=result['evaluation_state'], combined_report=detail)
            else:
                result.update(state='NEEDS_REVIEW', issues=['报告发布待核验，保留本次结果与原发布收据'])
            _write_atomic(run/'progress.json', result)
            c.execute('update publications set state=?,detail=?,updated=? where task=?',
                      ('VERIFIED' if verified else 'NEEDS_REVIEW', json.dumps(detail), time.time(), task))
            c.execute('update tasks set state=?,notice=?,updated=? where id=?',
                      (result['state'], '报告已发布并完成本人账号回读' if verified else result['issues'][0], time.time(), task))

    def tick(self, spawn=None):
        super().tick(spawn=spawn)
        # Publication is a separate queued stage after native workers exit.
        with self.connect() as c:
            rows = c.execute("select id from tasks where state='WAITING_REPORT'").fetchall()
        for row in rows:
            path = self.root/'runs'/row['id']/'progress.json'
            result = load_json(path)
            if result['state'] != 'WAITING_REPORT':
                with self.connect() as c:
                    c.execute('update tasks set state=?,updated=? where id=?',
                              (result['state'], time.time(), row['id']))
