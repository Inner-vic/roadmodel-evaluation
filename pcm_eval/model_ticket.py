"""One model delivery, one dataset, all configured model evaluations."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from urllib.parse import urlsplit

EXAMPLE_INPUTS = {'model_delivery_url':'https://li.feishu.cn/record/PUBREF000000000000000000001',
 'dataset_ref':'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1'}
DEFAULTS = {'model_delivery_url':'', 'dataset_ref':''}


def inputs(form):
    if not isinstance(form,dict) or set(form)!=set(DEFAULTS):raise ValueError('请提供模型交付链接和一份评测数据两个字段')
    result={k:form.get(k,v) for k,v in DEFAULTS.items()}
    if any(not isinstance(v,str) or len(v)>1000 or any(c in v for c in '\r\n\x00') for v in result.values()):raise ValueError('每项输入限1000字符，不能换行')
    result={k:v.strip() for k,v in result.items()}
    u=urlsplit(result['model_delivery_url'])
    if u.scheme!='https' or u.hostname!='li.feishu.cn' or u.username or u.password or u.port not in (None,443):raise ValueError('请填写公司飞书模型交付链接')
    for key in ['dataset_ref']:
        value=result[key]
        if value.startswith('https://'):
            u=urlsplit(value)
            if u.hostname!='lpai.lixiang.com' or u.username or u.password or u.port not in (None,443):raise ValueError('数据链接请使用LPAI地址')
        else:
            path=value.split(':',1)[-1] if value.startswith(('工作站:','开发空间:')) else value
            if not path.startswith('/') or '..' in PurePosixPath(path).parts:raise ValueError('数据请填写LPAI链接或完整Linux路径，可加“工作站:”或“开发空间:”')
    return result


def resolve(value,catalog,*,dataset_source=None):
    value=inputs(value);issues=[]
    model=next((m for m in catalog['models'] if m['url']==value['model_delivery_url']),None)
    def source(ref):
        if catalog.get('runtime_config',{}).get('execution_location')=='lpai':
            if dataset_source is None:
                issues.append('尚未完成按本次输入解析LPAI数据版本及实际挂载，不会使用空间默认数据或工作站缓存')
                return None
            d=dict(dataset_source(ref))
            receipt=d.get('mount_receipt',{})
            if (d.get('source_ref')!=ref or receipt.get('state')!='MOUNT_VERIFIED'
                or receipt.get('execution_location')!='lpai' or receipt.get('workstation_data_used') is not False
                or receipt.get('index_sha256')!=d.get('index_sha256')
                or receipt.get('asset_binding_sha256')!=d.get('asset_binding_sha256')):
                raise ValueError('本次输入与LPAI挂载/索引核验收据不一致')
            if 'frames_by_head' in d:
                frames=d.pop('frames_by_head')
                signature=hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if signature!=d.get('frame_manifest_sha256') or signature!=receipt.get('frame_manifest_sha256'):
                    raise ValueError('空间内冻结帧清单哈希不一致')
                d['occ_frames']=frames.get('occ',[]);d['vision_frames']=frames.get('vision',[])
            else:
                # Compatibility for already frozen engineering fixtures only;
                # the deployed preparation worker returns in-space manifests.
                raw=d.pop('index_bytes')
                if hashlib.sha256(raw).hexdigest()!=d['index_sha256']:
                    raise ValueError('空间内索引内容与冻结哈希不一致')
                index=json.loads(raw);d.pop('index',None)
                d['occ_frames']=[[s,f['frame_index']] for s,v in index.get('eval',{}).items() for f in v.get('frames',[])]
                d['vision_frames']=[[s,f['frame_index']] for s,v in index.get('vision_eval',{}).items() for f in v.get('frames',[])]
            return d
        matches=[d for d in catalog['datasets'] if ref in d['aliases']]
        if len(matches)!=1:return None
        d=dict(matches[0]);p=Path(d['index_path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=d['index_sha256']:return None
        index=json.loads(p.read_text());d['occ_frames']=[[s,f['frame_index']] for s,v in index.get('eval',{}).items() for f in v.get('frames',[])]
        d['vision_frames']=[[s,f['frame_index']] for s,v in index.get('vision_eval',{}).items() for f in v.get('frames',[])]
        return d
    main=source(value['dataset_ref'])
    if not model:issues.append('模型交付尚未绑定可核验的权重、配置和评测脚本')
    if not main:issues.append('主数据链接/路径尚未登记或索引已变化，需要核对所在空间、索引与GT；不会自动替换')
    required=(model or {}).get('evaluation_heads',[])
    if model and not required:issues.append('模型尚未定义完整默认评测配置，需核对交付记录')
    fields={'occ':'occ_frames','vision':'vision_frames'}
    if set(required)-set(fields):issues.append('模型默认评测项尚未全部绑定原生评测脚本，不能跳过后执行')
    if main:
        if 'occ' in required and not main['occ_frames']:issues.append('本份评测数据缺少OCC验证帧/GT，请补齐同一数据入口')
        if 'vision' in required and not main['vision_frames']:issues.append('本份评测数据缺少视觉分类验证帧/GT，请补齐同一数据入口；不会沿用其他数据集或只跑部分评测')
    heads={}
    for head in required:
        if main and head in fields:
            frames=main[fields[head]];heads[head]={'dataset_id':main['id'],'version':main['version'],
             'source_ref':value['dataset_ref'],
             'frames':frames,'scene_count':len({f[0] for f in frames}),'frame_count':len(frames),
             'index_sha256':main['index_sha256'],'index_path':main['index_path']}
    if not issues and catalog.get('runtime_config',{}).get('native_dataset_bindings'):
        bindings=catalog['runtime_config']['native_dataset_bindings']
        if any(bindings.get(h)!=v['dataset_id'] for h,v in heads.items()):issues.append('该数据入口尚未完成全部原生评测项的数据绑定，需核验适配；不会使用其他数据替代')
    if catalog.get('runtime_config',{}).get('execution_location')=='lpai' and catalog['runtime_config'].get('lpai_native_execution_ready') is False:
        issues.append('模型执行器向LPAI完整迁移尚未验收，当前不会派发旧工作站数据准备器')
    return {'ready':not issues,'issues':issues,'inputs':value,'model':model,'heads':heads,
            'dataset_mount':(main or {}).get('mount_receipt'),
            'evaluation_heads':required,'scope':'full','single_dataset':True,'compile_enabled':False}


class Tickets:
    def __init__(self,root,catalog_file,*,dataset_resolver=None):
        self.dataset_resolver=dataset_resolver
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.catalog_file=Path(catalog_file)
        with self.db() as c:c.executescript('''
          CREATE TABLE IF NOT EXISTS tickets(id TEXT PRIMARY KEY,number TEXT UNIQUE,owner TEXT,session TEXT,message_id TEXT,nonce TEXT,state TEXT,inputs TEXT,detail TEXT,created REAL,sent_digest TEXT DEFAULT '');
          CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,ticket TEXT,inputs TEXT);
        ''')
    @contextmanager
    def db(self):
        c=sqlite3.connect(self.root/'tickets.sqlite3',timeout=15);c.row_factory=sqlite3.Row
        try:
            with c:yield c
        finally:c.close()
    def get(self,tid,owner):
        with self.db() as c:r=c.execute('select * from tickets where id=? and owner=?',(tid,owner)).fetchone()
        if not r:raise ValueError('工单不存在或不属于本人')
        d=dict(r);d['inputs']=json.loads(d['inputs']);d['detail']=json.loads(d['detail']);return d
    def open(self,owner,session):
        if not re.fullmatch(r'ou_[A-Za-z0-9_-]{3,100}',owner) or session.split(':')[-1]!=owner:raise ValueError('需要本人私聊身份')
        tid='rmm-'+uuid.uuid4().hex[:20]
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE');day=time.strftime('%Y%m%d',time.gmtime(time.time()+8*3600))
            n=c.execute('select count(*) from tickets where number like ?',(f'RM-{day}-%',)).fetchone()[0]+1
            c.execute('insert into tickets(id,number,owner,session,message_id,nonce,state,inputs,detail,created) values(?,?,?,?,?,?,?,?,?,?)',
                (tid,f'RM-{day}-{n:04d}',owner,session,'',uuid.uuid4().hex,'DRAFT',json.dumps(DEFAULTS),'{}',time.time()))
        return self.get(tid,owner)
    def bind(self,task,mid):
        with self.db() as c:
            if c.execute("update tickets set message_id=? where id=? and owner=? and message_id='' and state='DRAFT'",(mid,task['id'],task['owner'])).rowcount!=1:raise ValueError('卡片已绑定')
    def submit(self,owner,session,event):
        form=inputs(event.get('form_value'));v=event['value'];tid=v.get('task_id');eid=event.get('event_id')
        if v.get('entry_schema')!='model-ticket/v2':raise ValueError('入口已简化，请使用最新RoadModel工单')
        if not isinstance(eid,str) or not eid:raise ValueError('缺少真实回调事件')
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('select * from tickets where id=? and owner=?',(tid,owner)).fetchone()
            if not row or row['session']!=session or row['message_id']!=event.get('message_id') or event.get('operator_id')!=owner or event.get('chat_id')!=session.split(':')[1]:raise ValueError('本人、会话或原卡核验失败')
            old=c.execute('select * from events where id=?',(eid,)).fetchone()
            if old:
                if old['ticket']==tid and json.loads(old['inputs'])==form:return tid
                raise ValueError('回调事件已使用')
            if row['nonce']!=v.get('nonce') or row['state'] not in ('DRAFT','BLOCKED'):raise ValueError('旧卡或工单已提交，不重复执行')
            c.execute('insert into events values(?,?,?)',(eid,tid,json.dumps(form)))
            c.execute("update tickets set state='CHECKING',inputs=?,detail='{}',nonce=?,sent_digest='' where id=?",(json.dumps(form),uuid.uuid4().hex,tid))
        return tid
    def tick(self,spawn=None):
        catalog=json.loads(self.catalog_file.read_text())
        with self.db() as c:rows=[dict(r) for r in c.execute("select * from tickets where state='RUNNING'")]
        for row in rows:
            path=self.root/'runs'/row['id'];p=path/'progress.json'
            if p.exists():
                try:d=json.loads(p.read_text())
                except json.JSONDecodeError:continue
                native={}
                for head,rel,done,total in [('OCC','model/progress.json','completed_OCC_frames','expected_OCC_frames'),('视觉分类','vision/progress.json','completed_frames','expected_frames')]:
                    q=path/'evaluation'/rel
                    if q.exists():
                        try:
                            v=json.loads(q.read_text());native[head]={'completed':v.get(done,0),'expected':v.get(total,0)}
                        except json.JSONDecodeError:pass
                if native:d['native_progress']=native
                state=d.get('state','RUNNING')
                if state=='SUCCEEDED':
                    proof=path/'publication/verification.json'
                    try:
                        v=json.loads(proof.read_text())
                        if v.get('state')!='VERIFIED' or v.get('ticket_id')!=row['id'] or not v.get('all_model_table_cells_match'):state='NEEDS_REVIEW'
                    except (OSError,json.JSONDecodeError):state='NEEDS_REVIEW'
                if state=='RUNNING' and (path/'dispatch.json').exists():
                    dispatch=json.loads((path/'dispatch.json').read_text());pid=dispatch.get('pid');proc=Path('/proc',str(pid),'cmdline')
                    if pid and (not proc.exists() or str(path).encode() not in proc.read_bytes()):
                        state='NEEDS_REVIEW';d['issues']=['运行器已退出，需核验原执行收据；不会自动重跑']
                if state not in ('RUNNING','SUCCEEDED','NEEDS_REVIEW','BLOCKED'):state='NEEDS_REVIEW'
                with self.db() as c:c.execute('update tickets set state=?,detail=? where id=?',(state,json.dumps(d),row['id']))
            elif (path/'dispatch.json').exists():
                dispatch=json.loads((path/'dispatch.json').read_text());pid=dispatch.get('pid')
                if pid and not Path('/proc',str(pid)).exists():
                    with self.db() as c:c.execute("update tickets set state='NEEDS_REVIEW',detail=? where id=?",(json.dumps({'issues':['运行进程已退出，需核验原派发；不会自动重跑']}),row['id']))
        with self.db() as c:pending=[dict(r) for r in c.execute("select * from tickets where state in ('CHECKING','PREPARING') order by created")]
        for row in pending:
            try:
                form=json.loads(row['inputs']);source=self.dataset_resolver
                prepare=catalog.get('runtime_config',{}).get('space_preparation')
                if catalog.get('runtime_config',{}).get('execution_location')=='lpai' and prepare:
                    if not any(m['url']==form['model_delivery_url'] for m in catalog['models']):
                        raise ValueError('模型交付尚未绑定可核验的权重、配置和评测脚本')
                    profile=prepare.get('by_owner',{}).get(row['owner'])
                    if not profile:raise ValueError('Agent Runtime尚未配置本人的模型环境，不会套用其他人的空间')
                    from .space_preparation_queue import PreparationQueue
                    prep=PreparationQueue(self.root/'space-preparation').ensure(row['id'],row['owner'],form,profile,revision=row['nonce'])
                    if prep['state'] in ('QUEUED','RUNNING'):
                        detail={'stage':prep['detail'].get('stage','正在准备LPAI空间与本次数据挂载'),
                                'preparation_id':prep['id'],'inputs':form}
                        with self.db() as c:c.execute("update tickets set state='PREPARING',detail=? where id=?",(json.dumps(detail),row['id']))
                        continue
                    if prep['state']!='DATA_READY':
                        reason=prep['detail'].get('reason','空间准备结果待核验')
                        with self.db() as c:c.execute("update tickets set state='NEEDS_REVIEW',detail=? where id=?",(json.dumps({'issues':[reason],'preparation_id':prep['id']}),row['id']))
                        continue
                    prepared=prep['detail']['dataset_source'];source=lambda ref:prepared
                result=resolve(form,catalog,dataset_source=source)
                if row['owner'] not in catalog.get('delivery_by_owner',{}):result['issues'].append('本人尚未配置模型运行环境和报告交付位置');result['ready']=False
            except Exception as e:result={'ready':False,'issues':['材料检查未完成：'+str(e)[:180]]}
            with self.db() as c:c.execute('update tickets set state=?,detail=? where id=?',('QUEUED' if result['ready'] else 'BLOCKED',json.dumps(result),row['id']))
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute("select 1 from tickets where state='RUNNING'").fetchone():return
            row=c.execute("select * from tickets where state='QUEUED' order by created limit 1").fetchone()
            if not row:return
            c.execute("update tickets set state='RUNNING' where id=?",(row['id'],))
        run=self.root/'runs'/row['id'];run.mkdir(parents=True,exist_ok=False)
        request={'ticket_id':row['id'],'number':row['number'],'owner':row['owner'],'resolved':json.loads(row['detail']),
                 'runtime_config':catalog['runtime_config'],'delivery':catalog['delivery_by_owner'][row['owner']]}
        (run/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
        receipt={'state':'DISPATCHING','request_sha256':hashlib.sha256((run/'request.json').read_bytes()).hexdigest()}
        (run/'dispatch.json').write_text(json.dumps(receipt))
        try:
            with (run/'worker.log').open('xb') as f:
                p=(spawn or subprocess.Popen)([catalog['python'],catalog['worker'],str(run)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(state='STARTED',pid=p.pid);(run/'dispatch.json').write_text(json.dumps(receipt))
        except Exception:
            with self.db() as c:c.execute("update tickets set state='NEEDS_REVIEW',detail=? where id=?",(json.dumps({'issues':['派发状态待核验，不自动重试']}),row['id']))
            raise
