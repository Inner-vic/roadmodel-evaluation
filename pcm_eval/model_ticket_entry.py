"""RoadModel work-order cards on the existing authenticated Bot connection."""
import hashlib
import json
from pathlib import Path
import time

from .model_ticket import Tickets, DEFAULTS

ROOT=Path(__file__).resolve().parents[1]
COMMANDS={'RoadModel','RoadModel评测','RoadModel 评测','roadmodel','RoadModel模型评测'}
STATES={'DRAFT':'填写材料','CHECKING':'检查材料','PREPARING':'准备LPAI环境','BLOCKED':'待补充材料','QUEUED':'等待模型评测','RUNNING':'模型评测中','SUCCEEDED':'已完成','NEEDS_REVIEW':'待核验','SUPERSEDED':'已替换'}


def runtime(cfg):return Tickets(Path(cfg['state_root'])/'roadmodel/model-tickets',ROOT/'configs/model_ticket_catalog.json')


def render(task):
    d=task['detail'];value=task['inputs'];plain=lambda s:{'tag':'plain_text','content':s}
    box=lambda content:{'tag':'column_set','flex_mode':'none','background_style':'grey','columns':[{'tag':'column','width':'weighted','weight':1,'elements':[{'tag':'markdown','content':content}]}]}
    elements=[box('**'+STATES[task['state']]+'**\n材料检查 → 数据准备 → 模型评测 → 报告交付')]
    if task['state'] in ('DRAFT','BLOCKED'):
        if d.get('issues'):elements.append({'tag':'markdown','content':'\n'.join('• '+s for s in d['issues'])})
        form=[]
        def field(name,label,prompt,required):
            return {'tag':'input','name':name,'label':plain(label),'label_position':'top','placeholder':plain(prompt),
                    'default_value':value.get(name,''),'required':required,'input_type':'text','width':'fill','max_length':1000}
        form.extend([field('model_delivery_url','模型交付链接','飞书模型交付记录链接',True),
                     field('dataset_ref','评测数据','LPAI数据集版本链接，或开发空间挂载根目录/索引JSON路径',True)])
        form.append({'tag':'markdown','content':'固定执行模型默认配置的全部评测项，使用完整验证集。'})
        form.append({'tag':'button','name':'roadmodel_confirm','type':'primary_filled','width':'fill',
             'text':plain('确认执行'),'form_action_type':'submit','behaviors':[{'type':'callback','value':{'action':'bench:roadmodel_submit','entry_schema':'model-ticket/v2','task_id':task['id'],'nonce':task['nonce']}}]})
        elements.append({'tag':'form','name':'roadmodel_ticket','element_id':'roadmodel_ticket','elements':form})
    else:
        resolved=d.get('resolved',d);heads=resolved.get('heads',{})
        info=['**模型交付**\n'+value['model_delivery_url'],'**评测数据**\n'+value['dataset_ref']]
        for key,label in [('occ','OCC'),('vision','视觉分类')]:
            h=heads.get(key)
            if h:info.append('**'+label+'**\n'+h['version']+'；'+str(h['scene_count'])+'场景 / '+str(h['frame_count'])+'帧\n'+h['source_ref'])
        elements.append(box('\n\n'.join(info)))
        if d.get('stage'):elements.append({'tag':'markdown','content':d['stage']})
        for head,v in d.get('native_progress',{}).items():
            elements.append({'tag':'markdown','content':head+'：已完成 '+str(v['completed'])+' / '+str(v['expected'])+' 帧'})
        for issue in d.get('issues',[]):elements.append({'tag':'markdown','content':issue})
        if d.get('report_url'):elements.append({'tag':'button','type':'primary_filled','width':'fill','text':plain('查看模型精度报告'),
              'behaviors':[{'type':'open_url','default_url':d['report_url']}]})
    elements.append({'tag':'markdown','content':task['number'],'text_size':'notation'})
    return {'schema':'2.0','config':{'update_multi':True,'enable_forward':False,'width_mode':'default'},
            'header':{'template':'blue','title':plain('RoadModel 模型评测')},'body':{'elements':elements}}


def handle(service,session,text,event=None):
    from bench_gateway.service import identity
    from bench_gateway.card_entities import send,update
    from bench_gateway.task_cards import _client
    rt=runtime(service.cfg);owner=identity(session)
    if event:
        if event.get('value',{}).get('entry_schema')!='model-ticket/v2':return 'RoadModel入口已简化为模型链接和一份评测数据，请使用新卡片；旧卡不会执行。'
        tid=rt.submit(owner,session,event);task=rt.get(tid,owner)
        try:update(service.store,_client(service.cfg),task['message_id'],render(task),lambda:8)
        except Exception:return 'RoadModel工单已接收，将在原卡更新；不要重复提交。'
    else:
        task=rt.open(owner,session)
        mid=send(service.store,_client(service.cfg),session.split(':')[1],render(task),hashlib.sha256(task['id'].encode()).hexdigest()[:32],lambda:8)
        rt.bind(task,mid)
    return 'NO_REPLY'


def tick(cfg):return runtime(cfg).tick()


def deliver(cfg,store):
    from bench_gateway.card_entities import update
    from bench_gateway.task_cards import _client
    deadline=cfg.get('_notification_deadline',time.monotonic()+8);rt=runtime(cfg)
    with rt.db() as c:rows=[dict(r) for r in c.execute("select id,owner,sent_digest from tickets where message_id<>'' and state<>'DRAFT'")]
    for row in rows:
        if time.monotonic()>=deadline:return
        task=rt.get(row['id'],row['owner']);card=render(task);digest=hashlib.sha256(json.dumps(card,sort_keys=True).encode()).hexdigest()
        if digest==row['sent_digest']:continue
        try:update(store,_client(cfg),task['message_id'],card,lambda:max(.1,deadline-time.monotonic()))
        except Exception:return
        with rt.db() as c:c.execute('update tickets set sent_digest=? where id=?',(digest,row['id']))
        return
