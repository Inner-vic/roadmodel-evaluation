"""Deployment wrapper; the original Bot connection and legacy tasks stay intact."""
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/mnt/data/bench-eval/service/roadmodel/model-tickets-20261103-r1')
COMMANDS={'RoadModel','RoadModel评测','RoadModel 评测','roadmodel','RoadModel模型评测'}


def backend(name):
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    m=importlib.import_module('pcm_eval.'+name)
    if ROOT not in Path(m.__file__).resolve().parents:raise ImportError('Different RoadModel release loaded')
    return m


def _handle(service,session,text):
    if text.strip() in COMMANDS:return backend('model_ticket_entry').handle(service,session,text)
    from .card_ui import PREFIX,decode
    if not text.startswith(PREFIX):return None
    event=decode(service.cfg,session,text)
    if event.get('action')!='roadmodel_submit':return None
    if event.get('value',{}).get('entry_schema') in ('model-ticket/v1','model-ticket/v2'):return backend('model_ticket_entry').handle(service,session,text,event)
    return backend('feishu_entry').handle(service,session,text)


def handle(service,session,text):
    try:return _handle(service,session,text)
    except Exception as error:
        print(json.dumps({'component':'roadmodel-ticket','stage':'handle','error_type':type(error).__name__}),file=sys.stderr,flush=True)
        return 'RoadModel本次处理未完成，请核对原工单；不会自动重发或转为台架回灌任务。'


def recover(cfg):
    try:return backend('feishu_entry').recover(cfg)
    except Exception as error:print('RoadModel recovery needs review: '+type(error).__name__,file=sys.stderr)


def tick(cfg):
    for name in ['model_ticket_entry','feishu_entry']:
        try:backend(name).tick(cfg)
        except Exception as error:print('RoadModel tick needs review: '+type(error).__name__,file=sys.stderr)


def deliver(cfg,store,deadline=None):
    limits=[x for x in [deadline,cfg.get('_notification_deadline')] if x is not None]
    if limits:
        limit=min(limits)
        if time.monotonic()>=limit:return
        cfg={**cfg,'_notification_deadline':limit}
    for name in ['model_ticket_entry','feishu_entry']:
        try:backend(name).deliver(cfg,store)
        except Exception as error:print('RoadModel card delivery needs review: '+type(error).__name__,file=sys.stderr)
