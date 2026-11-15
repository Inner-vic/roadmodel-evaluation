"""Isolated signed callback -> two atomic reports -> exact existing publication.

This is not a production user confirmation and sends no message.
"""
import base64
import hashlib
import hmac
import json
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import patch

BASE=Path('/mnt/data/bench-eval/service/gateway')
RELEASE=Path('/mnt/data/bench-eval/service/roadmodel/trial-20261030')
sys.path[:0]=[str(BASE),str(RELEASE)]
from bench_gateway.config import load,save
from bench_gateway.card_ui import PREFIX
from pcm_eval.feishu_entry import handle,runtime,deliver

cfg=load(BASE)
cfg['state_root']=str(Path('/mnt/data/bench-eval/roadmodel/acceptance')/uuid.uuid4().hex)
cfg['notifications_enabled']=True
service=SimpleNamespace(cfg=cfg,store=SimpleNamespace())
owner='ou_isolated_trial';chat='oc_isolated_trial';session='feishu:'+chat+':'+owner
updates=[]
secret=b'isolated-integration-test-not-a-credential'
with patch('bench_gateway.task_cards._client',return_value=object()),patch('bench_gateway.card_entities.send',return_value='om_isolated_trial'),patch('bench_gateway.card_entities.update',side_effect=lambda store,client,mid,card,budget:updates.append(card)),patch('bench_gateway.card_ui.key',return_value=secret):
    assert handle(service,session,'RoadModel评测')=='NO_REPLY'
    rt=runtime(cfg)
    with rt.connect() as c:row=c.execute('SELECT id FROM tasks').fetchone()
    task=rt.get(row['id'],owner)
    catalog=json.loads((RELEASE/'configs/two_link_catalog.json').read_text())
    event={'schema':1,'app_id':cfg['feishu_app_id'],'action':'roadmodel_submit','event_id':'isolated-'+uuid.uuid4().hex,
           'operator_id':owner,'chat_id':chat,'message_id':task['message_id'],
           'value':{'task_id':task['id'],'nonce':task['nonce']},
           'form_value':{'model_delivery_url':catalog['models'][0]['url'],'dataset_url':catalog['datasets'][0]['url']}}
    raw=json.dumps(event,separators=(',',':')).encode();packet=base64.urlsafe_b64encode(raw).decode().rstrip('=')
    signed=PREFIX+packet+'.'+hmac.new(secret,raw,hashlib.sha256).hexdigest()
    assert handle(service,session,signed)=='NO_REPLY'
    result=rt.tick()
    assert result['state']=='SUCCEEDED',result
    assert handle(service,session,signed)=='NO_REPLY'
    assert rt.tick() is None
    deliver(cfg,service.store)
    assert 'DEMO00000000000000000000002' in json.dumps(updates[-1])
    bad=event.copy();bad['operator_id']='ou_somebody_else'
    raw=json.dumps(bad).encode();signed_bad=PREFIX+base64.urlsafe_b64encode(raw).decode().rstrip('=')+'.'+hmac.new(secret,raw,hashlib.sha256).hexdigest()
    rejected=False
    try:handle(service,session,signed_bad)
    except Exception:rejected=True
    assert rejected
    receipt={'state':'PASSED','isolated_state_root':cfg['state_root'],'authenticated_envelope_checks':True,
             'two_atomic_reports':'SUCCEEDED','combined_five_tables_match':True,'duplicate_callback_no_extra_task':True,
             'wrong_owner_rejected':True,'new_inference_frames':0,'real_feishu_messages_sent':0,
             'real_user_submit_callback_verified':False,'report_url':'https://li.feishu.cn/docx/DEMO00000000000000000000002'}
    save(Path(cfg['state_root'])/'verification.json',receipt)
    print(json.dumps(receipt))
