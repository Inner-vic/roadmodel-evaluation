"""Validate an unsent card; optionally open the authorized user's real draft, never submit it."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
GATEWAY=Path('D:/developer/Desktop/Project/台架飞书统一入口')
sys.path[:0]=[str(ROOT),str(GATEWAY)]
from pcm_eval.common import save_json
from pcm_eval.feishu_entry import render,handle
from bench_gateway.config import load
from bench_gateway.core import Store
from bench_gateway.feishu_files import FeishuFiles
from bench_gateway.card_entities import encoded

OWNER='ou_00000000000000000000000000000001'


def main():
    p=argparse.ArgumentParser();p.add_argument('--open',action='store_true');args=p.parse_args()
    cfg=load(GATEWAY)
    draft={'id':'rm-schema-only','nonce':'schema-only-not-bound','state':'DRAFT','chains':[]}
    card=render(draft)
    # No message is sent by this CardKit schema check, and its nonce is not registered.
    response=FeishuFiles(cfg)._json('/cardkit/v1/cards',body=encoded({'type':'card_json','data':encoded(card)}).encode())
    receipt={'native_card_schema_accepted':True,'card_id':response['data']['card_id'],
             'required_fields':['model_delivery_url','dataset_url'],'confirmation_callback_verified':False,
             'hardware_dispatched':False,'inference_frames':0,'production_form_opened':False}
    if args.open:
        db=Path(cfg['state_root'])/'gateway.sqlite3'
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
            row=c.execute('SELECT session FROM input_cards WHERE owner=? ORDER BY last_sent DESC LIMIT 1',(OWNER,)).fetchone()
        if not row or not row[0].startswith('feishu:oc_'): raise ValueError('No verified private session for the requesting owner')
        from pcm_eval.feishu_entry import runtime
        rt=runtime(cfg)
        with rt.connect() as c:
            old=c.execute("SELECT id,message_id FROM tasks WHERE owner=? AND state='DRAFT' AND message_id<>'' ORDER BY created DESC LIMIT 1",(OWNER,)).fetchone()
        if old: receipt.update(production_form_opened=True,task_id=old['id'],message_id=old['message_id'],reused=True)
        else:
            result=handle(SimpleNamespace(cfg=cfg,store=Store(cfg['state_root'])),row[0],'RoadModel评测')
            assert result=='NO_REPLY'
            with rt.connect() as c: task=c.execute('SELECT id,message_id FROM tasks WHERE owner=? ORDER BY created DESC LIMIT 1',(OWNER,)).fetchone()
            receipt.update(production_form_opened=True,task_id=task['id'],message_id=task['message_id'])
    out=ROOT/'evidence/feishu-entry-20261029/card-validation.json'
    save_json(out,receipt,overwrite=out.exists())
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
