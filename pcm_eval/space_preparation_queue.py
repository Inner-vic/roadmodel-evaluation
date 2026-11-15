"""Workstation-owned queue for PC Agent Runtime preparation, no raw data files."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PreparationQueue:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        with self.db() as c:c.execute('''CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY,ticket TEXT,owner TEXT,digest TEXT,request TEXT,state TEXT,
          client TEXT,lease TEXT,detail TEXT,created REAL,updated REAL)''')
        with self.db() as c:c.execute('CREATE TABLE IF NOT EXISTS clients(id TEXT PRIMARY KEY,heartbeat REAL)')
    @contextmanager
    def db(self):
        c=sqlite3.connect(self.root/'preparations.sqlite3',timeout=10);c.row_factory=sqlite3.Row
        try:
            with c:yield c
        finally:c.close()
    def ensure(self,ticket,owner,inputs,profile,*,revision=None):
        if not re.fullmatch(r'rmm-[a-f0-9]{20}',ticket):raise ValueError('Invalid preparation ticket')
        request={'schema':'roadmodel.space-preparation/v1','ticket_id':ticket,'owner':owner,
                 'inputs':inputs,'profile':profile,'submission_revision':revision}
        digest=signature(request);jid=ticket+'-'+digest[:16]
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            c.execute('insert or ignore into jobs values(?,?,?,?,?,?,?,?,?,?,?)',
                      (jid,ticket,owner,digest,json.dumps(request),'QUEUED','','','{}',time.time(),time.time()))
            row=dict(c.execute('select * from jobs where id=?',(jid,)).fetchone())
            if row['digest']!=digest:raise ValueError('Preparation request digest collision')
        return {'id':jid,'state':row['state'],'detail':json.loads(row['detail']),'request_sha256':digest,'updated':row['updated']}
    def claim(self,client):
        if not re.fullmatch(r'[a-f0-9]{32}',client):raise ValueError('Invalid preparation client')
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            c.execute('insert or replace into clients values(?,?)',(client,time.time()))
            active=c.execute("select * from jobs where state='RUNNING'").fetchone()
            if active:
                if active['client']!=client:return None
                request=json.loads(active['request'])
                if signature(request)!=active['digest']:raise ValueError('Preparation request changed')
                return {'id':active['id'],'lease':active['lease'],'request':request,'request_sha256':active['digest'],'detail':json.loads(active['detail'])}
            row=c.execute("select * from jobs where state='QUEUED' order by created limit 1").fetchone()
            if not row:return None
            lease=secrets.token_hex(24)
            c.execute("update jobs set state='RUNNING',client=?,lease=?,updated=? where id=? and state='QUEUED'",(client,lease,time.time(),row['id']))
            request=json.loads(row['request'])
            if signature(request)!=row['digest']:raise ValueError('Preparation request changed')
            return {'id':row['id'],'lease':lease,'request':request,'request_sha256':row['digest'],'detail':json.loads(row['detail'])}
    def update(self,jid,client,lease,state,detail):
        if state not in ('RUNNING','DATA_READY','NEEDS_REVIEW','BLOCKED'):raise ValueError('Invalid preparation state')
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('select * from jobs where id=?',(jid,)).fetchone()
            if not row or row['state']!='RUNNING' or row['client']!=client or not secrets.compare_digest(row['lease'],lease):
                raise ValueError('Preparation lease no longer belongs to this worker')
            detail={**json.loads(row['detail']),**detail}
            c.execute('insert or replace into clients values(?,?)',(client,time.time()))
            if state=='DATA_READY':
                req=json.loads(row['request']);source=detail.get('dataset_source',{});receipt=source.get('mount_receipt',{})
                if (source.get('source_ref')!=req['inputs']['dataset_ref'] or receipt.get('state')!='MOUNT_VERIFIED'
                    or receipt.get('space_uid')!=req['profile']['space_uid']
                    or receipt.get('execution_location')!='lpai' or receipt.get('workstation_data_used') is not False):
                    raise ValueError('Prepared source does not match the confirmed ticket')
            c.execute('update jobs set state=?,detail=?,updated=? where id=?',(state,json.dumps(detail),time.time(),jid))
    def context(self,jid,client,lease):
        with self.db() as c:row=c.execute('select * from jobs where id=?',(jid,)).fetchone()
        if not row or row['state']!='RUNNING' or row['client']!=client or not secrets.compare_digest(row['lease'],lease):
            raise ValueError('Preparation worker lease mismatch')
        request=json.loads(row['request'])
        if signature(request)!=row['digest']:raise ValueError('Preparation request changed')
        return request
    def pending(self):
        with self.db() as c:
            return [dict(r) for r in c.execute("select id,ticket,state,updated from jobs where state in ('QUEUED','RUNNING')")]
