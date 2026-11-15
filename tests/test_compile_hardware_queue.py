"""New-card FIFO admission uses real bindings and never invokes hardware in tests."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, Mock

import pcm_eval.compile_hardware_queue as q
from pcm_eval.common import save_json, digest, json_digest


class Rejected(Exception):pass


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'roadmodel/evaluation'
        self.tid='rm-123abc';self.run=self.root/'runs'/self.tid;self.run.mkdir(parents=True)
        self.plan={'schema':'roadmodel.compile-scope-plan/v1','run_id':self.tid+'-compile'}
        self.plan['plan_sha256']=json_digest(self.plan)
        self.task={'id':self.tid,'owner':'owner','state':'RUNNING','event_id':'real-event','session':'private-session',
                   'inputs':{'model_delivery_url':'delivery','dataset_url':'dataset'}}
        self.job={'kind':'compile_alignment','ready':True,'native_adapter':'compile_full_scope','plan':self.plan}
        self.workflow={'run_id':self.tid,'inputs':{'model_delivery_url':'delivery','dataset_ref':'dataset'},'jobs':[self.job]}
        self.workflow['workflow_sha256']=json_digest(self.workflow);save_json(self.run/'workflow.json',self.workflow)
        self.cfg={'owner':'owner','bench_available':True,'compile_profile':{},'shared_hardware_queue':{
            'enabled':True,'action':'roadmodel_hardware','adapter_sha256':digest(Path(q.__file__))}}
        self.config=save_json(self.base/'config.json',self.cfg)
        self.db=self.base/'fixture.sqlite3'
        self.connections=[]
        def connect():
            db=sqlite3.connect(self.db);self.connections.append(db);return db
        self.addCleanup(lambda:[db.close() for db in self.connections])
        with connect() as db:
            db.execute('create table chains(task,kind,state,detail)')
            db.execute('insert into chains values(?,?,?,?)',(self.tid,'compile_alignment','PENDING','{}'))
            db.execute('create table jobs(id,owner,state,result,updated)')
            db.execute('insert into jobs values(?,?,?,?,?)',('gw-fixture','owner','RUNNING','{}',0))
        def get(tid,owner):
            if tid!=self.tid or owner!='owner':raise ValueError('foreign identity')
            return self.task
        self.rt=SimpleNamespace(root=self.root,config_file=self.config,get=get,connect=connect)
        self.gcfg={'state_root':str(self.base),'project_root':str(self.base/'service')}
        self.current={};self.payload=None;self.calls=0
        self.native={'state':'succeeded','plan_sha256':self.plan['plan_sha256'],'result_file':'verified-result'}
        def submit(owner,session,key,payload):
            self.calls+=1;self.payload=payload
            self.current={'id':'gw-fixture','owner':owner,'session':session,'payload':payload,'state':'RUNNING','result':{}}
            return self.current,False
        def store_job(owner,jid):
            if owner!='owner' or jid!='gw-fixture':raise ValueError('foreign gateway job')
            return self.current
        self.store=SimpleNamespace(submit=submit,job=store_job,connect=connect)
        for p in (patch.object(q,'EvaluationRuntime',return_value=self.rt),
                  patch.object(q,'services',return_value=(self.gcfg,self.store))):
            p.start();self.addCleanup(p.stop)
        core=ModuleType('bench_gateway.core');core.PreDispatchRejected=Rejected
        p=patch.dict(sys.modules,{'bench_gateway.core':core});p.start();self.addCleanup(p.stop)
        self.target=self.run/'effective/compile_alignment/execution'

    def request(self):
        def finish(_):self.current.update(state='SUCCEEDED',result={'detail':self.native})
        with patch.object(q.time,'sleep',side_effect=finish):
            return q.submit_and_wait(self.plan,self.target,self.cfg)

    def test_worker_submits_fifo_unit_and_collects_without_direct_native_call(self):
        with patch('pcm_eval.compile_scope.execute',side_effect=AssertionError('No direct native call')):
            self.assertEqual(self.request(),self.native)
        self.assertEqual(self.calls,1)
        self.assertEqual(self.payload['action'],'roadmodel_hardware')
        self.assertEqual(self.payload['entry_runtime'],q.ENTRY)

    def test_three_inputs_bind_both_sources_and_compile_uses_occ(self):
        self.task['inputs']={'model_delivery_url':'delivery','occ_dataset_url':'dataset','vision_dataset_url':'visual'}
        self.workflow['inputs']={'model_delivery_url':'delivery','dataset_ref':'dataset','vision_dataset_ref':'visual'}
        self.workflow.pop('workflow_sha256')
        self.workflow['workflow_sha256']=json_digest(self.workflow)
        (self.run/'workflow.json').write_text(json.dumps(self.workflow))
        self.assertEqual(self.request(),self.native)
        self.assertEqual(self.calls,1)
        self.task['inputs']['vision_dataset_url']='foreign-visual'
        self.current['state']='RUNNING'
        with self.assertRaises(Rejected):q.execute_queued(self.gcfg,self.store,self.current)

    def test_only_claimed_gateway_worker_invokes_exact_frozen_native_plan(self):
        self.request();self.current.update(state='RUNNING',result={})
        with patch('pcm_eval.compile_scope.execute',return_value=self.native) as native:
            result=q.execute_queued(self.gcfg,self.store,self.current)
        native.assert_called_once_with(self.plan,str(self.target.resolve()),{},bench_available=True)
        self.assertEqual(result['detail'],self.native)

    def test_unknown_gateway_result_is_not_retried(self):
        def fail(_):self.current.update(state='NEEDS_REVIEW')
        with patch.object(q.time,'sleep',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'NEEDS_REVIEW'):
                q.submit_and_wait(self.plan,self.target,self.cfg)
        self.assertEqual(self.calls,1)

    def test_completed_or_unclaimed_unit_cannot_dispatch_again(self):
        self.request()
        with patch('pcm_eval.compile_scope.execute',side_effect=AssertionError('No replay')):
            with self.assertRaises(Rejected):q.execute_queued(self.gcfg,self.store,self.current)

    def test_tampered_confirmed_input_is_rejected_before_hardware(self):
        self.request();self.current['state']='RUNNING'
        self.task['inputs']['dataset_url']='other-source'
        with patch('pcm_eval.compile_scope.execute',side_effect=AssertionError('No foreign dispatch')):
            with self.assertRaises(Rejected):q.execute_queued(self.gcfg,self.store,self.current)

    def test_foreign_output_or_unverified_adapter_binding_is_rejected(self):
        with self.assertRaises(ValueError):q.submit_and_wait(self.plan,self.base/'foreign/a/b/execution',self.cfg)
        bad=copy.deepcopy(self.cfg);bad['shared_hardware_queue']['adapter_sha256']='other-code'
        with self.assertRaises(ValueError):q.submit_and_wait(self.plan,self.target,bad)
        self.assertEqual(self.calls,0)


if __name__=='__main__':unittest.main()
