"""Offline native-receipt recovery; fixture scores are never published."""
import ast
from copy import deepcopy
from concurrent.futures import Future
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pcm_eval.common import digest, json_digest, load_json, save_json
from pcm_eval.compile_execution import execute_prepared
from pcm_eval.compile_scope import prepare, verify_native_materials
from pcm_eval.gt_adapter import Counts

ROOT = Path(__file__).resolve().parents[1]


class CompileExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/'execution'
        self.native = self.output/'native-scope'
        (self.native/'compile/frames').mkdir(parents=True)
        self.frames = [['fixture-scene', 1], ['fixture-scene', 2]]
        self.request = {'kind':'compile_alignment','version_name':'offline-fixture',
                        'model':{'onnx_sha256':'onnx-fixture','packup_sha256':'packup-fixture'},
                        'evaluation':{}}
        self.plan = {'schema':'roadmodel.compile-scope-plan/v1','run_id':'offline-fixture',
                     'request':self.request,'frames':self.frames,'expected_frames':2}
        self.plan['plan_sha256'] = json_digest(self.plan)
        self.prepared = {'task_root':str(self.native), 'plan_sha256':self.plan['plan_sha256'],
                         'controller':str(self.native/'code/fast_compile_controller.py'), 'environment':{}}
        template = load_json(ROOT/'evidence/actual-runs/pcm-v7-multiscene-001/collected/cosine_summary.json')['frames'][0]
        self.per_frame=[]
        self.counts=Counts()
        for i, frame in enumerate(self.frames):
            folder=self.native/'compile/frames'/f'frame_{i:06d}'
            folder.mkdir()
            receipt={'state':'succeeded','selected_frames':[frame],'requested_frames':1,
                     'version_name':'offline-fixture','onnx_sha256':'onnx-fixture','packup_sha256':'packup-fixture',
                     'data_link_sha256':'index-fixture','finished_at':'2026-11-07T00:00:00+00:00',
                     'release':{'state':'RELEASED'},'offline_fixture':True}
            save_json(folder/'execution_receipt.json',receipt)
            save_json(folder/'input_validation.json',{'finite_outputs':{'state':'passed'}})
            save_json(folder/'output_integrity.json',{'state':'fresh_outputs_checked','finite':True})
            save_json(folder/'gt_frame_receipt.json',{'scene':frame[0],'frame_index':frame[1],
                      'frame_id':frame[0]+'_'+str(frame[1]),'offline_fixture':True})
            cosine=deepcopy(template);cosine['frame_id']=frame[0]+'_'+str(frame[1])
            save_json(folder/'cosine_summary.json',{'num_frames':1,'frames':[cosine]})
            c=Counts()
            # Unequal support proves counters are merged before IoU.
            gt=np.ones((1,1,2 if i==0 else 8),dtype=np.int32)
            pred=gt.copy() if i==0 else np.full_like(gt,2)
            c.update(pred,gt,np.ones_like(gt,dtype=bool),np.ones_like(gt))
            self.per_frame.append(c)
            self.counts.confusion+=c.confusion;self.counts.ds+=c.ds;self.counts.num_frames+=1
            for b in ('board','onnx'):save_json(folder/('gt_metrics_'+b+'.json'),c.summary())
            counts=folder/'single-counts.npz'
            with counts.open('wb') as handle:
                np.savez(handle,**{b+'_confusion':c.confusion for b in ('board','onnx')},
                         **{b+'_ds':c.ds for b in ('board','onnx')})
            save_json(folder/'single-counts-receipt.json',{'frame':frame,'counts_sha256':digest(counts),
                      'execution_sha256':digest(folder/'execution_receipt.json'),
                      'GT_sha256':digest(folder/'gt_frame_receipt.json')})
        self.checkpoint(self.counts,2)
        self.progress={'state':'SUCCEEDED','task_id':self.plan['run_id'],'expected_frames':2,'completed_frames':2}
        save_json(self.native/'compile/progress.json',self.progress)
        save_json(self.native/'dispatch.json',{'state':'STARTED','pid':99999999,
                  'plan_sha256':self.plan['plan_sha256'],'started_at':'2026-11-07T00:00:00+00:00'})

    def checkpoint(self,counts,completed):
        with (self.native/'compile/checkpoint.npz').open('wb') as f:
            np.savez(f,completed=completed,**{b+'_confusion':counts.confusion for b in ('board','onnx')},
                     **{b+'_ds':counts.ds for b in ('board','onnx')})

    def execute(self,**kwargs):
        return execute_prepared(self.plan,self.output,self.prepared,{'python':'unused'},**kwargs)

    def test_completed_native_results_collect_with_bench_disabled_and_no_inference(self):
        with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=AssertionError('No dispatch')):
            first=self.execute()
            self.assertEqual(self.execute(),first)
        report=load_json(Path(first['result_file']).parent/'report.json')
        self.assertEqual(report['result_validity'],'validated')
        self.assertEqual(len(report['tensor_alignment']['records']),36)
        result=load_json(first['result_file'])
        gt=load_json(Path(first['result_file']).parent/'metrics_board.json')
        self.assertAlmostEqual(gt['class_support'][0]['iou'],20.0)
        self.assertEqual(result['provenance']['live_run']['selected_frames'],self.frames)

    def test_collected_GT_result_generates_portable_atomic_report_outside_cwd(self):
        from pcm_eval.compile_report import write_compile_report
        receipt=self.execute()
        request=save_json(Path(self.temp.name)/'request.json',self.request)
        output=Path(self.temp.name)/'report/document.xml'
        report=write_compile_report(output,result_file=receipt['result_file'],request_file=request)
        self.assertEqual(report['frames'],2)
        xml=output.read_text(encoding='utf-8')
        self.assertIn('@./document.json',xml)
        self.assertIn('@./document.details.zip',xml)

    def test_collection_pointer_recovers_missing_outer_receipt_without_replay(self):
        receipt=self.execute()
        (self.output/'execution-receipt.json').unlink()
        with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=AssertionError('No dispatch')):
            self.assertEqual(self.execute()['result_file'],receipt['result_file'])

    def test_changed_original_frame_is_not_silently_reused(self):
        self.execute()
        path=self.native/'compile/frames/frame_000001/gt_metrics_board.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'原始结果'):self.execute()

    def test_checkpoint_must_match_the_exact_sum_of_verified_frame_counters(self):
        self.counts.confusion[1,1]+=1
        self.checkpoint(self.counts,2)
        with self.assertRaisesRegex(ValueError,'exact per-frame'):self.execute()

    def test_unknown_or_partial_native_dispatch_never_restarts(self):
        for state in ('RUNNING','STOPPED_AT_BOUNDARY','NEEDS_REVIEW'):
            with self.subTest(state=state):
                save_json(self.native/'compile/progress.json',dict(self.progress,state=state,completed_frames=1),overwrite=True)
                with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=AssertionError('No replay')):
                    with self.assertRaisesRegex(ValueError,'未完整结束'):self.execute(bench_available=True)

    def test_no_new_native_dispatch_without_explicit_bench_availability(self):
        (self.native/'dispatch.json').unlink()
        with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=AssertionError('No board')):
            with self.assertRaisesRegex(ValueError,'未获准'):self.execute()
        self.assertFalse((self.native/'dispatch.json').exists())

    def test_authorized_dispatch_occurs_once_and_report_failure_never_replays_it(self):
        (self.native/'dispatch.json').unlink();(self.native/'compile/progress.json').unlink()
        def launch(argv,**kwargs):
            self.assertEqual(argv,['unused',self.prepared['controller']])
            save_json(self.native/'compile/progress.json',self.progress)
            return SimpleNamespace(pid=123456789,poll=lambda:0,returncode=0)
        with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=launch) as child:
            with patch('pcm_eval.compile_aggregate.aggregate_compile',side_effect=TimeoutError('collection interrupted')):
                with self.assertRaises(TimeoutError):self.execute(bench_available=True)
            result=self.execute(bench_available=False)
            self.assertEqual(self.execute(),result)
            child.assert_called_once()

    def test_unknown_launch_receipt_is_not_retried(self):
        (self.native/'dispatch.json').unlink();(self.native/'compile/progress.json').unlink()
        with patch('pcm_eval.compile_execution.subprocess.Popen',side_effect=OSError('launch response unknown')) as child:
            with self.assertRaises(OSError):self.execute(bench_available=True)
            with self.assertRaises((OSError,ValueError)):self.execute(bench_available=True)
            child.assert_called_once()

    def test_active_lock_and_foreign_task_block_collection(self):
        lock=self.native/'compile/controller.lock';lock.write_text('retained-owner')
        with self.assertRaisesRegex(ValueError,'未完整结束'):self.execute()
        lock.unlink()
        save_json(self.native/'compile/progress.json',dict(self.progress,task_id='foreign'),overwrite=True)
        with self.assertRaisesRegex(ValueError,'未完整结束'):self.execute()

    def test_cached_scope_files_and_plan_are_revalidated(self):
        p=self.native/'selected-data-link.json';p.write_text('original-index')
        proof={'plan_sha256':self.plan['plan_sha256'],'code_files':[],
               'scope_files':[{'path':p.name,'sha256':digest(p)}]}
        save_json(self.native/'scope-preparation.json',proof)
        p.write_text('other-index')
        with self.assertRaisesRegex(ValueError,'范围文件'):prepare(self.plan,self.native,{})
        self.plan['frames'][0][1]=99
        with self.assertRaisesRegex(ValueError,'计划哈希'):prepare(self.plan,self.native,{})

    def test_new_scope_and_cached_scope_initialize_native_frame_parent(self):
        base=Path(self.temp.name)/'new-scope-fixture';base.mkdir()
        index=save_json(base/'index.json',{'eval':{'scene':{'frames':[{'frame_index':1}]}}})
        legacy=base/'legacy';legacy.mkdir()
        for name in ('formal_subnet_chain.py','subnet_contract.py','gt_adapter.py'):
            (legacy/name).write_text('# offline fixture')
        (legacy/'formal_compile_remote.py').write_text((ROOT/'tools/formal_compile_remote.py').read_text(encoding='utf-8'),encoding='utf-8')
        save_json(base/'gt-runtime.json',{'backend':'fixture'})
        plan={'run_id':'fixture','request':{'version_name':'v3.2.2','model':{'onnx_sha256':'fixture',
              'subnet_archives':[{'stage':i,'sha256':str(i)} for i in range(5)]}},
              'index_sha256':digest(index),'dataset_version':'fixture','frames':[['scene',1]],'scope':'validation'}
        plan['plan_sha256']=json_digest(plan)
        profile={'dataset_root':str(base),'index_path':str(index),'native_task_code':str(legacy),'native_release':str(base)}
        target=base/'prepared';prepare(plan,target,profile)
        frames=target/'compile/frames';self.assertTrue(frames.is_dir())
        frames.rmdir();prepare(plan,target,profile)
        self.assertTrue(frames.is_dir())

    def controller(self):
        # Execute the real main/finalize functions without importing hardware
        # dependencies. Engine construction or dispatch must be explicit below.
        source=ROOT/'tools/fast_compile_controller.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('main','finalize')]
        c=SimpleNamespace(ROOT=self.native/'compile',Counts=Counts,load=load_json,now=lambda:'fixture-time')
        namespace={'__file__':str(source),'os':os,'Path':Path,'np':np,'FULL':self.native,'c':c,
                   'sha':digest,'save':lambda p,d:save_json(p,d,overwrite=Path(p).exists()),
                   'TOTAL':2,'FRAMES':self.frames,'REQUEST':{'task_id':self.plan['run_id'],'model_version':'offline-fixture'},
                   'validated_settings':lambda p:{'frozen_request_sha256':digest(self.native/'request.json'),'workers':1,'backend':'memory_optimized'},
                   'reference_settings':lambda *a:{'workers':1},'Engine':Mock(side_effect=AssertionError('No engine'))}
        save_json(self.native/'request.json',namespace['REQUEST'])
        exec(compile(tree,str(source),'exec'),namespace)
        return namespace

    def test_all_frames_committed_can_finalize_without_indexing_next_frame_or_engine(self):
        namespace=self.controller()
        namespace['main']()
        namespace['Engine'].assert_not_called()
        self.assertEqual(load_json(self.native/'compile/progress.json')['state'],'SUCCEEDED')
        self.assertFalse((self.native/'compile/controller.lock').exists())

    def test_corrupt_checkpoint_and_user_pause_fail_before_engine(self):
        namespace=self.controller()
        self.checkpoint(self.counts,3)
        with self.assertRaisesRegex(ValueError,'outside'):namespace['main']()
        (self.native/'STOP_AFTER_FRAME').write_text('user pause')
        with self.assertRaisesRegex(ValueError,'paused'):namespace['main']()
        namespace['Engine'].assert_not_called()

    def test_native_materials_cannot_use_another_release_with_the_same_frame_scope(self):
        binding={'version':'v3.2.2','source_onnx_sha256':'source',
                 'subnet_archives':[{'stage':i,'sha256':str(i)} for i in range(5)]}
        source={'current':{'onnx_sha256':'source','stages':deepcopy(binding['subnet_archives'])}}
        verify_native_materials(binding,source)
        source['current']['stages'][4]['sha256']='other-model'
        with self.assertRaisesRegex(ValueError,'本次交付'):verify_native_materials(binding,source)

    def test_native_controller_resumes_after_first_committed_frame_without_repeating_it(self):
        namespace=self.controller();self.checkpoint(self.per_frame[0],1)
        first=self.native/'compile/frames/frame_000000/execution_receipt.json'
        first_digest=digest(first)
        second=self.native/'compile/frames/frame_000001/execution_receipt.json'
        receipt=load_json(second);second.unlink()
        class Immediate:
            def __init__(self,*a,**k):pass
            def submit(self,fn,*args):
                future=Future()
                try:future.set_result(fn(*args))
                except Exception as error:future.set_exception(error)
                return future
            def shutdown(self,**kwargs):pass
        engine=SimpleNamespace(producers=Immediate(),prepare=Mock(),close=Mock(),
                    run_board=Mock(side_effect=lambda args,root:save_json(root/'execution_receipt.json',receipt)))
        pipeline_calls=[]
        def pipeline(indices,where):
            pipeline_calls.append(indices)
            return iter([SimpleNamespace(meta={'scene_name':self.frames[i][0],'frame_idx':self.frames[i][1]},
                                         frame_dir=str(where)) for i in indices])
        namespace.update(Engine=lambda:engine,ProcessPoolExecutor=Immediate,make_pipeline=pipeline,
            multiprocessing=SimpleNamespace(get_context=lambda *a:None),initialize_native_gt=lambda *a:None,
            shutil=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=128*1024**3)),
            time=SimpleNamespace(time=lambda:1),hashlib=__import__('hashlib'),SimpleNamespace=SimpleNamespace,
            ensure_gt=lambda i,root:{b:(self.per_frame[i].confusion,self.per_frame[i].ds) for b in ('board','onnx')})
        with patch.dict(os.environ):namespace['main']()
        self.assertEqual(pipeline_calls,[[1]])
        engine.run_board.assert_called_once()
        self.assertEqual(digest(first),first_digest)
        with np.load(self.native/'compile/checkpoint.npz',allow_pickle=False) as cp:
            self.assertEqual(int(cp['completed']),2)
            np.testing.assert_array_equal(cp['board_confusion'],self.counts.confusion)


if __name__=='__main__':unittest.main()
