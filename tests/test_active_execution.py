import json
from pathlib import Path
from unittest.mock import patch
import unittest
from test_retained_runtime import directory
from pcm_eval.common import save_json,digest
from pcm_eval.active_execution import poll,verify
from pcm_eval.runtime import Runtime,execute_atomic

class ActiveTests(unittest.TestCase):
    def binding(self,path):
        request={'task_id':'formal-test','model_version':'v3.2.2','model_delivery_url':'https://li.feishu.cn/record/test',
                 'dataset_version':'26-10-20-1','frames':[['scene',1],['scene',2]]}
        save_json(path/'request.json',request)
        save_json(path/'compile/progress.json',{'state':'RUNNING','completed_frames':1,'pid':123})
        b={'schema':'roadmodel.active-execution/v1','kind':'compile_alignment','version':'v3.2.2','dataset_id':'ds',
           'frozen_request':{'path':str(path/'request.json'),'sha256':digest(path/'request.json')},
           'execution_root':str(path),'expected_frames':2,'native_task_id':'formal-test',
           'model_delivery_url':request['model_delivery_url'],'dataset_url':'https://lpai.lixiang.com/task/test',
           'delivery_receipt':str(path/'delivery.json')}
        save_json(path/'binding.json',b)
        return b,{'path':'binding.json','sha256':digest(path/'binding.json')}

    def test_native_running_is_not_report_success(self):
        with directory() as d:
            b,_=self.binding(d)
            with patch('pcm_eval.active_execution.process_alive',return_value=True):r=poll(b)
            self.assertEqual('RUNNING',r['state']);self.assertEqual(1,r['completed_frames'])
            self.assertIsNone(r['report_url'])
            self.assertEqual(0,r['repeated_completed_frames'])

    def test_missing_native_process_never_automatically_replays(self):
        with directory() as d:
            b,record=self.binding(d)
            with patch('pcm_eval.active_execution.process_alive',return_value=False),patch('pcm_eval.runner.execute',side_effect=AssertionError('no replay')):
                r=execute_atomic({'active_execution':record,'material_root':str(d),'kind':'compile_alignment','version':'v3.2.2','dataset_id':'ds'},d/'attachment')
            self.assertEqual('NEEDS_REVIEW',r['state'])

    def test_native_finished_requires_new_full_report_receipt(self):
        with directory() as d:
            b,_=self.binding(d)
            save_json(d/'compile/progress.json',{'state':'SUCCEEDED','completed_frames':2,'pid':123},overwrite=True)
            self.assertEqual('RUNNING',poll(b)['state'])
            receipt={'state':'VERIFIED','native_task_id':'formal-test','frozen_request_sha256':'wrong',
                     'scope':'full','five_tables_match':True,'report_url':'https://li.feishu.cn/docx/new'}
            save_json(d/'delivery.json',receipt);self.assertEqual('RUNNING',poll(b)['state'])
            receipt['frozen_request_sha256']=b['frozen_request']['sha256'];save_json(d/'delivery.json',receipt,overwrite=True)
            self.assertEqual('SUCCEEDED',poll(b)['state'])

    def test_frozen_input_change_cannot_attach_to_another_task(self):
        with directory() as d:
            b,record=self.binding(d);save_json(d/'request.json',{'changed':True},overwrite=True)
            with self.assertRaises(ValueError):verify(d,record,'compile_alignment','v3.2.2','ds')

    def test_restart_keeps_observing_native_work_without_dispatch(self):
        with directory() as d:
            b,_=self.binding(d);rt=Runtime(d/'runtime');owner='ou_test_active';session='feishu:oc_test_active:'+owner
            task=rt.open(owner,session)
            with rt.connect() as c:
                c.execute("UPDATE tasks SET state='RUNNING' WHERE id=?",(task['id'],))
                for kind in ('model_accuracy','compile_alignment'):
                    c.execute('INSERT INTO chains VALUES(?,?,?,?)',(task['id'],kind,'RUNNING',json.dumps({'mode':'active_native_execution','binding':b})))
            rt.recover();self.assertEqual('RUNNING',rt.get(task['id'],owner)['state'])
            with patch('pcm_eval.active_execution.process_alive',return_value=True),patch('pcm_eval.runner.execute',side_effect=AssertionError('no replay')):
                rt.tick()
            self.assertEqual('RUNNING',rt.get(task['id'],owner)['state'])

if __name__=='__main__':unittest.main()
