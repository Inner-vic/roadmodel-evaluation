from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from pcm_eval.common import save_json,load_json,digest
from pcm_eval.native_dispatch import start_or_attach
from test_retained_runtime import directory

class DispatchTests(unittest.TestCase):
    def binding(self,d):
        script=d/'controller.py';script.write_text('print("preflight")\n')
        prepared=d/'preparation.json';save_json(prepared,{'state':'VERIFIED','native_task_id':'native-test','frozen_request_sha256':'frozen'})
        return {'kind':'compile_alignment','execution_root':str(d/'run'),'native_task_id':'native-test',
                'frozen_request':{'sha256':'frozen'},'launch':{'preparation_receipt':{'path':str(prepared),'sha256':digest(prepared)},
                'programs':[{'role':'compile_controller','python':sys.executable,'script':str(script),'sha256':digest(script),'cwd':str(d),'env':{}}]}}

    def test_first_dispatch_is_durable_and_repeated_submit_cannot_spawn_again(self):
        with directory() as d:
            b=self.binding(d);seen=[]
            def spawn(*a,**kw):seen.append(a);return SimpleNamespace(pid=123)
            self.assertEqual(1,start_or_attach(b,spawn=spawn)['new_dispatches'])
            with self.assertRaises(FileExistsError):start_or_attach(b,spawn=spawn)
            self.assertEqual(1,len(seen));self.assertEqual('STARTED',load_json(Path(b['execution_root'])/'compile-dispatch.json')['state'])

    def test_existing_progress_adopts_without_starting_anything(self):
        with directory() as d:
            b=self.binding(d);save_json(Path(b['execution_root'])/'compile/progress.json',{'state':'RUNNING'})
            self.assertEqual(0,start_or_attach(b,spawn=lambda *a,**kw:self.fail('must adopt'))['new_dispatches'])

    def test_changed_program_is_rejected_before_dispatch(self):
        with directory() as d:
            b=self.binding(d);Path(b['launch']['programs'][0]['script']).write_text('changed')
            with self.assertRaises(ValueError):start_or_attach(b,spawn=lambda *a,**kw:self.fail('must reject'))

    def test_unknown_dispatch_outcome_is_preserved_without_retry(self):
        with directory() as d:
            b=self.binding(d)
            def fail(*a,**kw):raise OSError('unconfirmed dispatch')
            with self.assertRaises(OSError):start_or_attach(b,spawn=fail)
            with self.assertRaises(FileExistsError):start_or_attach(b,spawn=lambda *a,**kw:self.fail('must not retry'))
            self.assertEqual('NEEDS_REVIEW',load_json(Path(b['execution_root'])/'compile-dispatch.json')['state'])

if __name__=='__main__':unittest.main()
