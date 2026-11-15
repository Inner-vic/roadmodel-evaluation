import ast
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import model_rerun_timing as timing
import prepare_timed_model_rerun as preparation
from prepare_timed_model_rerun import render, SPACE, OLD_SPACE


class TimingTests(unittest.TestCase):
    def test_wall_clock_change_does_not_change_duration(self):
        stream=io.StringIO()
        class Sink:
            def __enter__(self):return stream
            def __exit__(self,*args):pass
        with patch.object(Path,'open',return_value=Sink()), patch.object(timing,'utc_now',side_effect=['later','earlier']), patch.object(timing.time,'monotonic_ns',side_effect=[1000000000,3000000000,3000000000]):
            with timing.span('unused','stage'):pass
        rows=[json.loads(s) for s in stream.getvalue().splitlines()]
        self.assertEqual(rows[-1]['duration_seconds'],2.)
        self.assertEqual(rows[0]['token'],rows[-1]['token'])

    def test_failure_has_timing_without_replaying_work(self):
        rows=[]
        def record(path,stage,state,**fields):
            row=dict(stage=stage,state=state,monotonic_ns=10,**fields);rows.append(row);return row
        with patch.object(timing,'event',side_effect=record),patch.object(timing.time,'monotonic_ns',return_value=20):
            with self.assertRaises(ValueError):
                with timing.span('unused','stage'):raise ValueError('test')
        self.assertEqual([r['state'] for r in rows],['STARTED','FAILED'])

    def test_native_commands_and_strict_checks_unchanged(self):
        original=(TOOLS/'formal_model_native_remote.py').read_text(encoding='utf8')
        changed=render('formal_model_native_remote.py',original)
        def calls(text):
            return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run']
        self.assertCountEqual(calls(original),calls(changed))
        self.assertIn("assert binding['selected_frames'] == launch['selected_frames']",changed)
        self.assertIn("'strict_model_and_loader_preflight'",changed)

    def test_new_results_separate_from_reused_input_cache(self):
        occ=render('formal_model_controller.py',(TOOLS/'formal_model_controller.py').read_text(encoding='utf8'))
        vision=render('formal_vision_controller.py',(TOOLS/'formal_vision_controller.py').read_text(encoding='utf8'))
        self.assertIn('SPACE = '+repr(SPACE),occ)
        self.assertIn('SPACE = '+repr(SPACE),vision)
        self.assertIn(OLD_SPACE+'/vision-input/full-data-link.json',vision)
        self.assertNotIn("run_id='model-v322-full-vision-20261029'",vision)
        self.assertNotIn("SPACE.rsplit('/formal-20261029'",occ)

    def test_remote_prepare_and_start_programs_parse_without_dispatch(self):
        programs=[]
        def capture(code,*args,**kwargs):programs.append(code);return 'not executed'
        with patch.object(preparation,'call',side_effect=capture),patch.object(preparation,'upload_bundle',return_value=('/unused','0'*64)),patch.object(preparation,'upload_bundle_digest',return_value='0'*64):
            preparation.prepare()
            preparation.start()
        self.assertEqual(len(programs),2)
        for code in programs:ast.parse(code)
        self.assertIn('Inspect prior dispatch first',programs[1])
        self.assertIn('old_results_imported',programs[1])


if __name__=='__main__':unittest.main()
