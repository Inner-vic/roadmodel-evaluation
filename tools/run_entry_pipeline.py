"""Durable report coordinator; no inference or shared service lifecycle changes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.entry_pipeline import wait_and_publish
from pcm_eval.common import save_json,utc_now

if __name__=='__main__':
    try:print(wait_and_publish(sys.argv[1]))
    except Exception as error:
        save_json(Path(sys.argv[1]).with_suffix('.failure.json'),{'state':'NEEDS_REVIEW','error_type':type(error).__name__,
                  'error':str(error)[:500],'at':utc_now()},overwrite=Path(sys.argv[1]).with_suffix('.failure.json').exists())
        raise
