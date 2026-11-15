"""Initialize this existing Bot deployment using verified identity references only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pcm_eval.common import load_json,save_json
from pcm_eval.installation_cli import main

if __name__=='__main__':
    output=ROOT/'evidence/rebuild-initialization-20261109'
    target=output/'request.json'
    if not target.exists():
        value=load_json(ROOT/'evidence/initialization-20261109/request-live.json')
        value['account'].update(feishu_user_id='ou_00000000000000000000000000000001',feishu_app_id='cli_0000000000000001')
        save_json(target,value)
    cfg=ROOT/'.runtime/evaluation-environment-desktop-20261106-r3/configs/agent.json'
    raise SystemExit(main(['--request',str(target),'--state-dir',str(ROOT/'.runtime/initialization-bot-20261109'),
        '--output',str(output/'plan'),'--auth-via-deployment',str(cfg),*sys.argv[1:]]))
