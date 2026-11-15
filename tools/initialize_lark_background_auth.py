"""Complete standard CLI user login in the actual desktop worker context."""
import argparse
import json
from pathlib import Path
import subprocess

SCOPES='base:app:read base:table:read base:field:read base:record:read docx:document:create docx:document:write_only docx:document:readonly docs:document.content:read docs:document.media:upload drive:file:upload drive:drive.metadata:readonly'


def main(config, mode):
    cfg=json.loads(Path(config).read_text(encoding='utf8'))
    root=Path(cfg['local_state'])/'background-auth';root.mkdir(parents=True,exist_ok=True)
    init=root/'init.json';destination=root/(mode+'.json')
    if destination.exists():raise ValueError('Inspect existing authorization flow before retrying')
    if mode=='begin':
        if init.exists():raise ValueError('Original authorization flow must be reconciled first')
        args=['auth','login','--scope',SCOPES,'--no-wait','--json']
    else:
        value=json.loads(init.read_text(encoding='utf8'));data=value.get('data',value)
        args=['auth','login','--device-code',data['device_code']]
    p=subprocess.run([cfg['lark_executable'],*args],capture_output=True,encoding='utf8',timeout=180,
                     creationflags=subprocess.CREATE_NO_WINDOW)
    try:value=json.loads(p.stdout or p.stderr)
    except ValueError:value={'state':'NEEDS_REVIEW','exit_code':p.returncode}
    destination.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    if mode=='begin' and p.returncode==0:init.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--mode',choices=('begin','complete'),required=True);args=parser.parse_args();main(args.config,args.mode)
