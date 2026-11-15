"""Bounded Lark CLI wrapper: capture exact JSON, never echo document bodies."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('args',nargs=argparse.REMAINDER)
args=p.parse_args()
cmd=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
proc=subprocess.run(cmd+args.args,capture_output=True)
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(proc.stdout)
if proc.stderr:(out.with_suffix('.stderr.txt')).write_bytes(proc.stderr)
try:
    data=json.loads(proc.stdout.decode('utf-8-sig'))
    print(json.dumps({'exit_code':proc.returncode,'ok':data.get('ok'),'error':data.get('error'),'output':str(out)},ensure_ascii=True))
except ValueError:print(json.dumps({'exit_code':proc.returncode,'output':str(out),'json':False}))
raise SystemExit(proc.returncode)
