"""Read one referenced LPAI model asset/version, retaining no credentials."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import utc_now
from pcm_eval.lpai import request

p=argparse.ArgumentParser();p.add_argument('--version');p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
command=['assets','modelsets','version' if args.version else 'get','--namespace','vc-chassis','--uid','d3be4c63495a45538cd470547c257d9f']
if args.version:command+=['--version',args.version]
def clean(value):
    if isinstance(value,dict):
        return {k:('[redacted]' if re.search(r'(?:^|_)(?:password|secret|api_key|access_key|access_token|refresh_token|authorization|credential)(?:$|_)',k,re.I) else clean(v)) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str) and re.search(r'(?:X-Amz-|Signature=|security-token=)',value,re.I):return value.split('?')[0]+'?[signed parameters withheld]'
    return value
record={'checked_at':utc_now(),'read_only':True,'command':command}
try:
    data=clean(request(command));record.update(state='queried',data=data)
except Exception as e:record.update(state='failed',error_type=type(e).__name__,error=str(e))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=True)[:16000])
