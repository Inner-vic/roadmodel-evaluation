import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request
ROOT=Path('evidence/model-evaluation/training-semantic-audit-20261027')
def unpack(x):return json.loads(x) if isinstance(x,str) else x
t=unpack(request(['tasks','get','--uid','8d226ad5-e769-411b-9c72-57f9dd822c0e']))
def scrub(v):
    if isinstance(v,dict):return {k:('[redacted]' if any(x in k.lower() for x in ['secret','token','credential','password','access_key','sign','authorization']) else scrub(x)) for k,x in v.items()}
    if isinstance(v,list):return [scrub(x) for x in v]
    return v
selected={k:scrub(v) for k,v in t.items() if any(x in k.lower() for x in ['output','artifact','log','tensor','result'])}
(ROOT/'output-metadata.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(selected,ensure_ascii=False)[:7000])
m=json.loads((ROOT/'task-log-scope.json').read_text())
base=['logs','pod','--namespace',m['namespace'],'--cluster',m['cluster'],'--pod',m['pod'],'--start',m['start'],'--end',m['end'],'--direction','forward','--limit','3000','--grep',r'\[Dataset|\[dataset|[Cc]lass.*([Cc]ount|[Dd]istrib|[Hh]ist)|[Cc]ement|[Ss]plit|[Tt]otal.*([Ff]rame|[Ss]ample)|[Tt]raj.*[Ll]oad|[Ff]reez']
response=unpack(request(base))
from pcm_eval.lpai import extract_lines
lines=[line for _,line in extract_lines(response) if not line.startswith('{')]
(ROOT/'distribution-and-transfer.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'distribution_log_lines':len(lines),'short_lines':[x for x in lines if len(x)<300][:25]},ensure_ascii=False))
