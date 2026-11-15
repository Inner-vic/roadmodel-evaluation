"""Extract repository location and revision evidence, withholding all credentials."""
import json,sys,re,hashlib
from pathlib import Path
from urllib.parse import urlsplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request,extract_lines
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/camera-GT-mapping-20261107';cli='C:/Users/developer/AppData/Local/bin/lpai-agent.exe'
task=request(['tasks','get','--uid','81e922d6-ee86-483c-9366-d5e3f6aca940'],cli=cli)
data=task.get('data',{});data=json.loads(data) if isinstance(data,str) else data
repos=[]
for item in data.get('lpai_git',[]):
    value=item.get('url','');url=urlsplit(value) if '://' in value else None
    if url:location={'host':url.hostname,'repo_path':url.path}
    else:
        m=re.fullmatch(r'(?:[^@]+@)?([^:]+):(.+)',value);location={'host':m[1],'repo_path':m[2]} if m else {'state':'UNRECOGNIZED'}
    repos.append({**location,'branch':item.get('branch'),'recorded_commit_id':item.get('commit_id'),'clone_depth':item.get('depth')})
pods=json.loads((OUT/'task-code-provenance.json').read_text())['pods'];records=[];log_locations=[]
for pod in [p for p in pods if p['step']=='dataPreparation' and p['pod_type']=='user'][:3]:
    try:
        raw=request(['logs','pod','--namespace','vc-chassis','--cluster',task['cluster'],'--pod',pod['name'],
         '--start',pod['create_timestamp']['time'],'--end','2026-10-23T15:40:00Z','--grep','commit|Cloning|HEAD|checkout|roadmodel','--direction','forward','--limit','100'],cli=cli)
        lines=extract_lines(raw);hits=[]
        for ts,line in lines:
            for location in re.findall(r'(?:https?|ssh)://[^\s\x27\x22]+',line):
                parsed=urlsplit(location)
                if parsed.hostname and ('gitlab' in parsed.hostname or parsed.path.endswith('.git')):
                    safe={'host':parsed.hostname,'repo_path':parsed.path}
                    if safe not in log_locations:log_locations.append(safe)
            heads=re.findall(r'HEAD is now at ([a-f0-9]{7,40})|commit_id:\s*([a-f0-9]{40})',line)
            heads=[a or b for a,b in heads] if not re.search(r'token|password|secret|authorization',line,re.I) else []
            # Clone logs can contain tokens. Retain only a Git HEAD announcement,
            # never arbitrary hex strings from commands or connection arguments.
            summary=None
            if ('HEAD' in line or 'commit' in line) and not re.search(r'token|password|secret|authorization',line,re.I):
                summary=re.sub(r'https?://\S+','[URL]',line)
                summary=re.sub(r'\S*[A-Za-z0-9_+/=-]{12,}\S*','[OPAQUE]',summary)[:420]
            hits.append({'timestamp':ts,'announced_HEAD':heads,'step_summary':summary,'git_markers':[s for s in ('commit','Cloning','HEAD','checkout','roadmodel') if s in line],'line_sha256':hashlib.sha256(line.encode()).hexdigest()})
        records.append({'pod':pod['name'],'returned_lines':len(lines),'revision_evidence':hits})
    except Exception as error:records.append({'pod':pod['name'],'error_type':type(error).__name__})
proof={'repositories':repos,'log_repository_locations':log_locations,'preparation_logs':records,'new_task_submitted':False}
(OUT/'repository-revision-evidence.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
