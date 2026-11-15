"""Read only historical task metadata and bounded semantic-related pod logs."""
import sys,json,re
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.lpai import request,select_pod,timestamp,extract_lines
ROOT=Path('evidence/model-evaluation/training-semantic-audit-20261027');ROOT.mkdir(parents=True,exist_ok=True)
UID='8d226ad5-e769-411b-9c72-57f9dd822c0e'
def unpack(x):
    return json.loads(x) if isinstance(x,str) else x
def save(name,x):
    (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
task=unpack(request(['tasks','get','--uid',UID]))
pods=unpack(request(['tasks','pods','--uid',UID]))
rows=pods.get('items',pods.get('pods',[])) if isinstance(pods,dict) else pods
safe=[{k:p.get(k) for k in ['name','step','pod_type','create_timestamp','finish_timestamp','status']} for p in rows]
save('pods.json',safe)
pod=select_pod(rows)
start=timestamp(pod.get('create_timestamp') or task['start_timestamp'])-timedelta(minutes=2)
end=timestamp(pod.get('finish_timestamp') or task['update_timestamp'])+timedelta(minutes=2)
meta={k:task.get(k) for k in ['uid','name','namespace','cluster','status','start_timestamp','update_timestamp']}
meta.update(pod=pod['name'],pod_count=len(rows),start=start.isoformat(),end=end.isoformat())
save('task-log-scope.json',meta);print(json.dumps(meta),flush=True)
base=['logs','pod','--namespace',task['namespace'],'--cluster',task['cluster'],'--pod',pod['name'],'--start',start.isoformat(),'--end',end.isoformat(),'--direction','forward']
for name,pattern,limit in [('semantic',r'Traj|surface_label|cement|[Ee]poch|[Vv]al|[Tt]rain.*[Ll]oss|dataset.*size|[Ss]amples',10000),('end',None,100)]:
    args=base+['--limit',str(limit)]
    if pattern:args+=['--grep',pattern]
    else:args[args.index('forward')]='backward'
    result=unpack(request(args))
    returned=extract_lines(result)
    lines=[(ts,line) for ts,line in returned if re.match(r'\d\d/\d\d \d\d:\d\d:\d\d - mmengine - ',line)]
    save(name+'-response.json',{'resultType':'streams','result':[{'values':[[str(ts),line] for ts,line in lines]}], 'platform_initialization_excluded':True,'returned_before_filter':len(returned)})
    (ROOT/(name+'.log')).write_text('\n'.join(line for _,line in lines)+'\n',encoding='utf-8')
    save(name+'-receipt.json',{'count':len(lines),'returned_before_filter':len(returned),'limit':limit,'filter':pattern,'truncated':len(returned)>=limit})
    print(json.dumps({'query':name,'count':len(lines)}),flush=True)
