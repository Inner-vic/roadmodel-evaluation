"""Read-only check of deployed runtime, original workers and report evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call

CODE="""import pathlib,sys,json,os
b=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/entry-runtime-20261030-r4');base=pathlib.Path('/mnt/data/bench-eval/service/gateway');sys.path[:0]=[str(b),str(base)]
from bench_gateway.config import load
from bench_gateway.deployment import status
from pcm_eval.runtime import Runtime
from pcm_eval.common import load_json
from pcm_eval.active_execution import poll,process_alive
cfg=load(base);s=status(cfg);rt=Runtime(pathlib.Path(cfg['state_root'])/'roadmodel',catalog_file=b/'configs/two_link_catalog.json')
with rt.connect() as c:row=c.execute('SELECT id,owner FROM tasks WHERE id=?',('rm-a361dbe3df384cccb053',)).fetchone()
t=rt.get(row['id'],row['owner']);chains=[]
for x in t['chains']:
 d=poll(x['detail']['binding']);spec=d['binding']['report_pipeline'];dispatch=load_json(spec['dispatch_receipt'])
 chains.append({'kind':x['kind'],'db_state':x['state'],'live_state':d['state'],'completed':d['completed_frames'],'expected':d['expected_frames'],
 'issues':d['issues'],'report_coordinator_alive':process_alive(dispatch['pid'],spec['script']),'proof':d['proof']})
valid=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3');v=Runtime(valid/'runtime',catalog_file=b/'configs/validation-catalog.json')
with v.connect() as c:vr=c.execute('SELECT id,owner FROM tasks').fetchone()
vt=v.get(vr['id'],vr['owner'])
print(json.dumps({'gateway':{'state':s['phase'],'active':s['active'],'instance':s['instance'],'children':s['children']},
 'production':{k:t.get(k) for k in ('id','state','message_id','sent_digest','notice')},'chains':chains,
 'validation':{'state':vt['state'],'notice':vt.get('notice'),'chains':[{'kind':x['kind'],'state':x['state'],'report_url':x['detail'].get('report_url')} for x in vt['chains']]},
 'publication':load_json(valid/'result-report-r4/publication/verification.json')},ensure_ascii=False))
"""
if __name__=='__main__':print(call(CODE))
