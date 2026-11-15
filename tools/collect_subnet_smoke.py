"""Collect bounded small-run JSON receipts; verify original Runner is released."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_subnet_remote import remote
from pcm_eval.common import save_json

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
    import re
    if not re.fullmatch(r'pcm-v322-subnets-[A-Za-z0-9_-]{1,40}',args.run_id):raise ValueError('Invalid run ID')
    code=r'''import json,pathlib,subprocess
root=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/subnet-runs/RUN_ID')
names=['prepare.json','chain-contract.json','input-reuse.json','public-outputs.json','all-subnet-outputs.json','execution.json','tensor-comparison.json']
out={'files':{name:json.loads((root/name).read_text()) for name in names if (root/name).is_file()}}
p=subprocess.run(['/home/lixiang/bench-eval/bin/bench-eval','queue'],capture_output=True,text=True,timeout=15)
q=json.loads(p.stdout) if p.returncode==0 else {}
out['guard_final']={'queue_readable':p.returncode==0,'active':q.get('active'),'queued_count':len(q.get('queued',[])),
 'waiting_count':len(q.get('waiting',[])),'lock_exists':pathlib.Path('/home/lixiang/bench-eval/locks/bench.lock').exists()}
print(json.dumps(out))'''.replace('RUN_ID',args.run_id)
    value=json.loads(remote(code));out=ROOT/'evidence/compile-subnet-review-20261029'/args.run_id
    out.mkdir(exist_ok=True)
    for name,content in value['files'].items():save_json(out/name,content,overwrite=(out/name).exists())
    save_json(out/'guard-final.json',value['guard_final'],overwrite=(out/'guard-final.json').exists())
    execution=value['files'].get('execution.json',{})
    comparison=value['files'].get('tensor-comparison.json',{})
    print(json.dumps({'execution':{k:v for k,v in execution.items() if k!='stages'},
        'stages':[{k:x[k] for k in ('stage','state','exit_code','output_count','seconds') if k in x} for x in execution.get('stages',[])],
        'comparisons':comparison.get('outputs'),'guard_final':value['guard_final']},ensure_ascii=False,indent=2))
