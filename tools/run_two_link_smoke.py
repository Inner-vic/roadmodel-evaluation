"""Operator-authorized two-link smoke; separate from the user's production confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json,save_json,utc_now
from pcm_eval.runtime import execute_atomic
from pcm_eval.two_link import resolve_pair

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--run-id',required=True)
    args=p.parse_args()
    output=ROOT/'evidence/feishu-entry-20261029'/args.run_id
    if output.exists():raise ValueError('Preserve existing smoke; use a new run ID')
    resolution=resolve_pair(load_json(ROOT/'examples/two_link_submission.json'),args.run_id)
    output.mkdir(parents=True)
    save_json(output/'resolution.json',resolution)
    if not args.live:
        print(json.dumps({'ready':resolution['ready'],'jobs':[{k:x[k] for k in ('kind','ready','issues')} for x in resolution['jobs']]},ensure_ascii=False));return
    receipt={'schema':'roadmodel.two-link-smoke/v1','source':'explicit_operator_request_in_codex',
             'production_callback_replayed':False,'production_confirmation_verified':False,
             'started_at':utc_now(),'scope':'small','jobs':[]}
    def run(job):
        if not job['ready']:return {'kind':job['kind'],'state':'BLOCKED','issues':job['issues']}
        return {'kind':job['kind'],**execute_atomic(job,output/job['kind'])}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for value in pool.map(run,resolution['jobs']):
            receipt['jobs'].append(value)
            save_json(output/'smoke.json',receipt,overwrite=(output/'smoke.json').exists())
    receipt['finished_at']=utc_now()
    receipt['state']='SUCCEEDED' if all(x['state']=='SUCCEEDED' for x in receipt['jobs']) else 'PARTIAL'
    save_json(output/'smoke.json',receipt,overwrite=True)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
