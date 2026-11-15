"""Read-only space status and compile-placement probe; never starts a space."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcm_eval.lpai import request
from pcm_eval.placement import PROBE_CODE

OUT=ROOT/'evidence/batch-upgrade-20261029/placement'


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.probe:
        data=request(['devspaces','get','--name','bmad-r3-a100-20261015'])
        if isinstance(data,str):data=json.loads(data)
        values=[]
        def visit(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in ('phase','status','state') and type(item) in (str,int):values.append({'field':key,'value':item})
                    elif isinstance(item,(dict,list)):visit(item)
            elif isinstance(value,list):
                for item in value:visit(item)
        visit(data)
        value={'space':'bmad-r3-a100-20261015','status_fields':values,'root_keys':list(data) if isinstance(data,dict) else [],'started_by_this_probe':False}
        (OUT/'space-status.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value));return
    from model_space_transport import run
    cfg={'board_host':'192.0.2.10','board_port':22,'paths':{
        'expected_dataset':'/mnt/datasets/smartchassisdata-v3/26-10-20-1',
        'expected_datalink':'/mnt/datasets/smartchassisdata-v3/26-10-20-1/data_link_26-10-20-1_train859_eval135_v7.json',
        'known_model_cache':'/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'}}
    value=json.loads(run(['python3','-c',PROBE_CODE,json.dumps(cfg)],timeout=50))
    value['vantage']='inside_bmad-r3-a100-20261015'
    (OUT/'space-probe.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
