"""Bounded evidence audit; this tool never dispatches an evaluation."""
import collections,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/vision-gt-audit-20261107';OUT.mkdir(exist_ok=True)

def local():
    index=ROOT/'evidence/requested-dataset/data_link_26-10-20-1_train859_eval135_v7.json'
    data=json.loads(index.read_text());splits={}
    for key,scenes in data.items():
        fields=collections.Counter(k for info in scenes.values() for frame in info.get('frames',[]) for k in frame)
        splits[key]={'scenes':len(scenes),'frames':sum(len(s.get('frames',[])) for s in scenes.values()),'frame_fields':dict(fields),
            'scene_fields':dict(collections.Counter(k for info in scenes.values() for k in info)),
            'example':next(({'scene':s,'scene_metadata':{k:v for k,v in info.items() if k!='frames'},'first_frame':info['frames'][0]} for s,info in scenes.items() if info.get('frames')),None)}
    inventory=json.loads((ROOT/'evidence/requested-dataset/eval135-file-inventory.json').read_text())
    counts=collections.Counter(Path(f['path']).suffix for f in inventory['files'])
    labels=[f for f in inventory['files'] if any(x in Path(f['path']).name.lower() for x in ('label','class','vision'))]
    with tarfile.open(ROOT/'evidence/model-evaluation/model-reference-20261027/source.tar') as tar:
        for name in ('dataset/dataset.py','dataset/__init__.py'):
            path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(name).read())
    proof={'index_splits':splits,'existing_required_file_inventory':{'files':len(inventory['files']),'extensions':dict(counts),'label_named_examples':labels[:10],'note':'Required files only; not evidence of all objects in dataset'}}
    (OUT/'local-evidence.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))

def remote_inventory():
    result=remote('''import pathlib,json,subprocess,importlib.util
code='import importlib.util;print(importlib.util.find_spec('+repr('asset')+') is not None)'
p=subprocess.run(['/home/lixiang/miniconda3/envs/tvg/bin/python','-c',code],capture_output=True,text=True)
base=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection')
print(json.dumps({'asset_here':importlib.util.find_spec('asset') is not None,'asset_other_python':p.stdout,'inspection_dirs':[p.name for p in base.iterdir()] if base.exists() else []}))
''')
    print(json.dumps(result))

if __name__=='__main__':
    local() if len(sys.argv)<2 else remote_inventory()
