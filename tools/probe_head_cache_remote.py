import json,pathlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
out={'previous_run_exists':(base/'execution_receipt.json').exists(),'weights':[(p.name,p.stat().st_size) for p in (base/'weights').glob('*.pth')],
     'cache_roots':{},'existing_sources':[]}
for p in [pathlib.Path('/lpai/dataset/smartchassisdata-v3'),pathlib.Path('/mnt/datasets/smartchassisdata-v3')]:
 if p.is_dir():out['cache_roots'][str(p)]=[c.name for c in p.iterdir()][:30]
scene='2026-09-08/2026-07-07_10-09-25_HLX33B166R1997378'
for root in ['/mnt/datasets/smartchassisdata-v3/26-08-14-0','/lpai/dataset/smartchassisdata-v3/26-08-14-0','/mnt/datasets/smartchassisdata-v3/1-0-10']:
 for sub in ['DataReorganized-V151','DataReorganized-VisionHead','DataReorganized-VisionHead-standard']:
  p=pathlib.Path(root)/sub
  if p.is_dir():out['existing_sources'].append(str(p))
print(json.dumps(out))
