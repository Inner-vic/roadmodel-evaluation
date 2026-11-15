import pathlib,json
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
out={}
for name in ['operator-build-state.json','execution_receipt.json','model-preflight.json']:
 p=base/name
 if p.exists():
  value=json.loads(p.read_text())
  if name=='execution_receipt.json':value={k:v for k,v in value.items() if k not in ['environment','outputs']}
  out[name]=value
for name in ['operator-build-launch.log','eval-launch.log','eval-console.log']:
 p=base/name
 if p.exists():out[name]=p.read_text(errors='replace')[-6000:]
print(json.dumps(out))
