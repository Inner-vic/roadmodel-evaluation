import pathlib,json
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/heads-20261027')
out={}
for name in ['workflow-state.json','workflow-console.log']:
 p=base/name
 if p.exists():out[name]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text(errors='replace')[-3000:]
for kind in ['vision','trajectory']:
 p=base/(kind+'-01');row={}
 for name in ['model-preflight.json','execution_receipt.json','trajectory_metrics.json']:
  f=p/name
  if f.exists():
   value=json.loads(f.read_text());row[name]={k:v for k,v in value.items() if k in ['state','error','dataset_size','strict_main_keys','classifier_overlay_keys','exit_code','duration_seconds','metrics','num_frames','gt_source']}
 for name in ['preflight-console.log','eval-console.log']:
  f=p/name
  if f.exists():row[name]=f.read_text(errors='replace')[-4500:]
 out[kind]=row
print(json.dumps(out))
