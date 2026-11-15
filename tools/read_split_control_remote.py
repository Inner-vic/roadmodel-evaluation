import json,pathlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/split-gt-control-20261027')
p=base/'run-01/receipt.json'
print(p.read_text() if p.exists() else json.dumps({'state':'starting'}))
log=base/'workflow-console.log'
if log.exists():print('CONSOLE_TAIL\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-14:]))
