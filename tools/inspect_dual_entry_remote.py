"""Read-only inventory of the currently installed evaluation entry."""
import json
from pathlib import Path
import sys

base = Path('/mnt/data/bench-eval/service/gateway')
sys.path.insert(0, str(base))
from bench_gateway.config import load
from bench_gateway.deployment import status

cfg = load(base)
print(json.dumps({'service': status(cfg), 'state_root': cfg['state_root']}, default=str))
print((base/'bench_gateway/roadmodel_plugin.py').read_text())
for release in sorted(Path('/mnt/data/bench-eval/service/roadmodel').glob('model-tickets*')):
    p = release/'configs/model_ticket_catalog.json'
    if p.exists():
        cat = json.loads(p.read_text())
        print(json.dumps({'release': str(release), 'catalog': cat}, ensure_ascii=False))
import sqlite3
root = Path(cfg['state_root'])/'roadmodel/model-tickets'
if (root/'tickets.sqlite3').exists():
    with sqlite3.connect('file:'+str(root/'tickets.sqlite3')+'?mode=ro', uri=True) as db:
        print(json.dumps({'tickets': db.execute('select number,state from tickets').fetchall()}))
