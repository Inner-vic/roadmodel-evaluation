import json,hashlib,difflib
from pathlib import Path
name='config/3-0-0-temporal-dit.py';ref='9e16e29351d17a726ce6beca6f821de79b7ade40'
paths=[Path('inspection/release-model-complete')/ref/name,Path('inspection/release-model-source')/ref/name]
data=[p.read_bytes() for p in paths]
print(json.dumps([{'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'CRLF':b.count(b'\r\n'),'LF':b.count(b'\n'),'normalized_sha256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()} for p,b in zip(paths,data)]))
print('TEXT_EQUAL',data[0].replace(b'\r\n',b'\n')==data[1].replace(b'\r\n',b'\n'))
if data[0].replace(b'\r\n',b'\n')!=data[1].replace(b'\r\n',b'\n'):
 print(''.join(list(difflib.unified_diff(data[1].decode().splitlines(keepends=True),data[0].decode().splitlines(keepends=True)))[:160]))
