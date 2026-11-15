import subprocess,json,tarfile
from pathlib import Path
commit='9e16e29351d17a726ce6beca6f821de79b7ade40'
repo='inspection/roadmodel-dev-v41'
out=Path('evidence/model-evaluation/model-reference-20261027/source.tar')
if not out.exists():
 p=subprocess.run(['git','-C',repo,'archive','--format=tar',commit],capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
 out.write_bytes(p.stdout)
with tarfile.open(out) as t:
 files=[m for m in t.getmembers() if m.isfile()]
 print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'files':len(files),'largest':[(m.name,m.size) for m in sorted(files,key=lambda x:x.size,reverse=True)[:10]]}))
