"""Compare saved tensor storage bytes across the two provided checkpoints.

This checks component provenance/equality, not inference or accuracy.
"""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path('evidence/model-evaluation/model-reference-20261027/weights')
summaries=[]
for version,filename in [('26-09-08-2','epoch_44.pth'),('26-09-08-1','epoch_98.pth')]:
 meta=json.loads((root/version/'metadata-inspection.json').read_text())
 values={};storage_hash={}
 with zipfile.ZipFile(root/version/filename) as archive:
  prefix=next(i.filename.rsplit('/',1)[0] for i in archive.infolist() if i.filename.endswith('/data.pkl'))
  for name,row in meta['tensor_metadata'].items():
   key=str(row['storage_key'])
   if key not in storage_hash:
    h=hashlib.sha256()
    with archive.open(prefix+'/data/'+key) as stream:
     for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    storage_hash[key]=h.hexdigest()
   values[name]={k:row[k] for k in ['shape','dtype','storage_offset','stride']}|{'storage_sha256':storage_hash[key]}
 summaries.append(values)
groups={}
for key in ['base','trajectory','classifier']:
 names=[n for n in summaries[0] if (n.startswith('head.traj_head.') if key=='trajectory' else n.startswith('classifier.') if key=='classifier' else not n.startswith(('head.traj_head.','classifier.')))]
 groups[key]={'keys':len(names),'equal':sum(summaries[0][n]==summaries[1].get(n) for n in names),
              'different':[n for n in names if summaries[0][n]!=summaries[1].get(n)]}
output={'state':'compared','model_executed':False,'method':'shape/dtype/view metadata plus exact SHA256 of stored tensor bytes',
        'components':groups,'unmatched_keys':sorted(set(summaries[0])^set(summaries[1]))}
path=root/'component-comparison.json';path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='components'}|{'components':{k:{'keys':v['keys'],'equal':v['equal'],'different_count':len(v['different'])} for k,v in groups.items()}}))
