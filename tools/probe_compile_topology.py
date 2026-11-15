"""Probe alternate existing mounts and the configured workstation from space."""
import json
import subprocess
from pathlib import Path
from model_space_transport import run

resolved=subprocess.run(['ssh.exe','-G','ubuntu-workstation'],capture_output=True,text=True,check=True)
fields=dict(line.split(' ',1) for line in resolved.stdout.splitlines() if ' ' in line)
target={'host':fields['hostname'],'port':int(fields.get('port','22'))}
code='''import json,os,pathlib,socket,hashlib
target=TARGET
out={'mounted_roots':{},'datalink_candidates':[],'workstation_route':{'host':target['host'],'port':target['port']}}
name='data_link_26-10-20-1_train859_eval135_v7.json'
roots=[pathlib.Path('/lpai/dataset/smartchassisdata-v3'),pathlib.Path('/mnt/datasets/smartchassisdata-v3')]
for root in roots:
 if not root.is_dir():continue
 children=[p for p in root.iterdir() if p.is_dir()]
 out['mounted_roots'][str(root)]=[p.name for p in children]
 for p in [root,*children]:
  for candidate in [p/name,p/'DataReorganized-V161-minidataset-test'/name]:
   if candidate.is_file():
    h=hashlib.sha256(candidate.read_bytes()).hexdigest()
    out['datalink_candidates'].append({'path':str(candidate),'sha256':h})
try:
 with socket.create_connection((target['host'],target['port']),timeout=5):out['workstation_route']['tcp_reachable']=True
except OSError as error:out['workstation_route'].update(tcp_reachable=False,error_type=type(error).__name__)
print(json.dumps(out))
'''.replace('TARGET',repr(target))
value=json.loads(run(['python3','-c',code],timeout=40))
out=Path('evidence/batch-upgrade-20261029/placement/topology-probe.json')
out.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
