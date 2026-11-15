"""Release the automatic model-space preparation path, without model inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from connect_active_entry import call

ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/model-tickets-20261103-r3'
STAGE='/mnt/data/bench-eval/roadmodel/deployment/model-tickets-20261103-r3'
PATTERNS=['test_dataset_mount.py','test_dataset_preparation.py','test_space_preparation_queue.py',
          'test_model_space_desktop.py','test_model_ticket.py','test_strict_model_template.py']


def prepare():
    folder=ROOT/'inspection/space-preparation-release';folder.mkdir(exist_ok=True,parents=True)
    archive=folder/'code.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in (ROOT/'pcm_eval').glob('*.py'):t.add(p,arcname=p.relative_to(ROOT).as_posix())
        for name in ['tools/model_ticket_worker.py','tools/model_ticket_plugin.py','tools/model_space_queue_remote.py','tools/model_space_desktop.py',
                     'schemas/space-preparation-result.schema.json','skills/roadmodel-space-preparation/SKILL.md',
                     'skills/roadmodel-model-accuracy/SKILL.md']+['tests/'+p for p in PATTERNS]:t.add(ROOT/name,arcname=name)
    signature=hashlib.sha256(archive.read_bytes()).hexdigest();remote=STAGE+'-code-'+signature+'.tar.gz'
    batch=folder/'upload.sftp';batch.write_text('put "'+archive.as_posix()+'" "'+remote+'"\n')
    p=subprocess.run([shutil.which('sftp'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-b',str(batch),'ubuntu-workstation'],capture_output=True,timeout=80)
    if p.returncode:raise ValueError('Release upload failed')
    code='RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\nARCHIVE='+repr(remote)+'\nSIGNATURE='+repr(signature)+'\nPATTERNS='+repr(PATTERNS)+'\n'+r'''
import pathlib,tarfile,hashlib,shutil,json,sys,subprocess,os,ast,uuid
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE);old=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/model-tickets-20261103-r1')
assert not (stage/'deployment.json').exists(),'Release already deployed'
stage.mkdir(exist_ok=True,parents=True)
archive=pathlib.Path(ARCHIVE);assert hashlib.sha256(archive.read_bytes()).hexdigest()==SIGNATURE
if not release.exists():shutil.copytree(old,release)
with tarfile.open(archive) as t:
 for m in t:
  assert m.isfile();p=(release/m.name).resolve();p.relative_to(release);body=t.extractfile(m).read()
  if p.suffix=='.py':ast.parse(body)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
plugin=release/'tools/model_ticket_plugin.py';plugin.write_text(plugin.read_text().replace(str(old),str(release)))
catalog=json.loads((old/'configs/model_ticket_catalog.json').read_text());rc=catalog['runtime_config']
profile={'namespace':'vc-chassis','space_name':'bmad-r3-a100-20261015','space_uid':'45f7a65d-0d2d-4c15-b538-eacabd48019d',
         'cluster':'cnhb01-algorithm-bidu-p-03','ssh_host':'ssh-d.lpai.lixiang.com','ssh_port':31017,'identity_file':rc['identity_file']}
rc.update(execution_location='lpai',lpai_native_execution_ready=False,
          space_preparation={'by_owner':{owner:profile for owner in catalog['delivery_by_owner']}})
rc.pop('native_dataset_bindings',None)
catalog['datasets']=[];catalog['worker']=str(release/'tools/model_ticket_worker.py')
(release/'configs/model_ticket_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
tests=[]
for pattern in PATTERNS:
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=release,env={**os.environ,'PYTHONPATH':str(release)},capture_output=True,text=True,timeout=60)
 (stage/(pattern+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,pattern+': '+p.stderr[-2500:];tests.append(pattern)
sys.path.insert(0,str(release));from pcm_eval.space_preparation_queue import PreparationQueue
from pcm_eval.model_ticket import EXAMPLE_INPUTS
queue=PreparationQueue(stage/'smoke-queue');smoke=queue.ensure('rmm-'+uuid.uuid4().hex[:20],next(iter(catalog['delivery_by_owner'])),EXAMPLE_INPUTS,profile,revision='engineering-protocol-check')
receipt={'state':'PREPARED','tests':tests,'release':str(release),'code_sha256':SIGNATURE,'new_inference_frames':0,
         'catalog_sha256':hashlib.sha256((release/'configs/model_ticket_catalog.json').read_bytes()).hexdigest(),
         'smoke_preparation_id':smoke['id'],'queue_isolated':True}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    print(call(code,timeout=200))


def deploy():
    import deploy_model_tickets as base
    base.RELEASE=RELEASE;base.STAGE=STAGE
    # The existing deployment checks use a file digest manifest. Add one only
    # after the isolated preparation protocol has passed with the actual mount.
    code='RELEASE='+repr(RELEASE)+'\nSTAGE='+repr(STAGE)+'\n'+r'''
import pathlib,json,hashlib
release=pathlib.Path(RELEASE);stage=pathlib.Path(STAGE)
receipt=json.loads((stage/'preparation.json').read_text());assert receipt['state']=='PREPARED'
proof=json.loads((stage/'smoke-verification.json').read_text());assert proof['state']=='DATA_READY' and proof['new_inference_frames']==0 and proof['job_id']==receipt['smoke_preparation_id']
receipt['files']={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['pcm_eval','tools','tests'] for p in (release/folder).glob('*.py')}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2));print('deployment_digest_manifest_ready')
'''
    print(call(code));base.deploy()


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else deploy()
