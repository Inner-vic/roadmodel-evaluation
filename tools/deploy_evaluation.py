"""Install the full work-order entry without restarting the shared Bot."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT))
from connect_active_entry import call
from pcm_eval.delivery_source import LarkSource

OLD = '/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r3'
NEW = '/mnt/data/bench-eval/service/roadmodel/evaluation-20261105-r1'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/evaluation-20261105-r1'
EXPECTED_PLUGIN = '6fc850fe5bfe3741648e6fc43f85244b0b102bcdcf7622ef0bb48efafb1680d4'
LOCAL = ROOT/'evidence/evaluation-20261105'


def remote(code, payload=b'', timeout=120):
    return json.loads(call('OLD='+repr(OLD)+'\nNEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\nEXPECTED_PLUGIN='+repr(EXPECTED_PLUGIN)+'\n'+code,
                           payload=payload, timeout=timeout))


def prepare():
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT/'pcm_eval').glob('*.py')]
    files += ['tools/evaluation_worker.py', 'tools/delivery_demo_worker.py', 'tools/delivery_demo_plugin.py',
              'tests/test_evaluation.py', 'tests/test_delivery_combined.py', 'tests/test_delivery_demo.py',
              'tests/test_delivery_workflow.py', 'tests/test_strict_model_template.py', 'tools/evaluation_agent.py',
              'tools/delivery_demo_agent.py', 'tools/model_space_desktop.py',
              'tests/test_dataset_mount.py', 'tests/test_dataset_preparation.py']
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for name in files:
            tar.add(ROOT/name, arcname=name)
        materials = LarkSource().read('https://li.feishu.cn/record/PUBREF000000000000000000001')
        raw = json.dumps(materials, ensure_ascii=False).encode('utf-8')
        info = tarfile.TarInfo('tests/fixtures/current-delivery-materials.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    code = r'''import pathlib,sys,io,tarfile,shutil,ast,json,subprocess,uuid,hashlib,re
old=pathlib.Path(OLD);new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Release already deployed'
stage.mkdir(parents=True,exist_ok=True)
if not new.exists():shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for item in tar:
  assert item.isfile();path=(new/item.name).resolve();path.relative_to(new)
  raw=tar.extractfile(item).read()
  if path.suffix=='.py':ast.parse(raw)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
plugin=new/'tools/delivery_demo_plugin.py';text=plugin.read_text()
prior='/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r1'
assert text.count(prior)==1;plugin.write_text(text.replace(prior,NEW))
cfg=json.loads((old/'configs/delivery_demo.json').read_text())
ticket=json.loads((old/'configs/model_ticket_catalog.json').read_text())
profile=ticket['runtime_config']['space_preparation']['by_owner'][cfg['owner']]
cfg.update(agent_id=uuid.uuid4().hex,profile=profile,bench_available=False)
(new/'configs/evaluation.json').write_text(json.dumps(cfg,indent=2))
tests=[]
for module in ['tests.test_evaluation','tests.test_delivery_demo','tests.test_delivery_workflow','tests.test_delivery_combined','tests.test_strict_model_template','tests.test_publication']:
 args=[sys.executable,'-m','unittest',module,'-q']
 p=subprocess.run(args,cwd=new,capture_output=True,text=True,timeout=60)
 text=p.stdout+p.stderr;(stage/(module+'.log')).write_text(text)
 assert p.returncode==0,text[-2500:]
 tests.append({'module':module,'count':int(re.search(r'Ran (\d+) tests?',text)[1])})
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_dataset*.py','-q'],cwd=new,capture_output=True,text=True,timeout=60)
text=p.stdout+p.stderr;(stage/'dataset-tests.log').write_text(text)
assert p.returncode==0,text[-2500:]
tests.append({'module':'test_dataset*.py','count':int(re.search(r'Ran (\d+) tests?',text)[1])})
sys.path[:0]=[str(new),str(new/'tools')]
from pcm_eval.delivery_workflow import prepare_delivery
from pcm_eval.delivery_source import LarkSource
from pcm_eval.compile_preparation import prepare_compile
from pcm_eval.common import json_digest
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.evaluation_entry import render
from types import SimpleNamespace
materials=json.loads((new/'tests/fixtures/current-delivery-materials.json').read_text())
source=SimpleNamespace(read=lambda url:materials)
# Reuse the live source read for offline archive/interface validation.
workflow=prepare_delivery('https://li.feishu.cn/record/PUBREF000000000000000000001',
 'https://lpai.lixiang.com/lpaiweb/xspacex_vc-chassis/subApp/dataset/detail/a92b698afa80429a9678826f482788f9?version=26-10-20-1',source=source,run_id='rm-offline-preparation')
job=next(j for j in workflow['jobs'] if j['kind']=='compile_alignment')
prepared=prepare_compile(job,stage/'compile-materials'/json_digest(job.get('request'))[:20])
assert prepared['state']=='PREPARED',prepared['issues']
assert not prepared['bench_contacted'] and len(prepared['contract']['edges'])==32
preview=EvaluationRuntime(stage/'preview-state');task=preview.open(cfg['owner'],'feishu:oc_preview:'+cfg['owner'])
(stage/'card-preview.json').write_text(json.dumps(render(task),ensure_ascii=False,indent=2))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'state':'PREPARED','release':NEW,'test_count':sum(t['count'] for t in tests),'tests':tests,
 'compile_contract_edges':len(prepared['contract']['edges']),'compile_external_inputs':len(prepared['contract']['external_inputs']),
 'bench_contacted':False,'new_inference_frames':0,
 'files':{str(p.relative_to(new)):sha(p) for name in ('pcm_eval','tools','configs','tests') for p in (new/name).glob('*') if p.is_file()}}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2))
agent={'release':NEW,'state_root':'/mnt/data/bench-eval/state/roadmodel/evaluation','owner':cfg['owner'],
 'agent_id':cfg['agent_id'],'python':sys.executable,'profile':profile,'rpc_timeout':110,'queue_kind':'evaluation'}
print(json.dumps({'receipt':{k:v for k,v in receipt.items() if k!='files'},'agent':agent}))
'''
    result = remote(code, archive.getvalue(), timeout=180)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL/'preparation.json').write_text(json.dumps(result['receipt'], ensure_ascii=False, indent=2), encoding='utf-8')
    # This contains only registered paths/IDs, no LPAI or Feishu credentials.
    config = {**result['agent'], 'local_state': str(ROOT/'.runtime/evaluation-agent'),
              'lpai_cli': 'C:/Users/developer/AppData/Local/bin/lpai-agent.exe'}
    (LOCAL/'agent-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['receipt'], ensure_ascii=False))


def deploy():
    code = r'''import pathlib,json,hashlib,sqlite3,shutil,os
new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Already deployed'
plugin=pathlib.Path('/mnt/data/bench-eval/service/gateway/bench_gateway/roadmodel_plugin.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(plugin)==EXPECTED_PLUGIN,'Gateway changed; inspect before deploying'
prepared=json.loads((stage/'preparation.json').read_text())
assert all(sha(new/name)==h for name,h in prepared['files'].items())
state=pathlib.Path('/mnt/data/bench-eval/state/roadmodel')
for name,table,db in [('delivery-demo','tasks','runtime.sqlite3'),('model-tickets','tickets','tickets.sqlite3')]:
 with sqlite3.connect(state/name/db) as c:
  assert not c.execute('select 1 from '+table+" where state in ('QUEUED','READY','RUNNING','PREPARING','CHECKING')").fetchone(),'Shared work must drain first'
backup=stage/'before.py';shutil.copy2(plugin,backup)
body=(new/'tools/delivery_demo_plugin.py').read_bytes();compile(body,str(plugin),'exec')
temp=plugin.with_suffix('.evaluation.tmp');temp.write_bytes(body);os.replace(temp,plugin)
receipt={'state':'DEPLOYED','release':NEW,'plugin_sha256':sha(plugin),'backup':str(backup),
 'single_existing_bot_connection':True,'shared_bot_restarted':False,'new_inference_frames':0,
 'bench_available':False,'real_user_callback_verified':False}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result = remote(code)
    (LOCAL/'deployment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'deploy'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else deploy()
