"""Stage the LPAI-native execution and task-owned space lifecycle release."""
import io
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from connect_active_entry import call
from pcm_eval.common import load_json, save_json

OLD = '/mnt/data/bench-eval/service/roadmodel/evaluation-20261105-r1'
NEW = '/mnt/data/bench-eval/service/roadmodel/evaluation-native-20261105-r2'
STAGE = '/mnt/data/bench-eval/roadmodel/deployment/evaluation-native-20261105-r2'
LOCAL = ROOT/'evidence/evaluation-native-20261105'


def remote(code, payload=b'', timeout=120):
    return json.loads(call('OLD='+repr(OLD)+'\nNEW='+repr(NEW)+'\nSTAGE='+repr(STAGE)+'\n'+code, payload, timeout=timeout))


def prepare():
    source_root = ROOT/'evidence/model-evaluation/model-reference-20261027'
    binding, launch = load_json(source_root/'model-input-binding.json'), load_json(source_root/'native-launch.json')
    old_space = load_json(source_root/'bmad-r3-a100-20261015.json')
    template = {'namespace': old_space['metadata']['namespace'], 'cluster': old_space['spec']['clusterName'],
        'project': 'C000002-底盘运动控制软件开发', 'resource_queue': 'ns-lpai-demo',
        'image': old_space['spec']['image'], 'resources': old_space['spec']['resources'],
        'accesses': {'ssh_credential': 'bmad-roadmodel-20261009', 'visibility': 'personal', 'ports': [22, 8888]},
        'idle_shutdown': {'enabled': True, 'minutes': 240}}
    runtime_root = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
    native = {'version': binding['model_version'], 'heads': ['occ', 'vision'],
        'source_root': runtime_root+'/source', 'source_manifest': runtime_root+'/source-manifest.json',
        'source_commit': binding['code']['evaluation_commit'], 'py_config': binding['code']['config'],
        'binding': {k: binding[k] for k in ('code', 'gt_contract', 'main_checkpoint', 'classifier_checkpoint')},
        'environment': launch['environment'], 'batch_size': 100, 'dataset_class': 'SmartChassisDatasetV7', 'gpu_index': 0}
    for field in ('main_checkpoint', 'classifier_checkpoint'):
        native[field] = {'path': runtime_root+'/weights/'+binding[field]['filename'], 'sha256': binding[field]['sha256']}
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT/'pcm_eval').glob('*.py')]
    files += ['tools/evaluation_worker.py', 'tools/evaluation_agent.py', 'tools/delivery_demo_agent.py',
              'tools/delivery_demo_plugin.py', 'tools/model_space_desktop.py',
              'tests/test_evaluation.py', 'tests/test_lpai_native_model.py', 'tests/test_delivery_combined.py',
              'tests/test_delivery_workflow.py', 'tests/test_dataset_mount.py', 'tests/test_dataset_preparation.py',
              'tools/formal_compile_remote.py', 'tools/fast_compile_controller.py', 'tools/fast_subnet_driver.py',
              'tools/batch_subnet_driver.py', 'tools/board_batch_subnet.py', 'tools/board_validate_subnet.py',
              'tools/evaluation_space_actor.py',
              'schemas/space-preparation-result.schema.json']
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in files:
            archive.add(ROOT/name, arcname=name)
        payload = json.dumps({'template': template, 'native': native}).encode()
        member = tarfile.TarInfo('configs/native-bootstrap.json'); member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    result = remote(r'''import pathlib,sys,io,tarfile,shutil,ast,json,subprocess,hashlib,re
old=pathlib.Path(OLD);new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Release already deployed'
stage.mkdir(parents=True,exist_ok=True)
if not new.exists():shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for member in archive:
  assert member.isfile();path=(new/member.name).resolve();path.relative_to(new);raw=archive.extractfile(member).read()
  if path.suffix=='.py':ast.parse(raw)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
cfg=json.loads((old/'configs/evaluation.json').read_text());bootstrap=json.loads((new/'configs/native-bootstrap.json').read_text())
cfg['profile']['environment_template']=bootstrap['template']
cfg.update(execution_scope='validation',validation_selection={'scene_count':1,'frames_per_scene':2},bench_available=False)
materials=json.loads((new/'tests/fixtures/current-delivery-materials.json').read_text())
sys.path.insert(0,str(new))
from pcm_eval.delivery_workflow import material_identity
cfg['native_profile']={'materials':material_identity(materials),'execution':bootstrap['native']}
dataset='/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1'
cfg['compile_profile']={'materials':material_identity(materials),'supported_index_sha256':['e86137f3fced2f3c4635d66cfc59904de36fba065ee2dd3e637f52850d59119c'],
 'dataset_root':dataset,'index_path':dataset+'/data_link_26-10-20-1_train859_eval135_v7.json',
 'native_task_code':'/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01/code',
 'native_release':'/mnt/data/bench-eval/service/roadmodel/compile-batch-20261030-r2','python':sys.executable}
(new/'configs/evaluation.json').write_text(json.dumps(cfg,indent=2))
plugin=new/'tools/delivery_demo_plugin.py';text=plugin.read_text();prior='/mnt/data/bench-eval/service/roadmodel/delivery-demo-20261103-r1'
assert text.count(prior)==1;plugin.write_text(text.replace(prior,NEW))
tests=[]
for module in ('tests.test_lpai_native_model','tests.test_evaluation','tests.test_delivery_workflow','tests.test_delivery_combined','tests.test_delivery_demo','tests.test_publication','tests.test_strict_model_template'):
 p=subprocess.run([sys.executable,'-m','unittest',module,'-q'],cwd=new,capture_output=True,text=True,timeout=60)
 output=p.stdout+p.stderr;(stage/(module+'.log')).write_text(output);assert p.returncode==0,output[-2500:]
 tests.append({'module':module,'count':int(re.search(r'Ran (\d+) tests?',output)[1])})
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_dataset*.py','-q'],cwd=new,capture_output=True,text=True,timeout=60)
output=p.stdout+p.stderr;(stage/'dataset-tests.log').write_text(output);assert p.returncode==0,output[-2500:]
tests.append({'module':'test_dataset*.py','count':int(re.search(r'Ran (\d+) tests?',output)[1])})
receipt={'state':'PREPARED','release':NEW,'tests':tests,'test_count':sum(x['count'] for x in tests),'bench_contacted':False,
 'files':{str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('pcm_eval','tools','configs','tests') for p in (new/folder).glob('*') if p.is_file()}}
(stage/'preparation.json').write_text(json.dumps(receipt,indent=2))
agent={'release':NEW,'state_root':'/mnt/data/bench-eval/state/roadmodel/evaluation','owner':cfg['owner'],'agent_id':cfg['agent_id'],
 'python':sys.executable,'profile':cfg['profile'],'native_profile':cfg['native_profile'],'compile_profile':cfg['compile_profile'],
 'rpc_timeout':110,'queue_kind':'evaluation'}
print(json.dumps({'receipt':{k:v for k,v in receipt.items() if k!='files'},'agent':agent}))
''', stream.getvalue(), timeout=180)
    LOCAL.mkdir(parents=True, exist_ok=True)
    save_json(LOCAL/'preparation.json', result['receipt'], overwrite=(LOCAL/'preparation.json').exists())
    agent = {**result['agent'], 'local_state': str(ROOT/'.runtime/evaluation-native-agent'),
             'lpai_cli': 'C:/Users/developer/AppData/Local/bin/lpai-agent.exe'}
    save_json(LOCAL/'agent-config.json', agent, overwrite=(LOCAL/'agent-config.json').exists())
    print(json.dumps(result['receipt']))


def deploy():
    result = remote(r'''import pathlib,json,hashlib,sqlite3,shutil,os,time
new=pathlib.Path(NEW);stage=pathlib.Path(STAGE)
assert not (stage/'deployment.json').exists(),'Release already deployed'
plugin=pathlib.Path('/mnt/data/bench-eval/service/gateway/bench_gateway/roadmodel_plugin.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(plugin)=='b25aadd0120b1b496fee87a4bde15521d0f2fea1d3828fc2fbab5aeec6434db1','Inspect changed gateway before promotion'
prepared=json.loads((stage/'preparation.json').read_text())
assert all(sha(new/name)==h for name,h in prepared['files'].items()),'Candidate changed after tests'
verified=json.loads((stage/'native-live-state/runs/rm-249df1fab8e44d708e76/progress.json').read_text())['combined_report']
assert verified['state']=='VERIFIED' and verified['report_url']=='https://li.feishu.cn/docx/DEMO00000000000000000000005','Actual native validation and publication required'
offline=json.loads((stage/'compile-scope-offline-verification.json').read_text());assert offline['state']=='VERIFIED'
cfg=json.loads((new/'configs/evaluation.json').read_text());assert cfg['bench_available'] is False and cfg['execution_scope']=='validation'
state=pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation');database=state/'runtime.sqlite3'
with sqlite3.connect(database) as source:
 assert not source.execute("select 1 from tasks where state in ('QUEUED','PREPARING','READY','RUNNING','WAITING_REPORT')").fetchone(),'Active evaluation must drain'
 with sqlite3.connect(stage/'before-evaluation.sqlite3') as target:source.backup(target)
 # Supersede the old unsubmitted full-range draft. Existing active/completed
 # work is never changed or replayed by installation.
 source.execute("update tasks set state='NEEDS_REVIEW',nonce='',notice=?,sent_digest='',updated=? where state='DRAFT'",('旧未提交卡片已停用。请重新发送RoadModel评测，使用本次小场景验证入口。',time.time()))
shutil.copy2(plugin,stage/'before-plugin.py')
body=plugin.read_bytes();prior=("ROOT = Path('"+OLD+"')").encode()
assert body.count(prior)==1,'Current shared routing root differs'
# Preserve the concurrently delivered routing, error isolation, and shared
# hardware admission logic byte-for-byte. Change only the RoadModel release.
body=body.replace(prior,("ROOT = Path('"+NEW+"')").encode());compile(body,str(plugin),'exec')
(stage/'merged-plugin.py').write_bytes(body)
temp=plugin.with_suffix('.native.tmp');temp.write_bytes(body);os.replace(temp,plugin)
receipt={'state':'DEPLOYED','release':NEW,'plugin_sha256':sha(plugin),'shared_bot_restarted':False,
 'single_existing_bot_connection':True,'execution_scope':'validation','maximum_frames_per_scene':2,
 'bench_available':False,'real_feishu_callback_verified':False,'background_cua_verified':False,
 'desktop_agent_pause_preserved':True,'new_inference_frames':0}
(stage/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
''')
    save_json(LOCAL/'deployment.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',nargs='?',choices=('prepare','deploy'),default='prepare')
    args=parser.parse_args()
    prepare() if args.action=='prepare' else deploy()
