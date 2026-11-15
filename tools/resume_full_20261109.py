"""Resume the sealed full scope after proving its old failure was pre-dispatch."""
import ast
import json
from pathlib import Path
from deploy_environment_flow import remote

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT/'tools/resume_compile_batch_full.py').read_text())
CODE = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets))
CODE = CODE.replace('compile-resume-20261102-r1', 'compile-resume-20261109-r1')
CODE = CODE.replace("assert p['state']=='STOPPED_AT_BOUNDARY' and p['completed_frames']==2151",
    "assert p['state']=='NEEDS_REVIEW' and p['completed_frames']==2151 and p['error_type']=='BoardInferError' and 'SSH connect failed' in p['error']")
CODE = CODE.replace("assert (full/'STOP_AFTER_FRAME').is_file()", "assert not (full/'STOP_AFTER_FRAME').exists()")
PREFLIGHT = '''# Confirm current card materials and the retained full request are identical.
workflow=load(pathlib.Path('/mnt/data/bench-eval/state/roadmodel/evaluation/runs/rm-6a1e7a96b8c84d3ba1ec/workflow.json'))
plan=next(j for j in workflow['jobs'] if j['kind']=='compile_alignment')['plan']
assert plan['index_sha256']==req['data_link_sha256']
expected=plan['request']['model'];actual=req['compile_request']['model']
assert actual['onnx_sha256']==expected['onnx_sha256']
assert {r['stage']:r['sha256'] for r in actual['subnet_archives']}=={r['stage']:r['sha256'] for r in expected['subnet_archives']}
queue=json.loads(subprocess.run(['/home/lixiang/bench-eval/bin/bench-eval','queue'],capture_output=True,text=True,check=True,timeout=20).stdout)
assert not queue.get('active') and not queue.get('queued') and not queue.get('waiting')
assert not pathlib.Path('/home/lixiang/bench-eval/locks/bench.lock').exists()
sys.path.insert(0,'/mnt/data/bench-eval/service/gateway')
from bench_gateway.config import load as load_gateway
from bench_gateway.core import Store
store=Store(load_gateway('/mnt/data/bench-eval/service/gateway')['state_root'])
with store.connect() as db:assert not db.execute("select id from jobs where state in ('QUEUED','RUNNING')").fetchone()
sys.path.insert(0,'/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
from board_infer_client import BoardInferClient,BoardInferConfig
client=BoardInferClient(BoardInferConfig(host='192.0.2.10',port=22,username='root',packup_dir='/userdata/benchEval/models'))
try:
 client.connect();assert not client._exec('pgrep -x InferenceExec',check=False)[1].strip()
 for suffix in ('','.request.tar'):assert not client._sftp_exists('/userdata/benchEval/models/pcm-v322-subnets-full-002151-optv1'+suffix)
finally:client.close()
failed=full/'compile/frames/frame_002151/execution.json'
record=load(failed);assert record['state']=='NEEDS_REVIEW' and not record['stages'] and 'SSH connect failed' in record['error']
assert not (failed.parent/'execution_receipt.json').exists()
assert shutil.disk_usage(full).free>64*1024**3
'''
CODE = CODE.replace('audit.mkdir();', PREFLIGHT+'\naudit.mkdir();')
ARCHIVE = '''# Preserve the prior pre-dispatch attempt and use a distinct native run ID.
shutil.copytree(release,audit/'controller',ignore=shutil.ignore_patterns('__pycache__'))
controller=audit/'controller/formal_compile_remote.py'
body=controller.read_text();old="run_id=f'pcm-v322-subnets-full-{index:06d}-optv1'";assert body.count(old)==1
body=body.replace(old,"run_id=f'pcm-v322-subnets-full-{index:06d}-optv1-resume20261109'")
controller.write_text(body)
(audit/'failed-before-dispatch.json').write_text(json.dumps(record,indent=2))
failed.replace(audit/'execution-before-dispatch.json')'''
CODE = CODE.replace("(full/'STOP_AFTER_FRAME').replace(audit/'STOP_AFTER_FRAME.before-resume')", ARCHIVE)
CODE = CODE.replace("str(release/'formal_compile_remote.py')", 'str(controller)').replace('cwd=release,', 'cwd=controller.parent,')
CODE = CODE.replace("(audit/'STOP_AFTER_FRAME.before-resume').replace(full/'STOP_AFTER_FRAME')",
                    "(audit/'execution-before-dispatch.json').replace(failed)")

if __name__ == '__main__':
    result = remote(CODE, timeout=80)
    (ROOT/'evidence/card-acceptance-full-20261109/full-resume-dispatch.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
