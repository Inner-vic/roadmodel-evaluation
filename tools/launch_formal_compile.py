"""Deploy isolated full worker and start the durable formal workstation job."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import digest, load_json, save_json
from prepare_subnet_remote import remote
from pcm_eval.planning import ssh_command

OUT = ROOT / 'evidence/formal-v322-20261029-01'
REMOTE = '/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'


def main():
    src = (ROOT / 'tools/subnet_chain_remote.py').read_text()
    replacements = [
        ("    sample=BASE/'runs/pcm-v7-multiscene-001/frame_000000'", "    sample=Path(args.cache_frame)\n    assert sample.resolve().is_relative_to(Path(args.task_root).resolve())"),
        ("    if frame['frame_id']!='2026-06-30_18-39-31_HLX34B167T1300224_18':raise ValueError('Wrong cached sample')", "    assert frame['meta']['temporal_enabled']==0.0"),
        ("    subprocess.run([sys.executable,str(PACKAGE/'convert_fp32_to_fp16_keep_fp8.py'),str(MATERIAL/'v322.onnx'),str(onnx_path)],check=True)", "    old=BASE/'subnet-runs/pcm-v322-subnets-one-02'\n    original=json.loads((old/'prepare.json').read_text())\n    assert sha(old/'reference_fp16.onnx')==original['onnx_fp16_sha256']\n    onnx_path.hardlink_to(old/'reference_fp16.onnx')"),
        ("    payload_bundle(stages,root/'payload.tar.gz')", "    assert sha(old/'payload.tar.gz')==original['payload_sha256']\n    (root/'payload.tar.gz').hardlink_to(old/'payload.tar.gz')"),
        ("'frame_id':'2026-06-30_18-39-31_HLX34B167T1300224_18'", "'frame_id':json.loads((root/'input-reuse.json').read_text())['frame']['frame_id']"),
        ("    finally:\n        client.close()", "    finally:\n        if safe_to_release and record.get('state')=='SUCCEEDED':\n            assert board=='/userdata/benchEval/models/'+args.run_id and args.run_id.startswith('pcm-v322-subnets-full-')\n            client._exec('rm -rf -- '+shlex.quote(board),check=True)\n            record['private_board_directory_removed_after_output_verification']=True\n        client.close()"),
    ]
    for old, new in replacements:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    driver = code / 'formal_subnet_chain.py'
    driver.write_text(src, encoding='utf-8')
    compile(src, str(driver), 'exec')
    compile((ROOT / 'tools/formal_compile_remote.py').read_text(), 'formal_compile_remote.py', 'exec')
    save_json(code / 'driver-derivation.json', {'original_sha256': digest(ROOT / 'tools/subnet_chain_remote.py'), 'derived_sha256': digest(driver),
               'changes': [x[0] for x in replacements], 'runtime_small_driver_unchanged': True, 'task_frames': 8337})
    files = [(driver, 'code/formal_subnet_chain.py'), (ROOT / 'pcm_eval/subnets.py', 'code/subnet_contract.py'),
             (ROOT / 'pcm_eval/gt_adapter.py', 'code/gt_adapter.py'), (ROOT / 'tools/formal_compile_remote.py', 'code/formal_compile_remote.py'),
             (OUT / 'request.json', 'request.json')]
    remote('import pathlib;pathlib.Path('+repr(REMOTE+'/code')+').mkdir(parents=True,exist_ok=True)')
    for path, name in files:
        target = REMOTE + '/' + name
        p = subprocess.run(['scp.exe', '-q', '-o', 'BatchMode=yes', str(path), 'ubuntu-workstation:' + target], capture_output=True, timeout=60)
        assert not p.returncode, 'Task source upload failed'
        assert remote('import pathlib,hashlib;print(hashlib.sha256(pathlib.Path('+repr(target)+').read_bytes()).hexdigest())').strip() == digest(path)
    code = """import pathlib,subprocess,json,sys
b=pathlib.Path(%r);assert not (b/'compile/progress.json').exists(),'Never duplicate a formal worker'
f=(b/'compile-controller.log').open('xb');p=subprocess.Popen(['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python','-u',str(b/'code/formal_compile_remote.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'directory':str(b),'expected_frames':8337}))
""" % REMOTE
    dispatch = json.loads(remote(code))
    save_json(OUT / 'compile-dispatch.json', dispatch)
    print(json.dumps(dispatch), flush=True)


if __name__ == '__main__': main()
