"""Two new cached frames using an isolated derivative of the verified five-subnet driver."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import digest, load_json, save_json
from pcm_eval.planning import ssh_command
from prepare_subnet_remote import remote

OUT = ROOT / 'evidence/coverage-supplement-20261029'
ADAPTER = '/mnt/data/developer/pcm-roadmodel-eval/coverage-adapter-20261029'
PRIOR = "BASE/'subnet-runs/pcm-v322-subnets-one-02'"


def main():
    source = (ROOT / 'tools/subnet_chain_remote.py').read_text(encoding='utf-8')
    replacements = [
        ("    sample=BASE/'runs/pcm-v7-multiscene-001/frame_000000'", "    sample=Path(args.cache_frame)\n    assert sample.resolve().parent==(BASE/'runs/pcm-v7-multiscene-001').resolve()\n    assert sample.name in ('frame_000002','frame_000016')"),
        ("    if frame['frame_id']!='2026-06-30_18-39-31_HLX34B167T1300224_18':raise ValueError('Wrong cached sample')", "    assert frame['meta']['temporal_enabled']==0.0"),
        ("    subprocess.run([sys.executable,str(PACKAGE/'convert_fp32_to_fp16_keep_fp8.py'),str(MATERIAL/'v322.onnx'),str(onnx_path)],check=True)", "    old="+PRIOR+"\n    oldprep=json.loads((old/'prepare.json').read_text())\n    assert sha(old/'reference_fp16.onnx')==oldprep['onnx_fp16_sha256']\n    shutil.copyfile(old/'reference_fp16.onnx',onnx_path)"),
        ("    payload_bundle(stages,root/'payload.tar.gz')", "    assert sha(old/'payload.tar.gz')==oldprep['payload_sha256']\n    shutil.copyfile(old/'payload.tar.gz',root/'payload.tar.gz')"),
        ("'frame_id':'2026-06-30_18-39-31_HLX34B167T1300224_18'", "'frame_id':json.loads((root/'input-reuse.json').read_text())['frame']['frame_id']"),
        ("p.add_argument('--request-json')", "p.add_argument('--request-json');p.add_argument('--cache-frame',required=True)"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    code = OUT / 'code'
    code.mkdir(exist_ok=False)
    driver = code / 'subnet_chain_remote.py'
    driver.write_text(source, encoding='utf-8')
    save_json(code / 'derivation.json', {'source': str(ROOT / 'tools/subnet_chain_remote.py'), 'source_sha256': digest(ROOT / 'tools/subnet_chain_remote.py'),
                                        'derived_sha256': digest(driver), 'changes': [x[0] for x in replacements], 'new_frames': 2,
                                        'original_runtime_driver_unchanged': True, 'reused_fp16_and_payload_from': 'pcm-v322-subnets-one-02'})
    remote('import pathlib;pathlib.Path('+repr(ADAPTER)+').mkdir(parents=True,exist_ok=True)')
    for path, filename in [(driver, 'subnet_chain_remote.py'), (ROOT / 'pcm_eval/subnets.py', 'subnet_contract.py')]:
        p = subprocess.run(['scp.exe', '-q', '-o', 'BatchMode=yes', str(path), 'ubuntu-workstation:'+ADAPTER+'/'+filename], capture_output=True, timeout=45)
        if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-1000:])
        assert remote('import pathlib,hashlib;print(hashlib.sha256(pathlib.Path('+repr(ADAPTER+'/'+filename)+').read_bytes()).hexdigest())').strip() == digest(path)
    for i, row in enumerate(load_json(OUT / 'compile-selected.json')):
        run_id = f'pcm-v322-subnets-cover-{i+1:02d}'
        argv = ['/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python', '-u', ADAPTER+'/subnet_chain_remote.py', '--run-id', run_id, '--cache-frame', row['cache'], '--board']
        with (OUT / (run_id+'.console.log')).open('xb') as log:
            p = subprocess.run(ssh_command('ubuntu-workstation', argv), stdout=log, stderr=subprocess.STDOUT, timeout=900)
        save_json(OUT / (run_id+'.dispatch.json'), {'run_id': run_id, 'argv': argv, 'exit_code': p.returncode, 'selected': row, 'driver_sha256': digest(driver)})
        print(json.dumps({'run_id': run_id, 'exit_code': p.returncode, 'frames': 1}), flush=True)
        if p.returncode: raise RuntimeError('Review retained execution; no automatic replay')


if __name__ == '__main__':
    main()
