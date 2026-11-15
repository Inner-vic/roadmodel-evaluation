"""Resume board admission after a proven pre-dispatch queue rejection.

Reuse the completed ONNX output. Only retry admission while no hardware execution
record exists; once board execution is uncertain, stop and retain its evidence.
"""
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import sys
import tarfile
import textwrap
import time

FULL=Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
PROGRESS=Path(os.environ['ROADMODEL_VALIDATION_ROOT'])
spec=importlib.util.spec_from_file_location('formal_compile_library',FULL/'code/formal_compile_remote.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def main():
    cursor=8336;request=c.load(FULL/'request.json');scene,fr=request['frames'][cursor]
    root=c.ROOT/'frames'/f'frame_{cursor:06d}'
    assert c.load(root/'prepare.json')['state']=='READY'
    assert c.load(root/'input-reuse.json')['frame']['frame_id']==scene+'_'+str(fr)
    assert not (root/'execution.json').exists() and not (root/'execution_receipt.json').exists(),'Unknown or completed NPU execution must not replay'
    def progress(state,**extra):
        c.save(PROGRESS/'compile/progress.json',{'state':state,'completed_frames':1 if state=='SUCCEEDED' else 0,
              'expected_frames':1,'pid':os.getpid(),'updated_at':c.now(),**extra})
    progress('WAITING_BENCH')
    graph=c.load(root/'chain-contract.json');infos={}
    for row in c.load(root/'source-audit.json')['current']['stages']:
        archive=c.driver.MATERIAL/PurePosixPath(row['archive'].replace('\\','/')).name
        with tarfile.open(archive) as tar:infos[row['stage']]=json.load(tar.extractfile('./model_info.json'))
    admission=2
    while True:
        assert not (root/'execution.json').exists()
        q=json.loads(c.subprocess.run(['/home/lixiang/bench-eval/bin/bench-eval','queue'],capture_output=True,text=True,check=True,timeout=15).stdout)
        if q.get('active') is not None or q.get('queued') or q.get('waiting') or Path('/home/lixiang/bench-eval/locks/bench.lock').exists():
            progress('WAITING_BENCH');time.sleep(2);continue
        run_id=f'pcm-v322-subnets-full-{cursor:06d}-admit{admission:02d}'
        if (c.driver.BASE/'external-bench-leases'/run_id).exists():admission+=1;continue
        args=c.SimpleNamespace(run_id=run_id)
        try:
            progress('RUNNING')
            c.driver.run_board(args,root,graph,infos)
            break
        except ValueError as error:
            if str(error)=='Original Runner has work; no hardware dispatch' and not (root/'execution.json').exists():
                admission+=1;progress('WAITING_BENCH');time.sleep(2);continue
            raise
    # Reuse the original GT/postprocessing implementation after the NPU receipt.
    source=(FULL/'code/formal_compile_remote.py').read_text()
    start=source.index("            receipt = load(root / 'execution_receipt.json')")
    end=source.index('            cursor += 1',start)
    body=textwrap.dedent(source[start:end]);acc={b:c.Counts() for b in ('board','onnx')};oracle_checks=0
    data=c.load(c.DATA/'data_link_26-10-20-1_train859_eval135_v7.json')
    item=next(f for f in data['eval'][scene]['frames'] if f['frame_index']==fr)
    occ=c.DATA/scene/'dataset'/str(fr)/item['occ_file']
    namespace={**c.__dict__,**locals()}
    exec(compile(body,str(FULL/'code/formal_compile_remote.py')+':GT-only','exec'),namespace)
    progress('SUCCEEDED',canonical_frame=str(root),selected_frames=[[scene,fr]],onnx_inference_repeated=False)

if __name__=='__main__':
    try:main()
    except Exception as error:
        c.save(PROGRESS/'compile/progress.json',{'state':'NEEDS_REVIEW','completed_frames':0,'expected_frames':1,
               'pid':os.getpid(),'updated_at':c.now(),'error':str(error)[:500]})
        raise
