"""Evaluate one not-yet-run final frame into its canonical full-run slot.

The full worker later adopts this frame's real receipt, so board/ONNX inference
is performed exactly once. All graph, board and GT work reuses the full adapter.
"""
import importlib.util
import os
from pathlib import Path
import sys
import textwrap

FULL=Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
VALIDATION=Path(os.environ['ROADMODEL_VALIDATION_ROOT'])
spec=importlib.util.spec_from_file_location('formal_compile_library',FULL/'code/formal_compile_remote.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def main():
    cursor=8336
    request=c.load(FULL/'request.json');expected=request['frames'];assert len(expected)==8337
    progress_path=VALIDATION/'compile/progress.json'
    def progress(state,**extra):
        c.save(progress_path,{'state':state,'completed_frames':1 if state=='SUCCEEDED' else 0,
              'expected_frames':1,'pid':os.getpid(),'updated_at':c.now(),**extra})
    progress('PREPARING')
    root=c.ROOT/'frames'/f'frame_{cursor:06d}'
    assert not root.exists(),'Never overwrite or repeat an existing frame attempt'
    live=c.load(c.ROOT/'progress.json');assert live['completed_frames']<=cursor
    link=c.DATA/'data_link_26-10-20-1_train859_eval135_v7.json'
    assert c.sha(link)==request['data_link_sha256']
    os.environ.update(DATASET_PATH=str(c.DATA),DATA_LINK_FILE=str(link),DATASET='SmartChassisDatasetV7')
    pipeline=c.NumpyBenchDataPipeline().setup(save_dir=str(VALIDATION/'canonical'),prefer_shm=False,max_frames=1,
                    eval_subsample=100,scene_sample=0,load_ground_truth=False)
    assert [[s,f['frame_index']] for s,f in pipeline._frames]==expected
    pipeline._frames=pipeline._frames[cursor:cursor+1]
    source_data=c.load(link);acc={b:c.Counts() for b in ('board','onnx')};oracle_checks=0
    (VALIDATION/'code').mkdir(parents=True,exist_ok=True)
    for name in ('formal_subnet_chain.py','subnet_contract.py'):
        target=VALIDATION/'code'/name
        if not target.exists():target.hardlink_to(FULL/'code'/name)
    source=(FULL/'code/formal_compile_remote.py').read_text()
    start=source.index("            root = ROOT / 'frames'")
    end=source.index('            cursor += 1',start)
    body=textwrap.dedent(source[start:end])
    for man in pipeline.iter_export():
        scene,fr=expected[cursor];assert man.meta['scene_name']==scene and man.meta['frame_idx']==fr
        item=next(f for f in source_data['eval'][scene]['frames'] if f['frame_index']==fr)
        occ=c.DATA/scene/'dataset'/str(fr)/item['occ_file']
        # execute the original, hash-pinned full worker's one-frame body unchanged.
        namespace={**c.__dict__,**locals(),'TASK':VALIDATION}
        exec(compile(body,str(FULL/'code/formal_compile_remote.py')+':one-frame','exec'),namespace)
        assert c.load(root/'execution_receipt.json')['state']=='succeeded'
        assert c.load(root/'gt_frame_receipt.json')['frame_id']==scene+'_'+str(fr)
        progress('SUCCEEDED',canonical_frame=str(root),selected_frames=[[scene,fr]])

if __name__=='__main__':
    try:main()
    except Exception as error:
        c.save(VALIDATION/'compile/progress.json',{'state':'NEEDS_REVIEW','completed_frames':0,'expected_frames':1,
               'pid':os.getpid(),'updated_at':c.now(),'error':str(error)[:500]})
        raise
