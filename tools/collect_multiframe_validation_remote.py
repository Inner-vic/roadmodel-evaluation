"""Recover collection/GT of already completed multi-frame inference; no board dispatch."""
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import formal_compile_remote as controller
import fast_subnet_driver as driver
from batch_subnet_driver import Engine,unpack_result
from board_multiframe_subnet import public_ports
from pcm_eval.gt_runtime import initialize_native_gt,validated_settings


def main():
    out=HERE.parent/'validation';save=driver.save
    assert not (out/'collection-recovery-started.json').exists()
    save(out/'collection-recovery-started.json',{'pid':os.getpid(),'new_NPU_inferences':0,'new_ONNX_inferences':0})
    reservation=json.loads((out/'reservation.json').read_text());board=json.loads((out/'board-result.json').read_text())
    assert board['state']=='SUCCEEDED' and board['frames']==2 and board['invocations']==5
    assert driver.sha(controller.FULL/'compile/checkpoint.npz')==reservation['checkpoint_sha256']
    assert (controller.FULL/'STOP_AFTER_FRAME').exists()
    engine=Engine();roots=[];records=[];start=time.monotonic()
    try:
        public=public_ports(engine.graph,engine.public);client=engine.connect()
        for slot,index in enumerate(reservation['indices']):
            root=controller.c.ROOT/'frames'/f'frame_{index:06d}';roots.append(root)
            record=json.loads((root/'execution.json').read_text())
            assert record['state']=='NEEDS_REVIEW' and record['release']['state']=='RELEASED'
            assert not (root/'execution_receipt.json').exists()
            save(out/('before-collection-'+str(index)+'.json'),record)
            remote='/userdata/benchEval/models/'+record['run_id']
            # The immutable old packet is the identity of each completed frame.
            packet=client._sftp_read_bytes(remote+'.request.tar')
            expected=__import__('hashlib').sha256(packet).hexdigest()
            bundle=board['bundles'][slot]
            assert bundle['slot']==slot and bundle['path']==remote+'/result.tar' and 0<bundle['nbytes']<8*1024**2
            blob=driver.stream_read(client,bundle['path'],bundle['nbytes'],bundle['sha256'])
            result=unpack_result(blob,root,public,expected)
            assert result['frame_slot']==slot and all(x['frame_slot']==slot for x in result['outputs'].values())
            save(root/'all-subnet-outputs.json',result['outputs'])
            engine.compare(root,record,result['outputs'])
            engine.reference_public_check(client,root,record,result['outputs'],public)
            record.update(state='SUCCEEDED',stages=result['stages'],public_outputs=18,tensor_only=True,
                          collection_recovered_without_inference=True)
            save(root/'execution.json',record);records.append(record)
    finally:engine.close()
    settings=validated_settings(HERE/'gt-runtime.json')
    initialize_native_gt(1,settings['backend']);os.environ['ROADMODEL_GT_BACKEND']=settings['backend']
    for index,root,record in zip(reservation['indices'],roots,records):
        driver.old.write_compatible_results(root,record)
        integrity=json.loads((root/'output_integrity.json').read_text())
        integrity['evidence']='Five original InferenceExec invocations, each with infer_num=2; 49 outputs verified for this frame'
        save(root/'output_integrity.json',integrity)
        receipt=json.loads((root/'execution_receipt.json').read_text());receipt['multiframe']=record['multiframe']
        save(root/'execution_receipt.json',receipt)
        controller.ensure_gt(index,root,validate=True)
    assert driver.sha(controller.FULL/'compile/checkpoint.npz')==reservation['checkpoint_sha256']
    value={'state':'VERIFIED_TWO_FRAME_PROTOCOL','frames':reservation['frames'],'indices':reservation['indices'],
        'actual_NPU_chains':2,'InferenceExec_invocations_total':5,'stage0_repeated':False,
        'per_frame_checked_outputs':49,'per_frame_public_outputs':18,'legacy_public_collector_exact':True,
        'original_GT_oracle_verified':True,'full_checkpoint_unchanged':True,'full_evaluation_resumed':False,
        'timing_scope':'RECOVERY_NOT_CLEAN','same_frame_serial_equivalence':'NOT_TESTED_NO_REPEAT',
        'collection_and_GT_recovery_seconds':time.monotonic()-start,'board_stages':board['stages']}
    save(out/'verification.json',value);save(out/'progress.json',value);print(json.dumps(value))


if __name__=='__main__':main()
