"""Bounded two-frame experiment. Never starts or commits the full controller."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import formal_compile_remote as controller
import fast_subnet_driver as driver
from batch_subnet_driver import Engine, unpack_result
from board_multiframe_subnet import public_ports
from pcm_eval.gt_runtime import initialize_native_gt, validated_settings

OUT = HERE.parent/'validation'
FULL = controller.FULL
save = driver.save
sha = driver.sha


def main():
    reservation = json.loads((OUT/'reservation.json').read_text())
    recovering = (OUT/'recovery-request.json').exists()
    prepared_only = (OUT/'admission-only-request.json').exists()
    indices = reservation['indices']
    assert len(indices) == 2
    assert sha(FULL/'compile/checkpoint.npz') == reservation['checkpoint_sha256']
    assert (FULL/'STOP_AFTER_FRAME').exists() and not (controller.c.ROOT/'controller.lock').exists()
    marker = OUT/('admission-only-started.json' if prepared_only else 'recovery-started.json' if recovering else 'started.json')
    assert not marker.exists(), 'Do not repeat a prior experiment'
    save(marker, {'pid':os.getpid(),'frames':2,'started_at':controller.c.now()})
    engine = None
    client = None
    claimed = None
    safe = False
    records = []
    roots = []
    times = {}
    start = time.monotonic()
    run_id = reservation.get('run_id','pcm-v322-subnets-full-twoframe-20261109-01')
    lease_id = run_id+('-recover-stage0' if recovering else '')
    board = '/userdata/benchEval/models/'+run_id
    sys.path.insert(0, str(driver.old.RUNNER/'current/automation'))
    import runner_state
    leases = driver.old.BASE/'external-bench-leases'
    try:
        engine = Engine()
        public = public_ports(engine.graph, engine.public)
        times['engine_initialization_seconds'] = time.monotonic()-start
        tick = time.monotonic()
        if prepared_only:
            for index in indices:
                root=controller.c.ROOT/'frames'/f'frame_{index:06d}'
                assert not (root/'execution.json').exists() and not (root/'execution_receipt.json').exists()
                prep=json.loads((root/'prepare.json').read_text());assert prep['state']=='READY'
                roots.append(root)
            times['retained_prepare_including_ONNX_seconds']=sum(json.loads((r/'prepare.json').read_text())['preparation_seconds'] for r in roots)
        elif recovering:
            for index in indices:
                root=controller.c.ROOT/'frames'/f'frame_{index:06d}'
                previous=json.loads((root/'execution.json').read_text())
                assert previous['state']=='NEEDS_REVIEW' and previous['release']['state']=='RELEASED'
                assert not (root/'execution_receipt.json').exists()
                save(OUT/('before-recovery-'+str(index)+'.json'),previous)
                roots.append(root)
        else:
            exporter = controller.make_pipeline(indices, OUT/'canonical')
            for index, man in zip(indices, exporter):
                assert [man.meta['scene_name'],man.meta['frame_idx']] == controller.FRAMES[index]
                root = controller.c.ROOT/'frames'/f'frame_{index:06d}'
                root.mkdir(exist_ok=False)
                engine.prepare(SimpleNamespace(cache_frame=str(man.frame_dir),task_root=str(OUT)),root)
                roots.append(root)
        assert len(roots) == 2
        times['data_and_ONNX_seconds'] = time.monotonic()-tick
        waiting_since=time.monotonic()
        sys.path.insert(0,'/mnt/data/bench-eval/service/gateway')
        from bench_gateway.config import load
        from bench_gateway.core import Store
        while True:
            q=json.loads(driver.old.subprocess.run([str(driver.old.RUNNER/'bin/bench-eval'),'queue'],
                        capture_output=True,text=True,check=True,timeout=20).stdout)
            with Store(load('/mnt/data/bench-eval/service/gateway')['state_root']).connect() as db:
                busy=bool(db.execute("select id from jobs where state in ('QUEUED','RUNNING')").fetchone())
            if not (q.get('active') or q.get('queued') or q.get('waiting') or (driver.old.RUNNER/'locks/bench.lock').exists() or busy):break
            save(OUT/'progress.json',{'state':'WAITING_BENCH','native_dispatched':False,'full_evaluation_resumed':False,
                'wait_seconds':time.monotonic()-waiting_since})
            if time.monotonic()-waiting_since>900:raise ValueError('Shared bench still busy after 15 minutes; no dispatch')
            time.sleep(3)
        times['bench_wait_seconds']=time.monotonic()-waiting_since
        assert (FULL/'STOP_AFTER_FRAME').exists() and sha(FULL/'compile/checkpoint.npz') == reservation['checkpoint_sha256']
        lease = leases/lease_id
        lease.mkdir(parents=True,exist_ok=False)
        runner_state.atomic_write_json(lease/'manifest.json',{'submitted_by':'developer','purpose':'authorized two-frame multiframe experiment'})
        runner_state.atomic_write_json(lease/'state.json',{'run_id':lease_id,'state':'READY'})
        runner_state.atomic_write_json(lease/'plan.json',{'preflight':{'status':'READY'},'steps':[],
            'activation':False,'replay':False,'frame_count':2,'request_source':'explicit Codex user request'})
        claimed = runner_state.claim_bench(leases,driver.old.RUNNER/'locks',lease_id)
        for slot, root in enumerate(roots):
            record={'schema':'roadmodel.five-subnet-execution/v1','state':'RUNNING','run_id':run_id+'-f'+str(slot),
                'version':'v3.2.2','requested_frames':1,'frame_id':json.loads((root/'input-reuse.json').read_text())['frame']['frame_id'],
                'stages':[],'guard':claimed,'original_active_app_modified':False,'multiframe':{'frames':2,'slot':slot,'parent_run_id':run_id},
                'code_sha256':{'driver':sha(Path(__file__)),'board_helper':sha(HERE/'board_multiframe_subnet.py')}}
            assert recovering or not (root/'execution.json').exists()
            records.append(record);save(root/'execution.json',record)
        client=engine.connect()
        assert not client._exec('pgrep -x InferenceExec',check=False)[1].strip()
        engine.ensure_cache(client)
        helper_names=('board_multiframe_subnet.py','board_batch_subnet.py','board_validate_subnet.py')
        signature=hashlib.sha256(b''.join((HERE/n).read_bytes() for n in helper_names)).hexdigest()
        helper_dir=engine.cache+'/twoframe-'+signature[:20]
        if not client._sftp_exists(helper_dir):
            client._exec('mkdir '+shlex.quote(helper_dir),check=True)
            for name in helper_names:client._sftp_write_bytes(helper_dir+'/'+name,(HERE/name).read_bytes())
        for name in helper_names:
            assert client._exec('sha256sum '+shlex.quote(helper_dir+'/'+name),check=True)[1].split()[0]==sha(HERE/name)
        packets=[]
        tick=time.monotonic()
        for slot, root in enumerate(roots):
            target=board+'-f'+str(slot);packet_path=target+'.request.tar'
            body,_=engine.input_packet(root,target)
            if not recovering:
                assert not client._sftp_exists(target) and not client._sftp_exists(packet_path)
                client._sftp.putfo(io.BytesIO(body),packet_path,file_size=len(body),confirm=True)
            packets.append({'root':target,'packet':packet_path,'sha256':hashlib.sha256(body).hexdigest()})
        manifest={'schema':'roadmodel.two-frame-subnets/v1','root':board,'cache':engine.cache,'frames':packets}
        manifest_path=board+'.json'
        if recovering:
            old=json.loads(client._sftp_read_bytes(manifest_path))
            assert old==manifest
            names=('stdout.log','stderr.log','model_info.json')
            evidence={n:hashlib.sha256(client._sftp_read_bytes(board+'/sub0/'+n)).hexdigest() for n in names}
            # Approximate historical stage time from retained file timestamps; excluded from clean benchmarks.
            elapsed=client._sftp.stat(board+'/sub0/stdout.log').st_mtime-client._sftp.stat(board+'/sub0/model_info.json').st_mtime
            manifest['recover_completed_stage0']={'stage0_files':evidence,'stage0_invocation_seconds':max(0,elapsed)}
            manifest_path=board+'.resume-1.json'
        raw=json.dumps(manifest).encode();manifest_sha=hashlib.sha256(raw).hexdigest()
        assert (recovering or not client._sftp_exists(board)) and not client._sftp_exists(manifest_path)
        client._sftp_write_bytes(manifest_path,raw)
        times['upload_seconds']=time.monotonic()-tick
        tick=time.monotonic()
        _, reply, _=client._exec(shlex.join(['python3',helper_dir+'/board_multiframe_subnet.py',manifest_path,manifest_sha]),timeout=480,check=True)
        result=json.loads(reply)
        assert result['state']=='SUCCEEDED' and result['manifest_sha256']==manifest_sha and result['invocations']==5 and result['frames']==2
        save(OUT/'board-result.json',result)
        times['remote_command_seconds']=time.monotonic()-tick
        tick=time.monotonic()
        diagnostic_seconds=0.0
        for slot, (root, record, bundle) in enumerate(zip(roots,records,result['bundles'])):
            assert bundle['slot']==slot and bundle['path']==board+'-f'+str(slot)+'/result.tar'
            assert 0<bundle['nbytes']<8*1024**2
            blob=driver.stream_read(client,bundle['path'],bundle['nbytes'],bundle['sha256'])
            frame_result=unpack_result(blob,root,public,packets[slot]['sha256'])
            assert frame_result['frame_slot']==slot
            assert all(x['frame_slot']==slot for x in frame_result['outputs'].values())
            record['stages']=frame_result['stages']
            save(root/'all-subnet-outputs.json',frame_result['outputs'])
            engine.compare(root,record,frame_result['outputs'])
            diagnostic_started=time.monotonic()
            engine.reference_public_check(client,root,record,frame_result['outputs'],public)
            diagnostic_seconds+=time.monotonic()-diagnostic_started
            record.update(state='SUCCEEDED',public_outputs=18,tensor_only=True)
        times['collect_and_compare_seconds']=time.monotonic()-tick
        times['diagnostic_legacy_collection_seconds']=diagnostic_seconds
        safe=True
    except Exception as error:
        for root, record in zip(roots,records):
            record.update(state='NEEDS_REVIEW',error=repr(error));save(root/'execution.json',record)
        save(OUT/'progress.json',{'state':'NEEDS_REVIEW','error':repr(error),'full_evaluation_resumed':False})
        raise
    finally:
        if claimed is not None:
            if not safe and client is not None:
                try:safe=not client._exec('pgrep -x InferenceExec',check=False)[1].strip()
                except Exception:safe=False
            if safe:
                release=runner_state.release_bench(leases,driver.old.RUNNER/'locks',lease_id,'NO_ACTIVATION',None)
                for root,record in zip(roots,records):
                    record['release']=release;save(root/'execution.json',record)
            else:
                for root,record in zip(roots,records):
                    record.update(state='NEEDS_REVIEW',lock_retained=True);save(root/'execution.json',record)
        if engine is not None:engine.close()
    # Original receipts, tensor metric collector and GT oracle on these same new outputs.
    tick=time.monotonic()
    settings=validated_settings(HERE/'gt-runtime.json')
    initialize_native_gt(1,settings['backend']);os.environ['ROADMODEL_GT_BACKEND']=settings['backend']
    for index,root,record in zip(indices,roots,records):
        assert record['state']=='SUCCEEDED' and record['release']['state']=='RELEASED'
        driver.old.write_compatible_results(root,record)
        integrity=json.loads((root/'output_integrity.json').read_text())
        integrity['evidence']='Five original InferenceExec invocations, each with infer_num=2; 49 outputs verified for this frame'
        save(root/'output_integrity.json',integrity)
        receipt=json.loads((root/'execution_receipt.json').read_text());receipt['multiframe']=record['multiframe']
        save(root/'execution_receipt.json',receipt)
        controller.ensure_gt(index,root,validate=True)
    times['GT_seconds']=time.monotonic()-tick
    times['end_to_end_seconds']=time.monotonic()-start
    assert sha(FULL/'compile/checkpoint.npz')==reservation['checkpoint_sha256'] and (FULL/'STOP_AFTER_FRAME').exists()
    value={'state':'VERIFIED_TWO_FRAME_PROTOCOL','indices':indices,'frames':[controller.FRAMES[i] for i in indices],
        'actual_NPU_chains':2,'InferenceExec_invocations':5,'ONNX_inferences':2,'per_frame_checked_outputs':49,
        'per_frame_public_outputs':18,'legacy_public_collector_exact':True,'original_GT_oracle_verified':True,
        'serial_NPU_same_frame_repeated':False,'full_checkpoint_unchanged':True,'full_evaluation_resumed':False,
        'times':times,'board_stages':result['stages'],'same_frame_serial_equivalence':'NOT_TESTED_NO_REPEAT',
        'timing_scope':'RECOVERY_NOT_CLEAN' if recovering else 'PREPARED_INPUTS_TWO_FRAME_RUN' if prepared_only else 'CLEAN_TWO_FRAME_RUN',
        'board_private_artifacts_retained_for_diagnosis':True}
    save(OUT/'verification.json',value);save(OUT/'progress.json',value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
