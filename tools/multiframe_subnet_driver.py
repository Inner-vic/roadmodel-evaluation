"""Bounded multi-frame transport; unchanged compiled networks and metric collector."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import time

import fast_subnet_driver as base
from batch_subnet_driver import Engine as SerialEngine, unpack_result
from board_multiframe_subnet import public_ports


def pack_batch(engine, roots, board):
    packets=[];parts=[]
    for slot,root in enumerate(roots):
        target=board+'-f'+str(slot);body,_=engine.input_packet(root,target)
        packets.append({'root':target,'packet':target+'.request.tar','sha256':hashlib.sha256(body).hexdigest()})
        parts.append(body)
    manifest={'schema':'roadmodel.multiframe-subnets/v1','root':board,'cache':engine.cache,'frames':packets}
    raw=json.dumps(manifest).encode();buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:') as archive:
        from board_batch_subnet import add_bytes
        add_bytes(archive,'manifest.json',raw)
        for slot,part in enumerate(parts):add_bytes(archive,f'frame_{slot}.tar',part)
    blob=buffer.getvalue()
    if len(blob)>128*1024**2:raise ValueError('Batch exceeds verified 128MiB packet bound')
    return blob,manifest,hashlib.sha256(raw).hexdigest()


def unpack_batch(blob, result, expected_manifest, packet_sha, roots, public):
    if result['batch_packet_sha256']!=packet_sha or result['frames']!=len(roots) or result['invocations']!=5:
        raise ValueError('Batch reply identity differs')
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        expected={'result.json'}|{f'frame_{i}.tar' for i in range(len(roots))}
        if set(names)!=expected or len(names)!=len(expected) or any(not m.isfile() for m in members):
            raise ValueError('Unsafe batch result inventory')
        value=json.loads(archive.extractfile('result.json').read())
        manifest_sha=hashlib.sha256(json.dumps(expected_manifest).encode()).hexdigest()
        if (value['manifest_sha256']!=manifest_sha or value['state']!='SUCCEEDED'
                or value['frames']!=len(roots) or value['invocations']!=5
                or len(value['bundles'])!=len(roots)):
            raise ValueError('Batch result manifest differs')
        rows=[]
        for slot,(root,row,frame) in enumerate(zip(roots,value['bundles'],expected_manifest['frames'])):
            body=archive.extractfile(f'frame_{slot}.tar').read()
            if (row['slot']!=slot or row['path']!=frame['root']+'/result.tar' or len(body)!=row['nbytes']
                    or hashlib.sha256(body).hexdigest()!=row['sha256']):
                raise ValueError('Frame result identity differs')
            data=unpack_result(body,root,public,frame['sha256'])
            if (data['frame_slot']!=slot or any(p['frame_slot']!=slot for p in data['outputs'].values())
                    or any(not p['remote_file'].startswith(frame['root']+'/') for p in data['outputs'].values())):
                raise ValueError('Cross-frame result mapping rejected')
            rows.append(data)
    return value,rows


class Engine(SerialEngine):
    def __init__(self):
        # ReferencePrefetch's verified workers own the reference sessions.
        # The board dispatcher only needs contracts, weights/cache bindings and comparisons.
        super().__init__(reference_session=False)

    def run_many(self, roots, run_id, *, validation=False, validate_collector=False):
        roots=list(map(Path,roots));count=len(roots)
        if count not in (1,2,4,8) or len(set(roots))!=count:raise ValueError('Invalid native batch')
        if (base.FULL/'STOP_AFTER_FRAME').exists() and not validation:raise ValueError('Full evaluation paused')
        for root in roots:
            if (root/'execution.json').exists() or (root/'execution_receipt.json').exists():
                raise ValueError('Existing or unknown hardware execution; never replay')
            if json.loads((root/'prepare.json').read_text())['state']!='READY':raise ValueError('Reference not ready')
        board='/userdata/benchEval/models/'+run_id
        from board_validate_subnet import checked_root
        checked_root(board)
        q=json.loads(base.old.subprocess.run([str(base.old.RUNNER/'bin/bench-eval'),'queue'],
                capture_output=True,text=True,check=True,timeout=20).stdout)
        if q.get('active') or q.get('queued') or q.get('waiting') or (base.old.RUNNER/'locks/bench.lock').exists():
            raise ValueError('Shared bench occupied; no native dispatch')
        sys.path.insert(0,str(base.old.RUNNER/'current/automation'));import runner_state
        leases=base.old.BASE/'external-bench-leases';lease=leases/run_id;lease.mkdir(parents=True,exist_ok=False)
        runner_state.atomic_write_json(lease/'manifest.json',{'submitted_by':'developer','purpose':'authorized bounded RoadModel multiframe evaluation'})
        runner_state.atomic_write_json(lease/'state.json',{'run_id':run_id,'state':'READY'})
        runner_state.atomic_write_json(lease/'plan.json',{'preflight':{'status':'READY'},'steps':[],'activation':False,
            'replay':False,'frame_count':count,'request_source':'explicit Codex user request'})
        claimed=runner_state.claim_bench(leases,base.old.RUNNER/'locks',run_id)
        client=None;safe=False;records=[];times={};started=time.monotonic()
        try:
            for slot,root in enumerate(roots):
                record={'schema':'roadmodel.five-subnet-execution/v1','state':'RUNNING','run_id':run_id+'-f'+str(slot),
                    'frame_id':json.loads((root/'input-reuse.json').read_text())['frame']['frame_id'],'version':'v3.2.2',
                    'requested_frames':1,'stages':[],'guard':claimed,'original_active_app_modified':False,
                    'multiframe':{'frames':count,'slot':slot,'parent_run_id':run_id},
                    'code_sha256':{'driver':base.sha(Path(__file__)),'helper':base.sha(Path(__file__).with_name('board_multiframe_subnet.py'))}}
                records.append(record);base.save(root/'execution.json',record)
            client=self.connect()
            if client._exec('pgrep -x InferenceExec',check=False)[1].strip():raise ValueError('Residual native process')
            self.ensure_cache(client)
            names=('board_multiframe_subnet.py','board_batch_subnet.py','board_validate_subnet.py')
            signature=hashlib.sha256(b''.join(Path(__file__).with_name(n).read_bytes() for n in names)).hexdigest()
            helper_dir=self.cache+'/multiframe-'+signature[:20]
            if getattr(self,'_multi_helper',None)!=helper_dir:
                if not client._sftp_exists(helper_dir):
                    client._exec('mkdir '+shlex.quote(helper_dir),check=True)
                    for name in names:client._sftp_write_bytes(helper_dir+'/'+name,Path(__file__).with_name(name).read_bytes())
                for name in names:
                    if client._exec('sha256sum '+shlex.quote(helper_dir+'/'+name),check=True)[1].split()[0]!=base.sha(Path(__file__).with_name(name)):
                        raise ValueError('Private helper differs')
                self._multi_helper=helper_dir
            tick=time.monotonic();blob,manifest,manifest_sha=pack_batch(self,roots,board)
            packet_sha=hashlib.sha256(blob).hexdigest();packet=board+'.bundle.tar'
            if client._sftp_exists(packet) or client._sftp_exists(board):raise ValueError('Unknown board attempt')
            client._sftp.putfo(io.BytesIO(blob),packet,file_size=len(blob),confirm=True)
            times['packet_build_and_upload_seconds']=time.monotonic()-tick
            tick=time.monotonic()
            _,text,_=client._exec(shlex.join(['python3',helper_dir+'/board_multiframe_subnet.py','bundle',packet,packet_sha]),timeout=480,check=True)
            result=json.loads(text)
            if result['bundle_path']!=board+'/batch-result.tar' or not 0<result['nbytes']<count*8*1024**2:
                raise ValueError('Wrong native result bundle')
            times['board_command_seconds']=time.monotonic()-tick
            tick=time.monotonic();blob=base.stream_read(client,result['bundle_path'],result['nbytes'],result['sha256'])
            public=public_ports(self.graph,self.public)
            board_result,frames=unpack_batch(blob,result,manifest,packet_sha,roots,public)
            diagnostic_seconds=0
            for root,record,data in zip(roots,records,frames):
                record['stages']=data['stages'];base.save(root/'all-subnet-outputs.json',data['outputs'])
                self.compare(root,record,data['outputs'])
                if validate_collector:
                    before=time.monotonic();self.reference_public_check(client,root,record,data['outputs'],public)
                    diagnostic_seconds+=time.monotonic()-before
                record.update(state='SUCCEEDED',public_outputs=18,tensor_only=True)
            times.update(download_and_collection_seconds=time.monotonic()-tick,
                         diagnostic_seconds=diagnostic_seconds,board_stages=board_result['stages'])
            if client._exec('pgrep -x InferenceExec',check=False)[1].strip():
                raise ValueError('Residual native process after batch; retain lease')
            safe=True
            try:
                _,cleanup,_=client._exec(shlex.join(['python3',helper_dir+'/board_multiframe_subnet.py','cleanup',board+'.json',manifest_sha]),timeout=60,check=True)
                if json.loads(cleanup)['state']!='CLEANED':raise ValueError('Cleanup not confirmed')
                times['private_board_artifacts_removed']=True
            except Exception as error:times['cleanup_issue']=str(error)[:200]
        except Exception as error:
            for record in records:record.update(state='NEEDS_REVIEW',error=repr(error))
            raise
        finally:
            if not safe and client is not None:
                try:safe=not client._exec('pgrep -x InferenceExec',check=False)[1].strip()
                except Exception:safe=False
            if safe:
                try:
                    release=runner_state.release_bench(leases,base.old.RUNNER/'locks',run_id,'NO_ACTIVATION',None)
                    for record in records:record['release']=release
                except Exception as error:
                    for record in records:record.update(state='NEEDS_REVIEW',lock_retained=True,release_issue=str(error)[:300])
            else:
                for record in records:record.update(state='NEEDS_REVIEW',lock_retained=True)
            for root,record in zip(roots,records):base.save(root/'execution.json',record)
        times['hardware_phase_seconds']=time.monotonic()-started
        if any(r['state']!='SUCCEEDED' or r.get('release',{}).get('state')!='RELEASED' for r in records):
            raise ValueError('Native result or lease release needs review; never repeat inference')
        for root,record in zip(roots,records):
            base.old.write_compatible_results(root,record)
            integrity=json.loads((root/'output_integrity.json').read_text())
            integrity['evidence']=f'Five original executable invocations, infer_num={count}, 49 checked outputs per frame'
            base.save(root/'output_integrity.json',integrity)
            receipt=json.loads((root/'execution_receipt.json').read_text());receipt['multiframe']=record['multiframe']
            base.save(root/'execution_receipt.json',receipt)
        return times
