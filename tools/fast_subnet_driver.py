"""Five original subnet binaries, same ORT/metrics, optimized transport only.

Loaded in a NEW controller. Never patches the original/shared driver in place.
"""
import hashlib,importlib.util,inspect,json,os,shlex,sys,tarfile,time,textwrap
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import numpy as np

FULL=Path(os.environ.get('ROADMODEL_COMPILE_TASK_ROOT','/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'))
PACKAGE=Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
sys.path[:0]=[str(FULL/'code'),str(PACKAGE)]
spec=importlib.util.spec_from_file_location('retained_subnet_driver',FULL/'code/formal_subnet_chain.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from board_infer_client import BoardInferClient,BoardInferConfig
import onnxruntime_infer as ort_io
import onnxruntime as ort
from subnet_contract import chain_contract,stage_info
from concurrent.futures import ThreadPoolExecutor

sha=old.sha
save=old.save


def stream_read(client,path,nbytes,expected_hash):
    """Drain binary stdout before status; do not use text _exec for tensor data."""
    stdin,stdout,stderr=client._ssh.exec_command('cat -- '+shlex.quote(path),timeout=30)
    try:
        stdin.close();body=stdout.read(nbytes+1)
        if len(body)!=nbytes:raise ValueError('Stream output byte count differs')
        error=stderr.read(4097)
        if stdout.channel.recv_exit_status()!=0 or error:raise ValueError('Binary stream command failed')
        if hashlib.sha256(body).hexdigest()!=expected_hash:raise ValueError('Stream output SHA differs')
        return body
    finally:
        stdout.channel.close();stdout.close();stderr.close()


class Engine:
    def __init__(self, *, reference_session=True):
        started=time.monotonic();self.source=json.loads((old.MATERIAL/'source-audit.json').read_text());self.infos={}
        if os.environ.get('ROADMODEL_COMPILE_TASK_ROOT'):
            from pcm_eval.compile_scope import verify_native_materials
            binding_file=FULL/'native-materials.json'
            request=json.loads((FULL/'request.json').read_text())
            if sha(binding_file)!=request['native_materials_sha256']:raise ValueError('Native material binding changed')
            verify_native_materials(json.loads(binding_file.read_text()),self.source)
        for stage in self.source['current']['stages']:
            archive=old.MATERIAL/PurePosixPath(stage['archive'].replace('\\','/')).name
            if sha(archive)!=stage['sha256']:raise ValueError('Compiled archive differs')
            with tarfile.open(archive) as tar:self.infos[stage['stage']]=json.load(tar.extractfile('./model_info.json'))
        self.graph=chain_contract(self.infos);self.reference=old.BASE/'subnet-runs/pcm-v322-subnets-one-02'
        self.binding=json.loads((self.reference/'prepare.json').read_text())
        self.onnx=self.reference/'reference_fp16.onnx';self.payload=self.reference/'payload.tar.gz'
        assert sha(self.onnx)==self.binding['onnx_fp16_sha256'] and sha(self.payload)==self.binding['payload_sha256']
        assert sha(old.MATERIAL/'v322.onnx')==self.source['current']['onnx_sha256']
        import onnx
        model=onnx.load(str(self.onnx));self.public=[]
        for output in model.graph.output:
            matches=[p for s in self.graph['stages'] for p in s['outputs'] if p['tensor']==output.name.replace('/','@')]
            if len(matches)!=1:raise ValueError('Public output mapping differs')
            self.public.append({'onnx_name':output.name,**matches[0]})
        assert len(self.public)==18;self.public_names={x['tensor'] for x in self.public}
        self.session=None;self.input_spec=None;self.output_names=None
        if reference_session:
            so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;so.log_severity_level=3
            self.session=ort.InferenceSession(str(self.onnx),sess_options=so,providers=['CPUExecutionProvider'])
            self.input_spec,self.output_names=ort_io._session_io_spec(self.session)
        self.client=None;self.cache='/userdata/benchEval/models/pcm-v322-fast-cache-'+self.binding['payload_sha256'][:16]
        self.helper=Path(__file__).with_name('board_validate_subnet.py');self.helper_sha=sha(self.helper)
        self.cache_ready=False;self.initialization_seconds=time.monotonic()-started
        self.producers=ThreadPoolExecutor(max_workers=1,thread_name_prefix='frame-prepare')
    def prepare(self,args,root):
        if self.session is None:raise ValueError('Reference preparation requires its dedicated CPU session')
        started=time.monotonic();root=Path(root)
        if (root/'execution_receipt.json').exists():return self.graph,self.infos
        if (root/'execution.json').exists():raise ValueError('Unknown existing hardware execution; never replay')
        if (root/'prepare.json').exists():
            prep=json.loads((root/'prepare.json').read_text())
            if prep['state']!='READY' or prep['onnx_fp16_sha256']!=self.binding['onnx_fp16_sha256']:raise ValueError('Existing preparation differs')
            return self.graph,self.infos
        if list(root.iterdir()):raise ValueError('Partial preparation needs reconciliation')
        sample=Path(args.cache_frame)
        allowed=Path(getattr(args,'task_root',FULL)).resolve()
        approved={FULL.resolve()}
        if os.environ.get('ROADMODEL_OPT_VALIDATION'):approved.add(Path(os.environ['ROADMODEL_OPT_VALIDATION']).resolve())
        if allowed not in approved:raise ValueError('Unapproved input staging root')
        sample.resolve().relative_to(allowed)
        frame=json.loads((sample/'frame_manifest.json').read_text());assert frame['meta']['temporal_enabled']==0.0
        (root/'source-audit.json').write_text(json.dumps(self.source))
        (root/'executed-adapter.py').write_bytes(Path(__file__).read_bytes())
        (root/'executed-contract.py').write_bytes((FULL/'code/subnet_contract.py').read_bytes())
        (root/'reference_fp16.onnx').hardlink_to(self.onnx);(root/'payload.tar.gz').hardlink_to(self.payload)
        save(root/'chain-contract.json',self.graph);save(root/'public-outputs.json',self.public)
        inputs=root/'inputs';inputs.mkdir();files=[]
        for port in self.graph['external_inputs']:
            p=sample/port['external_filename'];assert p.stat().st_size==port['nbytes']
            assert np.isfinite(np.fromfile(p,dtype=port['dtype'])).all()
            (inputs/p.name).hardlink_to(p);files.append({'filename':p.name,'nbytes':p.stat().st_size,'sha256':sha(p)})
        save(root/'input-reuse.json',{'frame':frame,'inputs':files,'dataset_downloads':0})
        feed=ort_io._load_inputs(inputs,self.input_spec)
        values=dict(zip(self.output_names,self.session.run(self.output_names,feed)))
        if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('Non-finite ORT output')
        ort_io._save_outputs(root/'onnx_outputs',values,onnx_path=root/'reference_fp16.onnx',input_dir=inputs,provider='CPUExecutionProvider')
        save(root/'prepare.json',{'state':'READY','version':'v3.2.2','frame_count':1,
             'onnx_source_sha256':self.source['current']['onnx_sha256'],'onnx_fp16_sha256':self.binding['onnx_fp16_sha256'],
             'payload_sha256':self.binding['payload_sha256'],'public_outputs':18,'elf_recompiled':False,'sample_goldens_deployed':False,
             'original_skill_reused':['BoardInferClient','onnxruntime_infer IO helpers','same ORT_DISABLE_ALL CPU session'],
             'session_reused':True,'preparation_seconds':time.monotonic()-started})
        return self.graph,self.infos
    def connect(self):
        if self.client is None:
            self.client=BoardInferClient(BoardInferConfig(host='192.0.2.10',port=22,username='root',packup_dir=self.cache));self.client.connect()
        return self.client
    def ensure_cache(self,c):
        if self.cache_ready:return
        marker=self.cache+'/verified.json';identity={'payload_sha256':self.binding['payload_sha256'],'helper_sha256':self.helper_sha}
        if c._sftp_exists(marker):
            if json.loads(c._sftp_read_bytes(marker))!=identity:raise ValueError('Different private model cache')
        else:
            if c._sftp_exists(self.cache):raise ValueError('Incomplete private cache; preserve for review')
            c._exec('mkdir '+shlex.quote(self.cache),check=True)
            c._sftp.put(str(self.payload),self.cache+'/payload.tar.gz')
            assert c._exec('sha256sum '+shlex.quote(self.cache+'/payload.tar.gz'),check=True)[1].split()[0]==identity['payload_sha256']
            c._exec('cd '+shlex.quote(self.cache)+' && tar -xzf payload.tar.gz',check=True)
            c._sftp_write_bytes(self.cache+'/board_validate_subnet.py',self.helper.read_bytes())
            assert c._exec('sha256sum '+shlex.quote(self.cache+'/board_validate_subnet.py'),check=True)[1].split()[0]==self.helper_sha
            c._sftp_write_bytes(marker,json.dumps(identity).encode())
        self.cache_ready=True
    def run_board(self,args,root,graph=None,infos=None,*,reference_check=False,admission_after_receipt=False):
        root=Path(root)
        if (root/'execution_receipt.json').exists():raise ValueError('Do not repeat completed hardware execution')
        if (root/'execution.json').exists():raise ValueError('Unknown hardware result; never replay')
        sys.path.insert(0,str(old.RUNNER/'current/automation'));import runner_state
        # During the live validation, claim immediately after the original worker
        # released its frame, while it is still calculating GT. Avoid its q/check
        # -> prepare -> claim gap. All actual ownership remains the original lock.
        while True:
            q=json.loads(old.subprocess.run([str(old.RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,check=True,timeout=15).stdout)
            available=not q.get('active') and not q.get('queued') and not q.get('waiting') and not (old.RUNNER/'locks/bench.lock').exists()
            if available and admission_after_receipt:
                progress=json.loads((FULL/'compile/progress.json').read_text());p=FULL/'compile/frames'/f"frame_{progress['completed_frames']:06d}"/'execution_receipt.json'
                available=p.exists() and time.time()-p.stat().st_mtime<2
            if available:break
            time.sleep(.15 if admission_after_receipt else 1)
        leases=old.BASE/'external-bench-leases';lease=leases/args.run_id;lease.mkdir(parents=True,exist_ok=False)
        runner_state.atomic_write_json(lease/'manifest.json',{'submitted_by':'developer','purpose':'authorized RoadModel optimized evaluation'})
        runner_state.atomic_write_json(lease/'state.json',{'run_id':args.run_id,'state':'READY'})
        runner_state.atomic_write_json(lease/'plan.json',{'preflight':{'status':'READY'},'steps':[],'activation':False,'replay':False,'frame_count':1,'request_source':'explicit Codex user request'})
        claimed=runner_state.claim_bench(leases,old.RUNNER/'locks',args.run_id)
        record={'schema':'roadmodel.five-subnet-execution/v1','state':'RUNNING','run_id':args.run_id,'version':'v3.2.2','requested_frames':1,
                'frame_id':json.loads((root/'input-reuse.json').read_text())['frame']['frame_id'],'stages':[],'guard':claimed,
                'board_inputs_from_existing_cache':True,'original_active_app_modified':False,
                'code_sha256':{'driver':sha(Path(__file__)),'contract':sha(FULL/'code/subnet_contract.py'),'board_validator':self.helper_sha},
                'optimization':{'all49_checks_retained':True,'public_downloads':18,'shared_runner_unchanged':True}}
        save(root/'execution.json',record);board='/userdata/benchEval/models/'+args.run_id
        safe=False;c=None;started=time.monotonic();outputs={};downloaded=0
        try:
            c=self.connect()
            if c._exec('pgrep -x InferenceExec',check=False)[1].strip():raise ValueError('Existing InferenceExec; do not dispatch')
            cache_start=time.monotonic();self.ensure_cache(c);record['cache_setup_seconds']=time.monotonic()-cache_start
            command=['python3',self.cache+'/board_validate_subnet.py','initialize',board,self.cache]
            c._exec(shlex.join(command),check=True)
            for port in self.graph['external_inputs']:
                p=root/'inputs'/port['external_filename'];expected=next(x['sha256'] for x in json.loads((root/'input-reuse.json').read_text())['inputs'] if x['filename']==p.name)
                assert sha(p)==expected;c._sftp.put(str(p),board+'/inputs/'+p.name)
                assert c._exec('sha256sum '+shlex.quote(board+'/inputs/'+p.name),check=True)[1].split()[0]==expected
            for stage in self.graph['stages']:
                i=stage['stage'];directory=board+'/sub'+str(i);info=stage_info(self.infos[i],stage,board)
                c._sftp_write_bytes(directory+'/model_info.json',json.dumps(info).encode())
                c._sftp_write_bytes(directory+'/port-contract.json',json.dumps(stage).encode())
                row={'stage':i,'state':'RUNNING','input_count':len(stage['inputs']),'output_count':len(stage['outputs'])};record['stages'].append(row);save(root/'execution.json',record)
                tick=time.monotonic();code,out,err=c._exec('cd '+shlex.quote(directory)+' && InferenceExec model_info.json',timeout=180,check=False)
                (root/f'sub{i}.stdout.log').write_text(out);(root/f'sub{i}.stderr.log').write_text(err);row['exit_code']=code
                if code:raise ValueError('Original subnet failed: '+str(i))
                _,body,_=c._exec(shlex.join(['python3',self.cache+'/board_validate_subnet.py','validate',board,directory+'/port-contract.json']),timeout=30,check=True)
                checked=json.loads(body);assert len(checked)==len(stage['outputs']);row['outputs']=checked
                for meta in checked:
                    outputs[meta['tensor']]=meta
                    if meta['tensor'] in self.public_names:
                        blob=stream_read(c,meta['remote_file'],meta['nbytes'],meta['sha256']);assert np.isfinite(np.frombuffer(blob,dtype=meta['dtype'])).all()
                        dest=root/('sub'+str(i))/meta['filename'];dest.parent.mkdir(exist_ok=True);dest.write_bytes(blob);downloaded+=len(blob)
                row.update(state='SUCCEEDED',seconds=time.monotonic()-tick);save(root/'execution.json',record)
            safe=True;assert len(outputs)==49;save(root/'all-subnet-outputs.json',outputs)
            self.compare(root,record,outputs)
            record['optimized_board_seconds_before_diagnostic']=time.monotonic()-started
            record['public_tensor_bytes_downloaded']=downloaded
            if reference_check:self.reference_check(c,root,record,outputs)
            record.update(state='SUCCEEDED',public_outputs=18,tensor_only=True)
        except Exception as error:
            record.update(state='FAILED',error=repr(error))
            try:safe=c is not None and not c._exec('pgrep -x InferenceExec',timeout=10,check=False)[1].strip()
            except Exception:safe=False
            raise
        finally:
            if safe and record.get('state')=='SUCCEEDED':
                assert PurePosixPath(board).parent==PurePosixPath('/userdata/benchEval/models') and args.run_id.startswith('pcm-v322-subnets-full-')
                c._exec('rm -rf -- '+shlex.quote(board),check=True);record['private_board_directory_removed_after_output_verification']=True
            if safe:record['release']=runner_state.release_bench(leases,old.RUNNER/'locks',args.run_id,'NO_ACTIVATION',None)
            else:record.update(state='NEEDS_REVIEW',lock_retained=True)
            save(root/'execution.json',record)
        if record['state']=='SUCCEEDED':old.write_compatible_results(root,record)
    @staticmethod
    def compare(root,record,outputs):
        source=inspect.getsource(old.run_board);start=source.index('        expected=json.loads');end=source.index('        record.update(state=',start)
        exec(compile(textwrap.dedent(source[start:end]),'original-tensor-comparison','exec'),{**old.__dict__,'root':root,'record':record,'outputs':outputs})
    def reference_check(self,c,root,record,outputs):
        tick=time.monotonic();ref=root/'legacy-collector-validation';ref.mkdir();total=0
        for meta in outputs.values():
            blob=c._sftp_read_bytes(meta['remote_file']);assert len(blob)==meta['nbytes']
            assert hashlib.sha256(blob).hexdigest()==meta['sha256'] and np.isfinite(np.frombuffer(blob,dtype=meta['dtype'])).all()
            dest=ref/('sub'+str(meta['stage']))/meta['filename'];dest.parent.mkdir(exist_ok=True);dest.write_bytes(blob);total+=len(blob)
            if meta['tensor'] in self.public_names:assert blob==(root/('sub'+str(meta['stage']))/meta['filename']).read_bytes()
        for name in ('public-outputs.json','onnx_outputs'):(ref/name).symlink_to(root/name,target_is_directory=(root/name).is_dir())
        self.compare(ref,record,outputs)
        assert json.loads((root/'tensor-comparison.json').read_text())==json.loads((ref/'tensor-comparison.json').read_text())
        record['reference_validation']={'all49_bytes_hash_finite_match':True,'all18_tensor_metrics_exact_match':True,'legacy_collector_seconds':time.monotonic()-tick,'diagnostic_bytes':total,'native_inferences':1}
    def close(self):
        self.producers.shutdown(wait=True)
        if self.client:self.client.close()
