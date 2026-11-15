"""One-frame five-network adapter, reusing the supplied BoardInferClient and ORT.

Run on the workstation with its existing venv. It never changes the active APP,
never compiles ELF files, and never consumes archive golden tensors as predictions.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

BASE=Path('/mnt/data/developer/pcm-roadmodel-eval')
PACKAGE=BASE/'package/benchEval-skill-20261020'
MATERIAL=BASE/'model-materials/v3.2.2-subnets'
RUNNER=Path('/home/lixiang/bench-eval')
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from subnet_contract import chain_contract,stage_info


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def payload_bundle(stages,path):
    """Preserve compiled files byte for byte; omit all shipped sample data/output."""
    with tarfile.open(path,'w:gz') as dest:
        for stage in stages:
            with tarfile.open(stage['archive']) as source:
                for member in source:
                    p=PurePosixPath(member.name)
                    if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():raise ValueError('Unsafe archive member')
                    parts=[x for x in p.parts if x!='.']
                    if not parts or parts[0] in ('data','output','infer_output'):continue
                    if not member.isfile() and not member.isdir():raise ValueError('Unsupported archive entry')
                    item=deepcopy(member);item.name=f"sub{stage['stage']}/"+'/'.join(parts)
                    dest.addfile(item,source.extractfile(member) if member.isfile() else None)


class Chain:
    def __init__(self,client,contract,infos,root,local,record):
        self.client,self.contract,self.infos=client,contract,infos
        self.root,self.local,self.record=root,local,record

    def execute(self):
        import numpy as np
        c=self.client;outputs={}
        for stage in self.contract['stages']:
            index=stage['stage'];directory=self.root+f'/sub{index}'
            c._exec('mkdir -p '+shlex.quote(directory+'/output')+' '+shlex.quote(directory+'/export'),check=True)
            info=stage_info(self.infos[index],stage,self.root)
            c._sftp_write_bytes(directory+'/model_info.json',json.dumps(info).encode())
            started=time.monotonic()
            row={'stage':index,'state':'RUNNING','input_count':len(stage['inputs']),'output_count':len(stage['outputs'])}
            self.record['stages'].append(row);save(self.local/'execution.json',self.record)
            try:
                code,out,err=c._exec('cd '+shlex.quote(directory)+' && InferenceExec model_info.json',timeout=180,check=False)
                (self.local/f'sub{index}.stdout.log').write_text(out)
                (self.local/f'sub{index}.stderr.log').write_text(err)
                row['exit_code']=code
                if code:raise RuntimeError(f'Sub{index} InferenceExec failed, exit={code}')
                names=c._exec('find '+shlex.quote(directory)+' -maxdepth 3 -type f -name '+shlex.quote('*.bin'),check=True)[1].splitlines()
                row['outputs']=[]
                for port in stage['outputs']:
                    basenames={port['filename'],PurePosixPath(port['declared_data_path']).name}
                    basenames|={'Frame_0_'+name for name in list(basenames)}
                    candidates=[name for name in names if PurePosixPath(name).name in basenames and '/export/' not in name]
                    candidates=[p for p in candidates if c._sftp.stat(p).st_size==port['nbytes']]
                    if not candidates:raise ValueError(f"Sub{index}: missing fresh output {port['tensor']}")
                    blobs=[c._sftp_read_bytes(p) for p in candidates]
                    if len({hashlib.sha256(x).hexdigest() for x in blobs})!=1:raise ValueError('Ambiguous fresh subnet output')
                    blob=blobs[0];array=np.frombuffer(blob,dtype=port['dtype'])
                    finite=bool(np.isfinite(array).all())
                    if not finite:raise ValueError(f"Sub{index} non-finite output: {port['tensor']}")
                    canonical=directory+'/export/'+port['filename']
                    c._exec('cp '+shlex.quote(candidates[0])+' '+shlex.quote(canonical),check=True)
                    meta={**port,'stage':index,'remote_file':canonical,'sha256':hashlib.sha256(blob).hexdigest(),'finite':finite}
                    row['outputs'].append(meta);outputs[port['tensor']]=meta
                    # Keep intermediate predictions for reproducible boundary checks, never goldens.
                    target=self.local/f'sub{index}'/port['filename'];target.parent.mkdir(exist_ok=True);target.write_bytes(blob)
                row.update(state='SUCCEEDED',seconds=time.monotonic()-started)
            except Exception as error:
                row.update(state='FAILED',seconds=time.monotonic()-started,error=str(error));raise
            finally:save(self.local/'execution.json',self.record)
        return outputs


def prepare(args,root):
    import onnx
    source=json.loads((root/'source-audit.json').read_text())
    infos={};stages=[]
    for stage in source['current']['stages']:
        archive=MATERIAL/PurePosixPath(stage['archive'].replace('\\','/')).name
        if sha(archive)!=stage['sha256']:raise ValueError('Compiled archive changed')
        with tarfile.open(archive) as tar:info=json.load(tar.extractfile('./model_info.json'))
        index=stage['stage'];infos[index]=info;stages.append({'stage':index,'archive':str(archive)})
    graph=chain_contract(infos);save(root/'chain-contract.json',graph)
    if sha(MATERIAL/'v322.onnx')!=source['current']['onnx_sha256']:raise ValueError('Wrong source ONNX')
    onnx_path=root/'reference_fp16.onnx'
    subprocess.run([sys.executable,str(PACKAGE/'convert_fp32_to_fp16_keep_fp8.py'),str(MATERIAL/'v322.onnx'),str(onnx_path)],check=True)
    model=onnx.load(str(onnx_path));expected=[]
    for output in model.graph.output:
        tensor=output.name.replace('/','@');matches=[p for s in graph['stages'] for p in s['outputs'] if p['tensor']==tensor]
        if len(matches)!=1:raise ValueError('ONNX output has no unique subnet producer: '+output.name)
        expected.append({'onnx_name':output.name,**matches[0]})
    if len(expected)!=18:raise ValueError('Unexpected public ONNX output count')
    save(root/'public-outputs.json',expected)
    sample=BASE/'runs/pcm-v7-multiscene-001/frame_000000'
    frame=json.loads((sample/'frame_manifest.json').read_text())
    if frame['frame_id']!='2026-06-30_18-39-31_HLX34B167T1300224_18':raise ValueError('Wrong cached sample')
    inputs=root/'inputs';inputs.mkdir()
    files=[]
    import numpy as np
    for port in graph['external_inputs']:
        path=sample/port['external_filename']
        if path.stat().st_size!=port['nbytes']:raise ValueError('Cached input shape/type mismatch')
        if not np.isfinite(np.fromfile(path,dtype=port['dtype'])).all():raise ValueError('Cached non-finite input')
        # Hardlink the verified 11-port cache; do not reread or download raw dataset assets.
        (inputs/path.name).hardlink_to(path)
        files.append({'filename':path.name,'nbytes':path.stat().st_size,'sha256':sha(path)})
    save(root/'input-reuse.json',{'frame':frame,'inputs':files,'dataset_downloads':0})
    payload_bundle(stages,root/'payload.tar.gz')
    from onnxruntime_infer import run_inference
    run_inference(onnx_path,inputs,root/'onnx_outputs',provider='CPUExecutionProvider')
    for entry in json.loads((root/'onnx_outputs/manifest.json').read_text())['outputs']:
        array=np.load(root/'onnx_outputs'/entry['npy_file'])
        if not np.isfinite(array).all():raise ValueError('Non-finite reference output; stop before hardware')
    save(root/'prepare.json',{'state':'READY','version':'v3.2.2','frame_count':1,'onnx_source_sha256':sha(MATERIAL/'v322.onnx'),
        'onnx_fp16_sha256':sha(onnx_path),'payload_sha256':sha(root/'payload.tar.gz'),'public_outputs':18,
        'original_skill_reused':['BoardInferClient','convert_fp32_to_fp16_keep_fp8','onnxruntime_infer'],
        'elf_recompiled':False,'sample_goldens_deployed':False})
    return graph,infos


def write_compatible_results(root,record):
    """Map measured results into the existing single-report collector contract."""
    measured=json.loads((root/'tensor-comparison.json').read_text())
    prep=json.loads((root/'prepare.json').read_text())
    reused=json.loads((root/'input-reuse.json').read_text())
    source=json.loads((root/'source-audit.json').read_text())
    rows=[]
    for row in measured['outputs']:
        rows.append({'onnx_output_name':row['name'],'status':'ok','cosine':row['cosine'],
            'max_abs_diff':row['max_abs_error'],'mean_abs_diff':row['mean_abs_error'],
            'compare_dtype':'int32' if row['board_dtype']=='int32' else 'float32',
            'shape_onnx':row['shape'],'shape_board':row['shape'],'board_dtype':row['board_dtype'],
            'integer_exact_ratio':row['exact_ratio'] if row['board_dtype']=='int32' else None})
    mean=sum(x['cosine'] for x in rows)/len(rows)
    frame={'frame_id':measured['frame_id'],'mean_cosine':mean,'min_cosine':min(x['cosine'] for x in rows),'outputs':rows}
    save(root/'cosine_summary.json',{'num_frames':1,'mean_cosine_all':mean,'min_cosine_all':mean,
        'tensors':{r['onnx_output_name']:{'mean_cosine':r['cosine'],'min_cosine':r['cosine'],'num_frames':1} for r in rows},
        'frames':[frame]})
    directory=root/'frame_000000';directory.mkdir(exist_ok=True)
    save(directory/'cosine_board_onnx.json',frame)
    manifest=deepcopy(reused['frame']);manifest['frame_dir']=str(directory)
    save(directory/'frame_manifest.json',manifest)
    save(root/'state.json',{'total_frames':1,'summary':{'done':1,'failed':0,'pending':0},
        'config':{'skip_metrics':True,'dataset':'SmartChassisDatasetV7','version':'v3.2.2','subnet_chain':True,
                  'onnx':str(root/'reference_fp16.onnx'),'board_packup':'/userdata/benchEval/models/'+record['run_id']},
        'frames':{'0':{'index':0,'status':'done','meta':manifest['meta']}}})
    save(root/'input_validation.json',{'state':'validated','frame_ids':[measured['frame_id']],
        'external_inputs':reused['inputs'],'source':'reused verified canonical V7 one-frame cache','dataset_downloads':0,
        'finite_outputs':{'state':'passed','checks':[{'backend':b,'frame_id':measured['frame_id'],'name':r['name'],'finite':True}
            for b in ('board','onnx') for r in measured['outputs']]}})
    save(root/'output_integrity.json',{'state':'fresh_outputs_checked','frame_count':1,'onnx_public_outputs':18,
        'all_subnet_outputs':49,'finite':True,'sample_golden_files_deployed':False,
        'evidence':'execution.json records fresh isolated outputs and hashes for each of five successful InferenceExec calls'})
    execution={'schema':'pcm.live-compile-receipt/v1','state':'succeeded','run_id':record['run_id'],
        'version_name':'v3.2.2','requested_frames':1,'selected_frames':[[manifest['meta']['scene_name'],manifest['meta']['frame_idx']]],
        'onnx_sha256':prep['onnx_fp16_sha256'],'source_onnx_sha256':prep['onnx_source_sha256'],
        'reference_onnx_path':str(root/'reference_fp16.onnx'),
        'packup_sha256':prep['payload_sha256'],'packup_format':'five isolated original SubNetOps directories, no ELF merge',
        'subnet_archives':[{'stage':x['stage'],'sha256':x['sha256'],'name':PurePosixPath(x['archive'].replace('\\','/')).name} for x in source['current']['stages']],
        'original_active_app_modified':False,'guard':record['guard'],'release':record['release'],
        'gt_metrics_computed':False,'started_at':record['guard']['lock']['claimed_at'],
        'finished_at':datetime.fromtimestamp((root/'execution.json').stat().st_mtime,timezone.utc).isoformat(),
        'data_link_sha256':sha(BASE/'datasets/smartchassisdata-v3/26-10-20-1/data_link_26-10-20-1_train859_eval135_v7.json'),
        'runtime':{'python':sys.version.split()[0],'onnx_provider':'CPUExecutionProvider'},
        'code_sha256':record.get('code_sha256') or {'driver':sha(root/'executed-adapter.py'),'contract':sha(root/'executed-contract.py')}}
    save(root/'execution_receipt.json',execution)


def run_board(args,root,graph,infos):
    from board_infer_client import BoardInferClient,BoardInferConfig
    sys.path.insert(0,str(RUNNER/'current/automation'))
    import runner_state
    # A separate external-inference lease uses the original atomic-JSON lock API.
    # It performs no APP activation/replay and never records a forged approval.
    leases=BASE/'external-bench-leases';lease=leases/args.run_id
    lease.mkdir(parents=True,exist_ok=False)
    runner_state.atomic_write_json(lease/'manifest.json',{'submitted_by':'developer','purpose':'authorized RoadModel one-frame InferenceExec'})
    runner_state.atomic_write_json(lease/'state.json',{'run_id':args.run_id,'state':'READY'})
    runner_state.atomic_write_json(lease/'plan.json',{'preflight':{'status':'READY'},'steps':[],
        'activation':False,'replay':False,'request_source':'explicit Codex user request','frame_count':1})
    q=subprocess.run([str(RUNNER/'bin/bench-eval'),'queue'],capture_output=True,text=True,check=True,timeout=15)
    queue=json.loads(q.stdout)
    if queue.get('active') is not None or queue.get('queued') or queue.get('waiting'):raise ValueError('Original Runner has work; no hardware dispatch')
    claimed=runner_state.claim_bench(leases,RUNNER/'locks',args.run_id)
    record={'schema':'roadmodel.five-subnet-execution/v1','state':'RUNNING','run_id':args.run_id,'version':'v3.2.2',
        'requested_frames':1,'frame_id':'2026-06-30_18-39-31_HLX34B167T1300224_18','stages':[],
        'guard':claimed,'board_inputs_from_existing_cache':True,'original_active_app_modified':False}
    record['code_sha256']={'driver':sha(Path(__file__)),'contract':sha(Path(__file__).with_name('subnet_contract.py'))}
    save(root/'execution.json',record)
    board='/userdata/benchEval/models/'+args.run_id
    client=BoardInferClient(BoardInferConfig(host='192.0.2.10',port=22,username='root',packup_dir=board))
    safe_to_release=False
    try:
        client.connect()
        if client._exec('pgrep -x InferenceExec',check=False)[1].strip():raise ValueError('Existing board InferenceExec; refuse dispatch')
        client._exec('test ! -e '+shlex.quote(board)+' && mkdir -p '+shlex.quote(board+'/inputs'),check=True)
        client._sftp.put(str(root/'payload.tar.gz'),board+'/payload.tar.gz')
        if client._exec('sha256sum '+shlex.quote(board+'/payload.tar.gz'),check=True)[1].split()[0]!=sha(root/'payload.tar.gz'):
            raise ValueError('Board payload changed in transfer')
        client._exec('cd '+shlex.quote(board)+' && tar -xzf payload.tar.gz',check=True)
        for port in graph['external_inputs']:
            path=root/'inputs'/port['external_filename']
            expected=next(x['sha256'] for x in json.loads((root/'input-reuse.json').read_text())['inputs'] if x['filename']==path.name)
            if sha(path)!=expected:raise ValueError('Canonical input changed after ONNX inference')
            client._sftp.put(str(path),board+'/inputs/'+path.name)
            if client._exec('sha256sum '+shlex.quote(board+'/inputs/'+path.name),check=True)[1].split()[0]!=expected:
                raise ValueError('Board input transfer SHA256 mismatch')
        outputs=Chain(client,graph,infos,board,root,record).execute()
        safe_to_release=True
        expected=json.loads((root/'public-outputs.json').read_text());save(root/'all-subnet-outputs.json',outputs)
        import numpy as np
        ort=json.loads((root/'onnx_outputs/manifest.json').read_text());by={x['onnx_output_name']:x for x in ort['outputs']}
        comparisons=[]
        for port in expected:
            meta=outputs[port['tensor']]
            a=np.load(root/'onnx_outputs'/by[port['onnx_name']]['npy_file'])
            b=np.fromfile(root/f"sub{meta['stage']}"/port['filename'],dtype=port['dtype']).reshape(port['shape'])
            if a.shape!=b.shape:raise ValueError('Comparison shape mismatch')
            af=a.astype(np.float64).ravel();bf=b.astype(np.float64).ravel();den=np.linalg.norm(af)*np.linalg.norm(bf)
            cosine=float(np.dot(af,bf)/den) if den else 1.0 if np.array_equal(a,b) else 0.0
            comparisons.append({'name':port['onnx_name'],'shape':port['shape'],'board_dtype':port['dtype'],
                'onnx_dtype':str(a.dtype),'cosine':cosine,'max_abs_error':float(np.max(np.abs(af-bf))),
                'mean_abs_error':float(np.mean(np.abs(af-bf))),'exact_ratio':float(np.mean(a==b)),
                'board_finite':True,'onnx_finite':True,'board_sha256':meta['sha256']})
        save(root/'tensor-comparison.json',{'frame_id':record['frame_id'],'version':'v3.2.2',
            'frame_count':1,'outputs':comparisons,'tensor_count':len(comparisons),'mixed_score':None,'gt_metrics':'not_computed_in_tensor_smoke'})
        record.update(state='SUCCEEDED',public_outputs=len(comparisons),tensor_only=True)
    except Exception as error:
        record.update(state='FAILED',error=repr(error))
        try:safe_to_release=not client._exec('pgrep -x InferenceExec',timeout=10,check=False)[1].strip()
        except Exception:safe_to_release=False
        raise
    finally:
        client.close()
        if safe_to_release:
            record['release']=runner_state.release_bench(leases,RUNNER/'locks',args.run_id,'NO_ACTIVATION',None)
        else:record['state']='NEEDS_REVIEW';record['lock_retained']=True
        save(root/'execution.json',record)
    if record['state']=='SUCCEEDED':write_compatible_results(root,record)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--board',action='store_true');p.add_argument('--request-json')
    p.add_argument('--validate-request-only',action='store_true');args=p.parse_args()
    if args.request_json:
        request=json.loads(args.request_json)
        evaluation=request.get('evaluation',{})
        if (request.get('kind')!='compile_alignment' or request.get('version_name')!='v3.2.2'
                or type(evaluation.get('max_frames')) is not int or evaluation.get('max_frames')!=1
                or type(evaluation.get('expected_frames')) is not int or evaluation.get('expected_frames')!=1
                or evaluation.get('selection',{}).get('frames')!=[['2026-06-30_18-39-31_HLX34B167T1300224',18]]
                or evaluation.get('compute_gt_metrics') is not False):raise ValueError('Unverified request scope')
        args.run_id=request['run_id'];args.board=True
    import re
    if not isinstance(args.run_id,str) or not re.fullmatch(r'(?:pcm-v322-subnets-|rm-)[A-Za-z0-9_-]{1,48}',args.run_id):raise ValueError('Invalid isolated run ID')
    if args.validate_request_only:
        if not args.request_json:raise ValueError('Request JSON is required')
        print(json.dumps({'state':'REQUEST_VALIDATED','inference_frames':0,'run_id':args.run_id}));return
    root=BASE/'subnet-runs'/args.run_id
    if root.exists():raise ValueError('Preserve old run; no automatic replay')
    root.mkdir(parents=True)
    shutil.copyfile(__file__,root/'executed-adapter.py')
    shutil.copyfile(Path(__file__).with_name('subnet_contract.py'),root/'executed-contract.py')
    if args.request_json:save(root/'request.json',request)
    shutil.copyfile(MATERIAL/'source-audit.json',root/'source-audit.json')
    graph,infos=prepare(args,root)
    if args.board:run_board(args,root,graph,infos)
    print(json.dumps({'run_dir':str(root),'state':'SUCCEEDED' if args.board else 'PREPARED','frames':1}),flush=True)


if __name__=='__main__':main()
