"""Run the supplied reference ONNX/packup on real V7 frames under the existing bench lock."""
import argparse
import fcntl
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path('/home/lixiang/developer/pcm-roadmodel-eval')
PACKAGE = ROOT/'package/benchEval-skill-20261020'
DATASET = ROOT/'datasets/smartchassisdata-v3/26-10-20-1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--onnx-only',action='store_true')
    parser.add_argument('--max-frames', type=int, default=3)
    parser.add_argument('--include-gt-metrics', action='store_true')
    args=parser.parse_args()
    if not 1 <= args.max_frames <= 3:
        raise ValueError('The verified smoke manifest contains only three frames')
    if not all(c.isalnum() or c in '-_' for c in args.run_id):
        raise ValueError('Invalid run ID')
    run=ROOT/'runs'/args.run_id
    if run.exists(): raise ValueError('Choose a new run ID')
    record_dir=ROOT/'run-evidence'/args.run_id
    record_dir.mkdir(parents=True,exist_ok=False)
    receipt={'run_id':args.run_id,'source':'user-supplied-v3.2.3-reference','dataset_version':'26-10-20-1',
             'requested_frames':args.max_frames,'state':'preparing','run_dir':str(run),
             'metric_scope':'tensor_alignment_and_gt' if args.include_gt_metrics else 'tensor_alignment'}
    def save():
        (record_dir/'execution.json').write_text(json.dumps(receipt,indent=2))
        if run.is_dir():
            (run/'execution_receipt.json').write_text(json.dumps(receipt,indent=2))
    save()
    lock=open('/home/lixiang/bench-eval/locks/bench.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        receipt.update(state='bench_busy');save();return 73
    log=(record_dir/'console.log').open('wb')
    def execute(argv):
        process=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.write(process.stdout);log.flush()
        if process.returncode:
            sys.stdout.buffer.write(process.stdout[-6000:]);sys.stdout.flush()
            raise RuntimeError('Command failed: '+argv[0]+' exit='+str(process.returncode))
        return process.stdout.decode(errors='replace')
    try:
        models=PACKAGE/'examples/assets'
        onnx=next(models.glob('*fp8.onnx'))
        pack=next(models.glob('*.tar.gz'))
        receipt['onnx_sha256']=hashlib.sha256(onnx.read_bytes()).hexdigest()
        receipt['packup_sha256']=hashlib.sha256(pack.read_bytes()).hexdigest()
        argv=[str(ROOT/'venv/bin/python'),'-u','run_bench_eval.py',
              '--eval-dataset-root',str(DATASET),'--dataset-path',str(DATASET),
              '--data-link-file',str(DATASET/'smoke_data_link.json'),'--dataset','SmartChassisDatasetV7',
              '--onnx',str(onnx),'--max-frames',str(args.max_frames),'--eval-subsample','100','--no-shm',
              '--skip-git-check','--skip-mount','--keep-frame-inputs','--save-dir',str(run),
              '--shared-input-dir',str(ROOT/'shared-inputs'/args.run_id)]
        if not args.include_gt_metrics:
            argv.append('--skip-metrics')
        if args.onnx_only:
            argv+=['--infer-backend','onnx']
        else:
            with tarfile.open(pack) as archive:
                for member in archive.getmembers():
                    p=pathlib.PurePosixPath(member.name)
                    if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
                        raise ValueError('Unsafe packup archive member')
            remote='/userdata/benchEval/models/'+args.run_id
            sys.path.insert(0, str(PACKAGE))
            from board_infer_client import BoardInferClient, BoardInferConfig
            # Reuse the supplied live client's keyboard-interactive authentication.
            # Native ssh may depend on an expired interactive master connection.
            with BoardInferClient(BoardInferConfig(packup_dir=remote)) as client:
                client._exec('test ! -e '+shlex.quote(remote)+' && mkdir '+shlex.quote(remote))
                client._sftp.put(str(pack), remote+'/packup.tar.gz')
                digest=client._exec('sha256sum '+shlex.quote(remote+'/packup.tar.gz'))[1].split()[0]
                if digest != receipt['packup_sha256']:
                    raise ValueError('Board packup transfer checksum mismatch')
                client._exec('cd '+shlex.quote(remote)+' && tar -xzf packup.tar.gz')
                packup=client._exec('find '+shlex.quote(remote)+' -name model_info.json -type f')[1].strip().splitlines()
            if len(packup)!=1: raise ValueError('Could not locate a unique packup model_info.json')
            receipt['board_packup']=str(pathlib.PurePosixPath(packup[0]).parent)
            argv+=['--board-packup',receipt['board_packup'],'--board-ssh','ssh root@192.0.2.10']
        receipt.update(state='running',argv=argv);save()
        process=subprocess.run(argv,cwd=str(PACKAGE),stdout=log,stderr=subprocess.STDOUT)
        if process.returncode == 0 and not args.include_gt_metrics and not args.onnx_only:
            execute([str(ROOT/'venv/bin/python'), str(ROOT/'tools/finalize_tensor_alignment.py'),
                     '--run-dir', str(run)])
            execute([str(ROOT/'venv/bin/python'), str(ROOT/'tools/validate_compile_outputs.py'),
                     '--run-dir', str(run)])
        receipt.update(state='succeeded' if process.returncode==0 else 'failed',exit_code=process.returncode)
        save()
        if process.returncode:
            log.flush()
            print((record_dir/'console.log').read_text(errors='replace')[-9000:])
        print(json.dumps(receipt),flush=True)
        return process.returncode
    except Exception as error:
        receipt.update(state='failed',error_type=type(error).__name__,error=str(error));save()
        print(json.dumps(receipt),flush=True)
        return 2
    finally:
        log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__': raise SystemExit(main())
