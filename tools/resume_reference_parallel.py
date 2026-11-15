"""Fix comparison packaging and reuse the four prepared baselines, no baseline rerun."""
import io
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_active_entry import call

ROOT = Path(__file__).resolve().parents[1]
OUT = '/mnt/data/bench-eval/roadmodel/validation/compile-reference-parallel-20261030-r2'
DEPLOY = '/mnt/data/bench-eval/roadmodel/deployment/compile-reference-parallel-20261030-r2'


def program():
    source = (ROOT / 'tools/tune_reference_parallel_remote.py').read_text()
    source = source.replace("for index in indices:assert not (c.c.ROOT / 'frames' / f'frame_{index:06d}').exists()", "for index in indices:\n        root=c.c.ROOT/'frames'/f'frame_{index:06d}'\n        assert (root/'prepare.json').exists() and not (root/'execution.json').exists()")
    source = source.replace("os.environ['ROADMODEL_OPT_VALIDATION'] = str(out)", "original=Path('/mnt/data/bench-eval/roadmodel/validation/compile-reference-parallel-20261030-r1')\n    os.environ['ROADMODEL_OPT_VALIDATION']=str(original)")
    source = source.replace('engine = c.Engine(); jobs = []; times = []; baseline_start = time.monotonic()', "engine=SimpleNamespace(onnx=c.c.ROOT/'frames/frame_002159/reference_fp16.onnx',close=lambda:None);jobs=[];times=[]")
    start = source.index("        exporter = c.make_pipeline(")
    end = source.index('        cases = ', start)
    source = source[:start] + '''        manifests={}
        for path in (original/'canonical').rglob('frame_manifest.json'):
            meta=load_json(path)['meta'];manifests[(meta['scene_name'],meta['frame_idx'])]=path.parent
        for index in indices:
            root=c.c.ROOT/'frames'/f'frame_{index:06d}'
            prepared=load_json(root/'prepare.json');assert prepared['state']=='READY'
            times.append(prepared['preparation_seconds'])
            jobs.append(dict(root='',baseline_root=str(root),cache_frame=str(manifests[tuple(c.FRAMES[index])]),task_root=str(original),onnx_fp16_sha256=prepared['onnx_fp16_sha256']))
''' + source[end:]
    source = source.replace('baseline_new_ONNX_frames=4,', 'baseline_new_ONNX_frames=0, baseline_reused_frames=4,')
    compile(source, 'resumed-reference-comparison', 'exec')
    return source


def start():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        tar.add(ROOT / 'pcm_eval/reference_workers.py', arcname='reference_workers.py')
        raw = program().encode(); info = tarfile.TarInfo('compare.py'); info.size = len(raw); tar.addfile(info, io.BytesIO(raw))
    return call("""import pathlib,sys,json,tarfile,io,subprocess,os
out=pathlib.Path(%r);d=pathlib.Path(%r);assert not out.exists() and not d.exists()
old=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/compile-reference-parallel-20261030-r1');pid=json.loads((old/'dispatch.json').read_text())['pid'];p=pathlib.Path('/proc')/str(pid)/'cmdline'
assert not p.exists() or not p.read_bytes(),'Original comparison must exit'
assert not list(pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/compile-reference-parallel-20261030-r1/workers-2').glob('frame*')),'Inspect partial candidate work'
d.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  p=(d/m.name).resolve();p.relative_to(d);p.write_bytes(tar.extractfile(m).read())
with (d/'comparison.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,str(d/'compare.py'),str(out)],cwd=d,env={**os.environ,'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'state':'RESUMED_COMPARISON_WITH_RETAINED_BASELINES','pid':p.pid,'new_baseline_inferences':0,'private_candidate_forwards':8,'native_inference_frames':0};(d/'dispatch.json').write_text(json.dumps(r));print(json.dumps(r))
""" % (OUT, DEPLOY), stream.getvalue())


def status():
    return call("""import pathlib,json
out=pathlib.Path(%r);d=pathlib.Path(%r);r={}
for n in ('progress.json','verification.json'):
 if (out/n).exists():
  x=json.loads((out/n).read_text())
  for c in x.get('cases',[]):c.pop('frames',None)
  r[n]=x
r['log_tail']=(d/'comparison.log').read_text(errors='replace')[-1800:];print(json.dumps(r))
""" % (OUT, DEPLOY))


if __name__ == '__main__':print({'start':start, 'status':status}[sys.argv[1]]())
