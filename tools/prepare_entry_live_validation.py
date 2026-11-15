"""Prepare exact tail-slot validation without modifying running full controllers."""
import io,json,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.planning import ssh_command
ROOT=Path(__file__).resolve().parents[1]
RELEASE='/mnt/data/bench-eval/service/roadmodel/entry-live-20261030-r3'
VALID='/mnt/data/bench-eval/roadmodel/validation/entry-live-20261030-r3'
FULL='/mnt/data/bench-eval/roadmodel/formal/pcm-formal-v322-20261029-01'
COMPILE='/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01'
PY='/mnt/data/developer/pcm-roadmodel-eval/venv/bin/python'

def main():
    source=(ROOT/'tools/formal_model_controller.py').read_text()
    derived=source
    edits=[("OUT / 'progress.json'","Path(os.environ['ROADMODEL_VALIDATION_ROOT']) / 'model/progress.json'"),
           ("'expected_OCC_frames': 8337","'expected_OCC_frames': 37"),
           ('for index, start in enumerate(range(0, 8337, 100)):','for index, start in [(83, 8300)]:'),
           ("launch['environment']['CUDA_VISIBLE_DEVICES'] = selected_gpu(OUT.parent)","launch['environment']['CUDA_VISIBLE_DEVICES'] = os.environ['ROADMODEL_VALIDATION_GPU']")]
    for old,new in edits:
        assert derived.count(old)==(3 if old=="OUT / 'progress.json'" else 1),old
        derived=derived.replace(old,new)
    compile(derived,'entry_model_controller.py','exec')
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as t:
        for folder in ('pcm_eval','configs','schemas'):
            for p in (ROOT/folder).glob('*'):
                if p.is_file():t.add(p,arcname=p.relative_to(ROOT).as_posix())
        for name in ('formal_model_stage_remote','formal_model_native_remote','formal_gpu','model_space_transport'):
            p=ROOT/'tools'/str(name+'.py');t.add(p,arcname='tools/'+p.name)
        t.add(ROOT/'tools/entry_tail_compile.py',arcname='tools/entry_tail_compile.py')
        t.add(ROOT/'tools/run_entry_pipeline.py',arcname='tools/run_entry_pipeline.py')
        for p in [ROOT/'evidence/model-evaluation/model-reference-20261027/native-launch.json',ROOT/'evidence/model-evaluation/remaining-heads-20261027/vision-launch.json']:
            t.add(p,arcname=p.relative_to(ROOT).as_posix())
        raw=derived.encode();m=tarfile.TarInfo('tools/entry_model_controller.py');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    code="""import pathlib,sys,tarfile,io,json,os,hashlib
release=pathlib.Path(%r);valid=pathlib.Path(%r);full=pathlib.Path(%r);compiled=pathlib.Path(%r)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  p=(release/m.name).resolve();p.relative_to(release);raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw) if not p.exists() else None
sys.path[:0]=[str(release),str(release/'tools')]
from pcm_eval.common import save_json,load_json,digest
assert not (full/'model/batch-0083-a02').exists(),'Tail model attempt already exists; do not repeat'
assert not (compiled/'compile/frames/frame_008336').exists(),'Tail board attempt already exists; do not repeat'
assert load_json(full/'model/progress.json')['completed_OCC_frames']<=8300
assert load_json(compiled/'compile/progress.json')['completed_frames']<=8336
os.environ['ROADMODEL_IDENTITY']='/mnt/data/bench-eval/service/gateway/runtime/roadmodel/model-space.key'
from model_space_transport import run
from formal_gpu import PROBE
inventory=run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
candidates=[(int(line.split(',')[1]),int(line.split(',')[0])) for line in inventory.splitlines() if line.strip() and int(line.split(',')[0])!=4]
free,index=max(candidates);assert free>=10240,{'no_other_free_GPU_MiB':free}
memory=json.loads(run(['python3','-c',PROBE,str(index)]).decode());assert memory['free_GiB']>=10,memory
proof=json.loads(run(['python3','-c',"import pathlib,json;b=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/formal-20261029/occ/batch-0083-a02');print(json.dumps({'exists':b.exists()}))"]).decode());assert not proof['exists']
request=load_json(full/'request.json');cat=load_json(release/'configs/two_link_catalog.json');cat['material_root']=str(release)
import numpy as np
ds=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/datasets/smartchassisdata-v3/26-10-20-1');data=load_json(ds/'data_link_26-10-20-1_train859_eval135_v7.json')
scene,frame=request['frames'][-1];item=next(f for f in data['eval'][scene]['frames'] if f['frame_index']==frame);a=np.load(ds/scene/'dataset'/str(frame)/item['occ_file'],allow_pickle=False)
keep=(a[:,0]<400)&(a[:,2]>=50)&(a[:,2]<150);assert int((a[keep,4]==1).sum())>0,'Tail GT must support the original drivable metric'
for kind,frames in [('model_accuracy',request['frames'][8300:]),('compile_alignment',request['frames'][8336:])]:
 r=dict(request,task_id='pcm-entry-live-20261030',frames=frames,expected_frames=len(frames))
 rp=save_json(valid/(kind+'-request.json'),r)
 prepared=save_json(valid/(kind+'-preparation.json'),{'state':'VERIFIED','native_task_id':r['task_id'],'frozen_request_sha256':digest(rp),
      'source_full_request_sha256':digest(full/'request.json'),'selected_frames':frames,'already_executed_frames':0,'native_tail_slot_empty':True})
 script=release/'tools'/('entry_model_controller.py' if kind=='model_accuracy' else 'entry_tail_compile.py')
 env={'ROADMODEL_TASK_ROOT':str(full),'ROADMODEL_STAGE_LOCAL':'1','ROADMODEL_IDENTITY':os.environ['ROADMODEL_IDENTITY'],
      'ROADMODEL_VALIDATION_ROOT':str(valid),'ROADMODEL_VALIDATION_GPU':str(index),'PYTHONUNBUFFERED':'1'}
 launch={'preparation_receipt':{'path':str(prepared),'sha256':digest(prepared)},'programs':[{'role':'model_controller' if kind=='model_accuracy' else 'compile_controller','script':str(script),'sha256':digest(script),'python':%r,'cwd':str(release),'env':env}]}
 binding={'schema':'roadmodel.active-execution/v1','kind':kind,'version':'v3.2.2','dataset_id':cat['datasets'][0]['id'],
      'native_task_id':r['task_id'],'model_delivery_url':cat['models'][0]['url'],'dataset_url':cat['datasets'][0]['url'],
      'execution_root':str(valid),'expected_frames':len(frames),'frozen_request':{'path':str(rp),'sha256':digest(rp)},'vision_required':False,
      'launch':launch,'delivery_receipt':str(valid/'publication/verification.json'),
      'source_full_task':str(full),'report_scope':'selected_validation','result_source':str(full/'model/batch-0083-a02') if kind=='model_accuracy' else str(compiled/'compile/frames/frame_008336')}
 path=save_json(release/('active/'+kind+'.json'),binding)
 cat['models'][0]['chains'][kind]={'version':'v3.2.2','dataset_id':binding['dataset_id'],'active_execution':{'path':str(path.relative_to(release)),'sha256':digest(path)}}
save_json(release/'configs/two_link_catalog.json',cat,overwrite=True)
print(json.dumps({'state':'PREPARED','model_frames':37,'compile_frames':1,'model_GPU':memory,'new_inference_frames':0,'full_workers_unchanged':True}))
""" % (RELEASE,VALID,FULL,COMPILE,PY)
    p=subprocess.run(ssh_command('ubuntu-workstation',[PY,'-c',code]),input=stream.getvalue(),capture_output=True,timeout=100)
    print(p.stdout.decode());print(p.stderr.decode());sys.exit(p.returncode)

if __name__=='__main__':main()
