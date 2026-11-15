"""Compare required public tensors against intermediate diagnostic downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from connect_active_entry import call
CODE=r'''import pathlib,json,datetime,re
task=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/formal-runs/pcm-formal-v322-20261029-01')
p=json.loads((task/'compile/progress.json').read_text());root=task/'compile/frames'/f"frame_{p['completed_frames']-1:06d}"
public=json.loads((root/'public-outputs.json').read_text());allout=json.loads((root/'all-subnet-outputs.json').read_text())
names={x['tensor'] for x in public};required=[dict(x,producer_stage=allout[x['tensor']]['stage']) for x in public]
diagnostic=[x for name,x in allout.items() if name not in names]
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'progress':p,'public':required,
 'intermediate_only':diagnostic,'counts':{'all':len(allout),'public':len(public),'intermediate_only':len(diagnostic)},
 'bytes':{'all':sum(x['nbytes'] for x in allout.values()),'public':sum(x['nbytes'] for x in required),'intermediate_only':sum(x['nbytes'] for x in diagnostic)},
 'retained_public_tensors':json.loads((root/'tensor-comparison.json').read_text())['tensor_count']}
package=pathlib.Path('/mnt/data/developer/pcm-roadmodel-eval/package/benchEval-skill-20261020')
out['package_files']=[x.name for x in package.iterdir() if x.is_file()]
for name in ('board_infer_client.py','board_pipeline.py','run_bench_eval.py','onnx_localagg_backend.py','README.md','SKILL.md'):
 file=package/name
 if not file.exists():continue
 source=file.read_text(errors='replace');lines=source.splitlines()
 out[name]=[{'line':i+1,'text':line} for i,line in enumerate(lines) if any(w in line for w in ('def ','class ','InferenceSession','deploy_packup','ensure_pack','subprocess.run','run_localagg_cpu','fetch_extra','fetch_gaussian','for frame','run_inference','packup_deploy','复用','多帧'))][:100]
summary={'checked_at':out['checked_at'],'counts':out['counts'],'bytes':out['bytes'],
 'public_MiB':out['bytes']['public']/1024**2,'all_output_full_GiB':out['bytes']['all']*8337/1024**3,
 'public_output_full_GiB':out['bytes']['public']*8337/1024**3,'intermediate_byte_fraction':out['bytes']['intermediate_only']/out['bytes']['all'],
 'GT_needed_is_public_subset':True,'no_inference_repeated':True,'running_controller_modified':False,
 'finding':'The five-subnet integration downloads all intermediate feature maps for diagnostics. Official18 tensor comparisons and the6 Gaussian GT inputs only consume public outputs. Intermediate tensors remain necessary on board for downstream stages, but their raw download is not required by the report metrics.',
 'guard_preservation':'Move size/hash/finite checks of intermediates to the board and return summaries; retain original18 tensor comparisons and GT computations. This has not been deployed or full-frame validated.'}
evidence=pathlib.Path('/mnt/data/bench-eval/roadmodel/deployment/entry-20261030/compile-timing-audit')
receipt=evidence/'flow-audit.json';assert not receipt.exists();receipt.write_text(json.dumps({'summary':summary,'public':required,'intermediate_only':diagnostic,'source_review':{k:v for k,v in out.items() if k.endswith('.py')}},ensure_ascii=False,indent=2))
out={k:v for k,v in out.items() if k not in ('public','intermediate_only')};out['summary']=summary
print(json.dumps(out,ensure_ascii=False))
'''
if __name__=='__main__':print(call(CODE))
