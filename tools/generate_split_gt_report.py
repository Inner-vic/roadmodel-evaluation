"""Offline atomic report generation from a completed bounded diagnostic JSON."""
import argparse,hashlib,json,sys,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.model_report_split_gt import split_gt_section
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
a=json.loads(args.analysis.read_text(encoding='utf-8'));dest=args.output;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'report.xml').exists()
report={'schema':'roadmodel.split-gt-report/v1','analysis':a,'generation':{'existing_artifacts_only_during_generation':True,'source_new_inference_frames':a['file_collection']['new_inference_frames'],'new_inference_frames_during_generation':0,'new_training_runs':0,'PCM_release_template_used':False}}
(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
root=args.analysis.parent
names=['analysis.json','manifest-audit.json','effective-splits.json','preprocessing-config-audit.json','gt-sample-plan-corrected.json','selection-annotation-correction.json','gt-sample-receipt.json','gt-sample-analysis.json','aligned-semantic-metrics.json','source-evidence.json','object-timestamps.json']
files=[root/n for n in names];files.append(args.analysis);inputs=json.loads((root/'gt-sample-receipt.json').read_text(encoding='utf-8'))
files.extend(Path(f['local_path']) for f in inputs['files'])
files.extend(Path('tools')/n for n in ['audit_split_manifest.py','inspect_split_preprocessing.py','collect_split_gt_samples.py','analyze_split_gt_samples.py','align_retained_semantic_metrics.py','finalize_split_gt_audit.py','generate_split_gt_report.py'])
files.append(Path('pcm_eval/model_report_split_gt.py'))
files.extend(p for p in (root/'model-control-01').rglob('*') if p.is_file())
files.extend([root/'model-control-verification.json',root/'two-frame-input-receipt.json',root/'model-control-collection.json'])
files.extend(Path(x['path']) for x in a['example_images'])
files.extend(Path('tools')/n for n in ['run_split_control_remote.py','verify_split_control.py','finalize_split_gt_control.py'])
manifest=[]
with zipfile.ZipFile(dest/'evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
    for i,f in enumerate(files):
        b=f.read_bytes();assert len(b)<16*1024*1024
        n=f'{i:02d}-{f.name}';z.writestr(n,b);manifest.append({'path':n,'source':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
xml=split_gt_section(a)
for f,label in [(dest/'report.json','训练验证数据与GT核查结果.json'),(dest/'evidence.zip','训练验证GT样本配置与核查证据.zip')]:xml+='<source path="@./'+f.as_posix()+'" name="'+label+'"/>'
ET.fromstring('<body>'+xml+'</body>');(dest/'report.xml').write_text(xml,encoding='utf-8')
print(json.dumps({'state':'generated','files':len(files),'evidence_bytes':(dest/'evidence.zip').stat().st_size,'new_inference_during_generation':0,'source_forward_frames':a['file_collection']['new_inference_frames'],'new_training_runs':0}))
