"""Generate an atomic training-audit report from retained JSON; never runs models."""
import argparse,hashlib,json,sys,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.model_report_training import training_semantic_section
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
a=json.loads(args.analysis.read_text(encoding='utf-8'));dest=args.output;dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'report.xml').exists(),'Preserve prior generated report'
root=args.analysis.parent
report={'schema':'roadmodel.training-semantic-report/v1','analysis':a,'generation':{'existing_artifacts_only':True,'new_training_runs':0,'new_inference_frames':0,'PCM_release_template_used':False}}
(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
files=[root/n for n in ['analysis.json','epoch-summary.csv','source-evidence.json','task-log-scope.json','semantic-receipt.json','original-output-inventory.json','checkpoint-object-binding.json']]
files.extend((root/'original-output').rglob('*'));files=[f for f in files if f.is_file() and f.suffix!='.pth']
files.extend(Path('tools')/n for n in ['parse_training_events.py','summarize_training_semantics.py','finalize_training_semantic_audit.py','generate_training_semantic_report.py'])
files.append(Path('pcm_eval/model_report_training.py'))
manifest=[]
with zipfile.ZipFile(dest/'evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
    for i,f in enumerate(files):
        raw=f.read_bytes();assert len(raw)<16*1024*1024
        name=f'{i:02d}-{f.name}';z.writestr(name,raw);manifest.append({'path':name,'source':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
xml=training_semantic_section(a)
for f,label in [(dest/'report.json','原训练任务语义核查结果.json'),(dest/'evidence.zip','原训练日志配置与语义核查证据.zip')]:xml+='<source path="@./'+f.as_posix()+'" name="'+label+'"/>'
ET.fromstring('<body>'+xml+'</body>');(dest/'report.xml').write_text(xml,encoding='utf-8')
print(json.dumps({'state':'generated','output':str(dest),'evidence_bytes':(dest/'evidence.zip').stat().st_size,'new_training_runs':0,'new_inference_frames':0}))
