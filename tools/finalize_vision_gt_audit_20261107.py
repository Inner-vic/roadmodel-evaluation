"""Save the evidence conclusion and deploy only the conservative diagnosis fix."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/vision-gt-audit-20261107'
mounted=json.loads((OUT/'mounted-evidence.json').read_text(encoding='utf-8'))
tags=json.loads((OUT/'tag-contract.json').read_text(encoding='utf-8'))
template=json.loads((ROOT/'configs/report_template.json').read_text(encoding='utf-8'))
required={x['key'] for x in template['model_accuracy']['vision_tasks']}
findings={
 'dataset_version':'26-10-20-1','index_sha256':mounted['index_sha256'],
 'previous_success':{'dataset_version':'26-08-14-0','index':'DataReorganized-VisionHead-standard/data_link_vision_3230train_577eval_20260821.json','dataset_class':'SmartChassisDatasetV6'},
 'current_index':{'vision_eval_scenes':135,'vision_eval_frames':0,'eval_scenes':135,'eval_frames':8337,'explicit_img_label_file_references':0},
 'physical_inspection':{'scenes':6,'frame_directories':12,'frame_classification_JSON_found':0,'scene_tags_found':6,'full_recursive_inventory_performed':False},
 'annotation_scope':'Scene-level vlm_tags_list and label_stats exist; direct per-frame evaluation binding is absent in checked frames.',
 'last_two_frame_check':next(r['last_two_frame_check'] for r in tags['rows'] if 'last_two_frame_check' in r),
 'report_tasks_missing_from_scene_tags':sorted(required-set(tags['rows'][0]['vlm_tags_list'])),
 'mapping_conflicts':[{'scene':r['scene'],'differences':r['scene_label_vs_frame_histogram_differences']} for r in tags['rows']],
 'native_loader':{'source_sha256':tags['actual_native_source']['sha256'],'class':'SmartChassisDatasetV7','missing_label_reference_behavior':'returns empty img_class; no scan or scene-tag fallback'},
 'confirmed_pipeline_omission':'Did not inspect scene-level tags before declaring the data source missing GT.',
 'confirmed_execution_blocker':'No indexed vision validation frames and no verified mapping from each evaluated image to task labels.',
 'not_proven':'No claim that all annotation files are absent from every directory in the dataset.',
 'required_contract':'Provide/confirm same-version per-frame classification labels and vision validation membership; resolve scene-vs-frame conflicts and the missing task fields. No automatic broadcast of vlm_tags or inferred per-frame labels from histograms.',
 'inference_frames':0,'dataset_changed':False,'original_report_metrics_changed':False}
(OUT/'conclusion.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf-8')
code=(ROOT/'pcm_eval/lpai_native_model.py').read_bytes()
new='''                # Empty indexed coverage is not proof that annotations do not
                # exist elsewhere (for example scene-level tags.json).
                if kind == 'vision':
                    incomplete.append('vision：本次索引 vision_eval 未列出验证帧；尚未建立可用于计分的帧级GT绑定')
                else:
                    incomplete.append(kind+'未列出本次验证帧；GT可用性待核验')'''.encode()
old="                incomplete.append(kind+'缺少本数据源的验证帧/GT')".encode()
assert code.count(new)==1
expected=hashlib.sha256(code.replace(new,old)).hexdigest()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tar:
    for name in ('pcm_eval/lpai_native_model.py','tests/test_lpai_native_model.py'):
        tar.add(ROOT/name,arcname=name)
    raw=json.dumps(findings).encode();member=tarfile.TarInfo('conclusion.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
receipt=remote('''import pathlib,sys,json,hashlib,tarfile,io,ast,shutil,subprocess,sqlite3
root=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-compile-offline-20261107-r4')
stage=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/vision-gt-audit-20261107');stage.mkdir(exist_ok=True)
assert not (stage/'diagnosis-deployment.json').exists()
expected=EXPECTED
path=root/'pcm_eval/lpai_native_model.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Inspect concurrent source changes first'
with sqlite3.connect('/mnt/data/bench-eval/state/roadmodel/evaluation/runtime.sqlite3') as db:
 assert not db.execute("select 1 from tasks where state in ('QUEUED','PREPARING','READY','RUNNING','WAITING_REPORT')").fetchone(),'Do not modify active evaluation code'
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar:
  assert member.isfile();raw=tar.extractfile(member).read()
  if member.name=='conclusion.json':(stage/member.name).write_bytes(raw);continue
  target=(root/member.name).resolve();target.relative_to(root);ast.parse(raw)
  backup=stage/'before'/member.name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
  target.write_bytes(raw)
p=subprocess.run([sys.executable,'-m','unittest','tests.test_lpai_native_model','-q'],cwd=root,capture_output=True,text=True,timeout=45)
(stage/'diagnosis-tests.log').write_text(p.stdout+p.stderr);assert p.returncode==0,(p.stdout+p.stderr)[-1500:]
receipt={'state':'DEPLOYED','scope':'Only the missing-index diagnosis changed; execution, labels and metrics unchanged','tests':(p.stdout+p.stderr)[-400:],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'new_inference_frames':0,'report_metrics_changed':False}
(stage/'diagnosis-deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('expected=EXPECTED','expected='+repr(expected)),stream.getvalue(),timeout=70)
(OUT/'diagnosis-deployment.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'conclusion':str(OUT/'conclusion.json'),'deployment':receipt}))
