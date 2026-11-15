import json,re,tarfile
from pathlib import Path
from model_space_transport import put

root=Path('evidence/model-evaluation/model-reference-20261027')
bind=json.loads((root/'model-input-binding.json').read_text())
source=Path('inspection/release-model-complete')/bind['code']['evaluation_commit']
# Only variables read by source configuration are eligible; execution overrides are explicit.
keys=set()
for p in (source/'config').glob('*.py'):
 keys.update(re.findall(r"os\.(?:environ\.get|getenv)\(\s*['\"]([A-Za-z_][A-Za-z_0-9]*)",p.read_text(encoding='utf-8')))
env={k:str(v) for k,v in bind['training_environment'].items() if k in keys}
base='/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027'
env.update(bind['first_validation_overrides'])
env.update(DATASET_PATH=base+'/input',DATA_LINK_FILE=base+'/input/native-two-frame-data-link.json',
 DATASET='SmartChassisDatasetV7',NUM_FRAMES='1',FRAME_STRIDE='1',AUG_PROB='0.0',
 CKPT_SKIP_KEYS='',CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
 PYTHONUNBUFFERED='1',PYTHONPATH=base+'/source',TMPDIR='/tmp')
assert not any('$' in v for v in env.values()),[(k,v) for k,v in env.items() if '$' in v]
config={'schema':'roadmodel.native-model-launch/v1','remote_root':base,'source_commit':bind['code']['evaluation_commit'],
        'environment':env,'argv':['python3','eval.py','--py-config','config/3-0-0-temporal-dit.py',
        '--work-dir',base+'/run-01','--resume-from',base+'/weights/epoch_44.pth',
        '--classifier-resume',base+'/weights/epoch_98.pth','--sample-freq','1','--zip-vis','none'],
        'expected_frames':2,'selected_frames':bind['dataset']['selected_frames'],'enabled_metrics':['occ','pillar'],
        'disabled_metrics':{'vision':'The selected V161 OCC datalink provides no frame-level classification GT.',
                            'trajectory':'Initial smoke explicitly uses HEAD_MODE=gs_only.'}}
p=root/'native-launch.json';p.write_text(json.dumps(config,indent=2)+'\n')
print('launch_uploaded',put(p,base+'/native-launch.json'))
print(json.dumps({'environment_keys':len(env),'overrides':bind['first_validation_overrides'],'frames':2}))
