"""Replace the stale example with the actually exercised OCC configuration."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcm_eval.common import digest,json_digest

base=ROOT/'evidence/model-evaluation/native-v3.2.2-smoke-20261027'
launch=json.loads((base/'native-launch.json').read_text())
binding=json.loads((ROOT/'evidence/model-evaluation/model-reference-20261027/model-input-binding.json').read_text())
source=json.loads((base/'source-manifest.json').read_text())
env=launch['environment']
data_link=ROOT/'evidence/model-evaluation/model-reference-20261027/two-frame-input/native-two-frame-data-link.json'
value={'schema':'pcm.eval-request/v1','kind':'model_accuracy','version_name':'v3.2.2',
 'source':{'repository':binding['code']['repository'],'commit_id':source['commit'],'require_verified_binding':True,
           'snapshot_manifest':launch['remote_root']+'/source-manifest.json','snapshot_manifest_sha256':digest(base/'source-manifest.json'),
           'configuration_env_sha256':json_digest(env),'model_reference_url':binding['model_reference_url'],
           'note':'Verified OCC-only small-scope request. Vision and trajectory have independent native launch contracts.'},
 'execution':{'transport':'ssh','ssh_alias':'lpai','python':'python3','repo_dir':launch['remote_root']+'/source',
              'output_root':launch['remote_root']+'/jobs','env':{},'configuration_env':env},
 'dataset':{'class_name':env['DATASET'],'path':env['DATASET_PATH'],'data_link_file':env['DATA_LINK_FILE'],'data_link_sha256':digest(data_link)},
 'model':{'config':binding['code']['config'],'config_sha256':source['files'][binding['code']['config']],
          'checkpoint':launch['remote_root']+'/weights/epoch_44.pth','checkpoint_sha256':binding['main_checkpoint']['sha256'],
          'classifier_checkpoint':launch['remote_root']+'/weights/epoch_98.pth','classifier_checkpoint_sha256':binding['classifier_checkpoint']['sha256']},
 'evaluation':{'frame_fraction':1.0,'max_frames':2,'expected_frames':2,'compute_scheme':False,'temporal':False,
               'heads':['occ'],'scene_names':list(dict.fromkeys(x[0] for x in launch['selected_frames']))}}
path=ROOT/'examples/model_request.json'
path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
profiles=ROOT/'configs/evaluation_profiles.json'
data=json.loads(profiles.read_text())
data['profiles']['model-v322-small'].update(request_file='examples/model_request.json',execution_heads=['occ'])
profiles.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':'frozen','request':str(path),'execution_heads':['occ'],'model_version':'v3.2.2'}))
