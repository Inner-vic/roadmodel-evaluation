"""Bind a freshly collected model result to the frozen request, never to a historical run ID."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re

from .common import load_json, save_json, digest


def prepare_model_result(job, receipt, output):
    result = deepcopy(load_json(receipt['result_file']))
    request = job['request']
    selected = request['evaluation'].get('selected_frames')
    if result['coverage']['observed'].get('model_samples') is None:
        console = Path(receipt['result_file']).parent/'model_inference.log'
        text = console.read_text(encoding='utf-8',errors='replace')
        counts = re.findall(r'\[SceneFilter\].*?,\s*\d+\s*(?:→|->)\s*(\d+)\s+samples',text)
        completed = re.findall(r'100%\|[^\r\n]*?\|\s*(\d+)/(\d+)',text)
        if (receipt.get('state')!='succeeded' or not counts or not completed
                or len(set(counts))!=1 or int(counts[-1])!=len(selected or [])
                or any(int(a)!=int(b) or int(a)!=len(selected or []) for a,b in completed)):
            raise ValueError('Successful execution console does not prove selected-frame completion')
        result['coverage']['observed']['model_samples']=int(counts[-1])
        result['collection']['issues']=[x for x in result['collection']['issues'] if x['code']!='unknown_model_coverage']
        result['artifacts'].append({'path':str(console.resolve()),'role':'completed_inference_console',
                                    'size_bytes':console.stat().st_size,'sha256':digest(console)})
        result['provenance']['coverage_derivation']={'source':str(console),'sha256':digest(console),
            'method':'successful frozen execution + SceneFilter sample count + completed progress counter'}
    if not selected or len(selected) != result['coverage']['observed'].get('model_samples'):
        raise ValueError('Frozen selected frames do not match observed inference count')
    if len(selected) != request['evaluation']['expected_frames']:
        raise ValueError('Frozen selected-frame count differs from expected count')
    weights = {}
    for key in ('checkpoint','classifier_checkpoint','trajectory_checkpoint'):
        if request['model'].get(key): weights[Path(request['model'][key]).name] = request['model'][key+'_sha256']
    result['provenance']['native_run'] = {
        'run_id':receipt['run_id'], 'kind':'occ', 'source_commit':request['source']['commit_id'],
        'checkpoint_sha256':weights, 'selected_frames':selected,
        'environment':request['execution']['configuration_env'],
        'duration_seconds':(datetime.fromisoformat(receipt['finished_at'])-datetime.fromisoformat(receipt['started_at'])).total_seconds(),
        'data_link_sha256':request['dataset']['data_link_sha256'],
        'binding_derivation':'successful frozen execution plan and its pinned two-frame datalink'}
    save_json(output,result)
    return str(Path(output).resolve())
