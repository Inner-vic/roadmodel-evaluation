"""Bind memory probes to the same existing physical GPU as native evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json

PROBE = "import os,sys;os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[1];import torch,json;f,t=torch.cuda.mem_get_info(0);print(json.dumps({'physical_GPU':sys.argv[1],'free_GiB':f/1024**3,'total_GiB':t/1024**3}))"


def selected_gpu(task, fallback='1'):
    path = Path(task) / 'model/gpu-selection.json'
    if not path.exists():
        return str(fallback)
    data = load_json(path)
    index = data['physical_index']
    if type(index) is not int or index not in data['visible_indices'] or not 0 <= index < 8:
        raise ValueError('GPU must be an existing visible device in this space')
    return str(index)
