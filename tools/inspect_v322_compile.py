"""Read archive metadata and ONNX interfaces without extracting or executing payloads."""
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import digest, save_json
ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'evidence/model-evaluation/model-reference-20261027'
out = {'version': 'v3.2.2', 'archives': []}
for path in sorted(base.glob('*v3.2.2*.tar.gz')):
    item = {'path': str(path), 'sha256': digest(path), 'metadata': {}, 'members': []}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            item['members'].append({'name': member.name, 'size': member.size, 'type': str(member.type)})
            if member.isfile() and member.size < 1000000 and member.name.endswith(('.json','.yaml','.txt','.cfg')):
                data = archive.extractfile(member).read()
                try: value = json.loads(data)
                except (ValueError,UnicodeError): value = data.decode('utf-8',errors='replace')
                item['metadata'][member.name] = value
    out['archives'].append(item)
try:
    import onnx
    model = onnx.load(str(base / 'v322.onnx'), load_external_data=False)
    out['onnx'] = {'sha256': digest(base / 'v322.onnx'),
                   'inputs': [str(x) for x in model.graph.input], 'outputs': [str(x) for x in model.graph.output]}
except ImportError: out['onnx_inspection'] = 'onnx not installed in this local interpreter'
target = ROOT / 'evidence/feishu-entry-20261029/v322-inventory.json'
save_json(target,out,overwrite=target.exists())
print(json.dumps({'output':str(target),'archives':[{'path':Path(x['path']).name,'members':len(x['members']),
    'metadata':list(x['metadata']),'elf':x['metadata']['./model_info.json'].get('elf_path'),
    'inputs':len(x['metadata']['./model_info.json'].get('input',[])),
    'outputs':len(x['metadata']['./model_info.json'].get('output',[])),
    'elf_files':[y['name'] for y in x['members'] if y['name'].endswith('.elf')]
    } for x in out['archives']], 'onnx_inspection':out.get('onnx_inspection','done')}, ensure_ascii=False))
