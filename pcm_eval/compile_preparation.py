"""Validate installed compiler artifacts without contacting a bench."""
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from .common import digest, json_digest, load_json, save_json
from .subnets import chain_contract


def prepare_compile(job, target):
    target = Path(target)
    model = (job.get('request') or {}).get('model', {})
    signature = json_digest({'version': job['version'], 'model': model})
    receipt_file = target/'preparation.json'
    if receipt_file.exists():
        prior = load_json(receipt_file)
        if prior['request_sha256'] != signature:
            expected = [(model.get('onnx'), model.get('onnx_sha256'))]
            expected.extend((r['path'], r['sha256']) for r in model.get('subnet_archives', []))
            actual = [(r['path'], r['sha256']) for r in prior.get('artifacts', [])]
            if prior['version'] != job['version'] or expected != actual or not prior.get('contract'):
                raise ValueError('编译材料准备目录已绑定其他请求')
        # Recheck the artifacts, rather than trusting an old ready flag.
        for item in prior.get('artifacts', []):
            if digest(Path(item['path'])) != item['sha256']:
                raise ValueError('已准备的编译材料发生变化')
        return prior
    started = time.monotonic()
    request = job.get('request') or {}
    model = request.get('model', {})
    artifacts, issues, infos = [], [], {}
    try:
        if not model.get('onnx') or not model.get('subnet_archives'):
            raise ValueError('交付尚未绑定本版ONNX与五分网编译包的本地校验路径')
        records = [{'path': model['onnx'], 'sha256': model.get('onnx_sha256'), 'role': 'onnx'}]
        archives = model['subnet_archives']
        if len(archives) != 5 or {r['stage'] for r in archives} != set(range(5)):
            raise ValueError('本版五分网包必须唯一覆盖sub0至sub4')
        records.extend({**r, 'role': 'subnet_archive'} for r in archives)
        for item in records:
            path = Path(item['path'])
            if not path.is_file() or digest(path) != item.get('sha256'):
                raise ValueError('材料缺失或哈希不一致：'+path.name)
            artifacts.append({'path': str(path), 'sha256': item['sha256'], 'role': item['role']})
            if item['role'] != 'subnet_archive':
                continue
            with tarfile.open(path, 'r:gz') as archive:
                members = archive.getmembers()
                matches = [m for m in members if PurePosixPath(m.name).name == 'model_info.json' and m.isfile()]
                if len(matches) != 1 or matches[0].size > 4*1024*1024:
                    raise ValueError('编译包需包含唯一model_info.json')
                with archive.extractfile(matches[0]) as source:
                    info = json.load(source)
                # Metadata is parsed as data. Nothing is extracted or executed.
                infos[item['stage']] = info
        contract = chain_contract(infos)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        issues.append(str(error))
        contract = None
    receipt = {'schema': 'roadmodel.compile-preparation/v1', 'state': 'PREPARED' if not issues else 'BLOCKED',
               'request_sha256': signature, 'version': job['version'], 'artifacts': artifacts,
               'contract': contract, 'issues': issues, 'elapsed_seconds': time.monotonic()-started,
               'bench_contacted': False, 'inference_frames': 0}
    save_json(receipt_file, receipt)
    return receipt
